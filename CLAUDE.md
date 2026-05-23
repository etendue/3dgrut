# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

通用项目说明（架构 / 安装 / 训练命令 / 数据集 / 导出格式）见
[README.md](README.md)；本文件只保留 **Claude 在此仓库工作时必须知道的项目特定约定**。

## A800 远程执行环境（重要）

Claude 可以**直接通过 `ssh a800-x2` 访问 A800 GPU 主机**执行训练 / 集成测试等 GPU 任务。不需要把"A800 上的任务"标记为待用户手动操作；不需要等待用户确认。所有 v2 plan 里写"A800 探测 / 5k smoke / 30k KPI"等任务都是 Claude 直接 ssh 跑。

- 远程主机别名：`a800-x2`（~/.ssh/config 已配置）
- 远程仓库路径：`/root/work/yusun/repo/3dgrut`
- 数据集路径：`/root/work/yusun/ncore-nurec/data/ncore/clips/...`
- 输出路径：`/root/work/yusun/ncore-nurec/output/...`
- 推荐 GPU：`export CUDA_VISIBLE_DEVICES=0` 或 `1`，看 `nvidia-smi` 选空闲卡
- 内存配置：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### ⚠️ conda env 激活（每次 ssh 必须）

**ssh non-interactive shell 不继承 conda PATH**——直接 `conda activate 3dgrut` 或 `python ...` 会报 `conda: command not found` / `python: command not found`。每个 `ssh a800-x2` 命令开头都必须先 source conda init：

```bash
ssh a800-x2 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut && cd /root/work/yusun/repo/3dgrut && python -c "..." '
```

或一次性导 env PATH（跑训练 / pytest 推荐，让 slangc 等 env 内工具都可见）：

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH && cd /root/work/yusun/repo/3dgrut && python train.py ...'
```

历史踩坑（v2_plan.md L950-955）：仅设 `CUDA_VISIBLE_DEVICES` 但不 source conda init 会触发 `FileNotFoundError: 'slangc'`（slangc 装在 env 内）。

可用 conda envs（`conda info --envs`）：`base`（miniforge3）/ `3dgrut`（主开发环境）/ `drivestudio` / `j6`（~/.bashrc 默认）。

执行模式：用 `Bash` 工具单条 `ssh a800-x2 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut && ...'` 心跳；长任务（≥ 5 min）用 `run_in_background=true`，Claude 会被自动通知完成；用 `tee /tmp/<run>.log` 同步 grep "PSNR" / 错误。

**触发同步 plan/architecture 时机不变**：远程任务跑完后，把实际 PSNR / commit hash / iter speed 等写回 v2_plan.md § 5 Done Log + v2_architecture.md § 7 关键不变量。

### A800 操作 + 文档同步 严格把关清单（防资源浪费）

跑 A800 训练一次成本 5-20 min + GPU 时间。代码不对 / 路径错位 / metric 没接通的情况下重跑就是双倍浪费。**每次 ssh a800-x2 跑训练前 + 文档标 ✅ 前必须严格走以下清单**：

**A. 远程代码就绪验证**（rsync 后、bash 跑训练前）：
1. `ssh a800-x2 "grep -n '<本次改动关键字符串>' /root/work/yusun/repo/3dgrut/<file>"` —— 确认改动真的同步到了远端，不只是 rsync 报"sent N bytes"。
2. `ssh a800-x2 "head -25 /root/work/yusun/repo/3dgrut/render.py"` —— 入口脚本 head 应该是 `import argparse` + `if __name__ == "__main__":`，**不是** `import json` / `import torch` 那种包内模块风格。本仓库历史踩坑：包内 `threedgrut/render.py` 曾被错放到顶层 `render.py`，导致 `python render.py` silent exit 0 + 没输出 + 没产物，浪费 1 次 A800 GPU 时间。
3. rsync 命令的目标路径要带尾部 `/`（rsync 把内容放进目标目录，而不是创建同名子目录）；如果改动多文件，宁可走目录级 rsync 而不是逐文件传，避免误把单文件丢到错误层级。
4. 远端工作仓库不是 git clone 而是 rsync mirror：`ssh a800-x2 "cd /root/work/yusun/repo/3dgrut && git log --oneline -1"` 看到的是非常老的 commit 是**正常的**（rsync 不动 .git），不要因此误以为代码没同步——而要直接 grep 关键字符串验证。

**B. 训练→eval→metrics 链路完整性验证**（避免"训练成功但 metric 没接通"的伪完成）：
5. 任何"新增 metric"任务（如 T6F.2 双指标）改完 `trainer.compute_metrics` 必须**同时**核查 `threedgrut/render.py` 的 eval loop（独立路径，单独写 `metrics.json`）—— 两处不同步会导致 metrics.json 里没有新字段，而 5k smoke 已经跑完。本仓库历史踩坑：T6F.2 只改 trainer 没改 render.py，第一次 A800 smoke 没有 `psnr_masked` 字段。**写本类任务的 plan 时必须列出 trainer.py + render.py 两处改动点**。
6. 跑完后 `ssh a800-x2 "cat <out_dir>/metrics.json"` 必须看到所有期望的新 key，否则 task 状态保持 🟡 不能标 ✅。
7. 训练日志最后的 `🎊 Training Statistics` + `⭐ Test Metrics` 两表都要看到。如果只有训练表没有测试表，说明 eval 没跑——多半是 `n_iterations` 未达 `val_frequency` 或 `test_last=false`。

**C. 文档标 ✅ 把关**（A800 出口数据回填）：
8. **A800 任务（T*.a / T*.b 出口 / Stage X 出口）的 ✅ 必须以 metrics.json 实测数据 + commit hash 双重证据为前提**。Mac 本地完成的任务可以标 ✅ + 备注"(Mac)"，但 A800 出口任务 ✅ 必须含实测数字（PSNR / SSIM / it/s）写入 v2_plan.md § 5 Done Log。
9. 把"伪完成"识别为"未完成"：训练 exit 0 + ckpt 写出 ≠ task ✅。必须 metric 数字达标 + 写进 Done Log + commit hash 入看板。
10. 跑挂了（exit ≠ 0 / 早期 RuntimeError）回头改代码时，**先写一个回归测试 pin 住这个 case**（如 4D vs 3D mask broadcast），再修代码 + Mac pytest，最后再 rsync + 重跑 A800。不要"改了就直接重跑 A800"，单测便宜，A800 贵。

## v2 开发工作流（分层高斯）

v2 分层高斯（LayeredGaussians）工作以两份活文档驱动：
- `v2_plan.md` —— Kanban 看板 + 任务级状态 + Done Log
- `v2_architecture.md` —— 模块/流程差异图（v1 vs v2）+ 文件清单

### 任务完成后必须同步更新这两份文档

**每个 task（按 v2_plan.md 中 T*.* 编号）执行完成并 commit 后，必须在同一个或紧随的 commit 中更新：**

1. **`v2_plan.md`**：
   - 把对应任务在 `1.1 顶层看板` 中从 Backlog/In Progress 移到 Done，标 ✅
   - 在 `1.2 任务级看板` 表格中把状态列从 ⬜/🟡 改为 ✅，"改动 / 新增"列填实际 commit 短 hash
   - 在 `1.3 当前 Stage 状态汇总` 更新完成数
   - 在 `5. Done Log` 追加一条：日期 + commit hash + 实际改动摘要 + 关键验收数据（PSNR / 测试数 / 耗时等）

2. **`v2_architecture.md`**：
   - 如果 task 新增/修改了模块或文件：在对应 mermaid 图中把该节点的 classDef 从 `:::todo` 改为 `:::done`，并在节点 label 中追加 commit 短 hash
   - 在 `1.3 模块级 diff 摘要` 或 `6.1/6.2 文件清单` 表格中把该任务从 ⬜ 标为 ✅
   - 不影响架构的纯测试任务（如 T1.4）可以只在 `7. 关键不变量` 表格中加一行验证锚点，无需改图

### 不更新文档 = 任务未完成

代码 commit 通过但文档未同步会让看板与实际状态脱节，导致后续任务依赖判断错误。把"更新 plan/architecture"视为 task 的 Step N+1（在跑测试和 commit 代码之后），与代码改动写在同一个 commit message 中：

```
feat(T1.2): <一句话改动>

<代码改动说明>

docs(plan): mark T1.2 done in v2_plan.md kanban + Done Log
docs(arch): flip T1.2 nodes to :::done in v2_architecture.md
```

或拆成两个相邻的 commit 也可，但必须在 push/合并前完成。
