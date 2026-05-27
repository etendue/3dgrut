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

## Vast.ai 远程执行环境（A800 占用时备用）

A800 被其他任务占用时，**Claude 可以自行起 vast.ai RTX 4090 实例**跑 V3 smoke / KPI。整套流程已在 2026-05-27 V3-L5/L8/L9 任务中跑通（详见 v3_plan.md § 5 Done Log "V3-L5 + V3-L8 + V3-L9" 条目）。

- vastai CLI: `/Users/etendue/repo/ncore/.venv/bin/vastai`（v1.0.3）
- API key: 写死在 `scripts/t8_12_fix_vast_create.sh` 里（也接受 `--api-key $VAST_API_KEY`）
- HF token: `~/.cache/huggingface/token`（如需从 HF 下数据）
- 推荐 image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`（vast 上 image pull 约 5-10 min on 8 Gbps host）
- 推荐 host: California / Norway 等 inet > 1 Gbps 的节点；**避开 France `mid=67891`**（image pull 卡 25+ min 不动）
- 5k smoke A/B 成本：~$0.45（45 min × $0.534/hr RTX 4090 48GB）

### 阶段 1：创建实例 + ssh config

```bash
# 起实例（自动选最便宜 RTX 4090, 写 ~/.ssh/config 别名 vast-rtx4090）
LABEL=v3_<task>_smoke DISK_GB=100 MAX_DPH=0.80 \
  bash scripts/t8_12_fix_vast_create.sh
```

⚠️ **t8_12_fix_vast_create.sh 已知 bug**：脚本里的 status polling 用了旧版字段名解析（'ssh_host' 字符串 substring 模糊匹配）会卡在"timed out waiting"。实例其实是创建好的，**手动从 `vastai show instance <id> --raw` 拿 `ssh_host` / `ssh_port` 字段自己写 `~/.ssh/config`**：

```python
# Python 写入 Mac 端 ~/.ssh/config (本地 Mac 用 python3 OK)
python3 - <<'PY'
import re, os
path = os.path.expanduser("~/.ssh/config")
alias = "vast-rtx4090"
with open(path) as f: txt = f.read()
pat = re.compile(rf"(^|\n)Host\s+{re.escape(alias)}\b.*?(?=\nHost\s|\Z)", re.S)
txt = pat.sub("", txt)
txt = txt.rstrip() + "\n\nHost vast-rtx4090\n    HostName ssh<N>.vast.ai\n    Port <PORT>\n    User root\n    IdentityFile ~/.ssh/id_ed25519\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n"
with open(path, "w") as f: f.write(txt)
os.chmod(path, 0o600)
PY
```

### 阶段 2：环境安装

不要在 vast 容器内跑 python3 来写文件（vast pytorch container **没装 python3 系统命令**，只有 conda python；用纯 shell + awk 或 scp 推 Mac 写好的脚本）：

```bash
# 在 vast 上 clone repo + 跑 install_env_uv.sh
ssh vast-rtx4090 'apt-get install -y -qq git python3.11-venv rsync \
    libxcb1 libxext6 libxrender1 libsm6 libice6 libgl1 libglib2.0-0 \
  && cd /root && git clone https://github.com/etendue/3dgrut.git \
  && cd 3dgrut && git checkout <branch> \
  && git submodule update --init --recursive \
  && bash install_env_uv.sh'
```

**必装的系统 lib**（opencv-python 在 headless container 缺这些）：`libxcb1 libxext6 libxrender1 libsm6 libice6 libgl1 libglib2.0-0`。否则 `import threedgrut.datasets` 会 ImportError on `libxcb.so.1`。

**install_env_uv.sh 用 `tail -30` buffer 上游输出**，看起来"卡住"实际在跑——通过 `ps -ef | grep nvcc` 或 `du -sh .venv` 监控真实进度。10-15 min 完成；slangc + kaolin 在末尾装。

### 阶段 3：数据传输（A800 → vast 反向推）

A800 上的 NCore 数据完整（`/root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/`），**优先从 A800 推到 vast 而不是从 HF 重下**（A800 速度 3-5 MB/s, 7.2 GB clip ~23-25 min）：

```bash
# 1. A800 上生成 ssh keypair（如果没有）
ssh a800-x2 '[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "a800@v3-rsync"'
A800_PUB=$(ssh a800-x2 'cat ~/.ssh/id_ed25519.pub')

# 2. 把 A800 pubkey 加入 vast authorized_keys（avoid 黏行：用 printf '\n%s\n'）
ssh vast-rtx4090 "mkdir -p ~/.ssh && chmod 700 ~/.ssh \
  && printf '\n%s\n' '$A800_PUB' >> ~/.ssh/authorized_keys \
  && chmod 600 ~/.ssh/authorized_keys"

# 3. A800 上写 ssh config 指向 vast（用 awk 不用 python3, A800 root 可能没 python3）
ssh a800-x2 'bash -s' <<EOF
SSH_CFG=~/.ssh/config
awk 'BEGIN{skip=0} /^Host vast-rtx4090\b/{skip=1; next} skip==1 && /^Host /{skip=0} skip==1{next} {print}' "\$SSH_CFG" > "\$SSH_CFG.tmp"
cat <<CFG >> "\$SSH_CFG.tmp"

Host vast-rtx4090
    HostName ssh<N>.vast.ai
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
CFG
mv "\$SSH_CFG.tmp" "\$SSH_CFG"
chmod 600 "\$SSH_CFG"
EOF

# 4. 从 A800 push to vast (run_in_background=true; ~25 min for 7.2 GB)
ssh a800-x2 'rsync -avz --info=progress2 \
  /root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/ \
  vast-rtx4090:/root/data/ncore/clips/<clip>/'
```

### 阶段 4：训练启动

```bash
# 启动脚本走 nohup, 用 scp 推 launcher 而不是 inline ssh heredoc
# （vast container bashrc 可能有 set -e, inline heredoc 在 pkill 没匹配时会中断）

cat > /tmp/launch_smoke.sh <<'EOF'
#!/bin/bash
rm -f /tmp/v3_smoke.log /tmp/v3_smoke.pid
cd /root/3dgrut
git pull 2>&1 | tail -3
nohup bash scripts/v3_l589_vast_smoke_ab.sh > /tmp/v3_smoke.log 2>&1 &
echo "$!" > /tmp/v3_smoke.pid
sleep 10; head -30 /tmp/v3_smoke.log
EOF
scp -q /tmp/launch_smoke.sh vast-rtx4090:/tmp/launch_smoke.sh
ssh vast-rtx4090 'bash /tmp/launch_smoke.sh'
```

### Hydra override 严格区分 `+` vs `++`

- `+key=value`：**新增** key（key 不存在）。若 yaml 已有此 key 会报 `Could not append to config. An item is already at '<key>'`。
- `++key=value`：**override** key（key 存在与否都 OK）。**通用且不报错**。
- 不带前缀：只能 override 已有 key（如顶层 `n_iterations=5000`）。

V3-L589 的 5k smoke 首次失败因为用 `+layers.overrides.dynamic_rigids.<key>=...` 但 multilayer.yaml 已经有默认值——必须 `++` 才能 override。**所有 `layers.overrides.<layer>.<field>` 类的 CLI override 一律用 `++`**。

### 销毁实例（任务完成立即清理）

```bash
echo y | /Users/etendue/repo/ncore/.venv/bin/vastai destroy instance <ID> \
  --api-key a6d4a47d11507fec636572f4ba555a1cb2395864eac29f33035eb1bcd5712f0d
```

不要让实例闲置——RTX 4090 即使空闲也按 $0.534/hr 计费。

### Monitor 使用注意

跑训练 + eval 时，**不要让 Monitor grep "PSNR"**——render.py eval 阶段会逐帧打 `Frame N, PSNR: X`，几千帧会让 Monitor 被 rate limit suppression 干掉。Monitor 只 grep **关键节点**：`RUN [0-9]:|⭐ Test Metrics|^=== |Traceback|FAILED|OOM|⚡ V3-L8|🎊 Training Statistics`。

## 训练配置约定（重要）

**v2 多层训练统一使用 `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`**，这是 dynfix 7 层递归链 (`dynfix → 4dviz → exposure → sky → full → road → ncore_3dgut_mcmc`) 的字节等价扁平版（Hydra compose 递归 diff 0 差异，A800 1k smoke 验证通过，详见 v2_plan.md § 5 Done Log 2026-05-26 "Config 重构"条目）。

**后续所有训练（smoke / 5k / 30k / 全量）默认用 multilayer**：

```bash
# A800 标准启动（sky_backend=mlp 必须，nvdiffrast 不可用）
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=<N> \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/pai_<clip>.json \
    trainer.sky_backend=mlp \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=<name>'
```

- **不要再用旧的 7 层 v2 yaml**（`v2_full_4dviz_dynfix` 等）起新训练；旧 yaml 仅保留供历史 commit 复现，新工作一律 multilayer。
- **camera_ids 5-cam ring 内置**：Stage 7 需 7-cam 时 CLI 覆盖 `'dataset.camera_ids=[...]'`，其他 override 直接 CLI 加在 multilayer 之上。
- **exposure 默认 ON**（与 dynfix 等价），如做 ablation/复跑 baseline 加 `trainer.use_exposure=false`（详见 v2_plan.md § 14.5 V3-P1）。
- **dataset path 必须是 manifest json 文件**（`pai_<clip>.json`），不是 clip 目录。

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
