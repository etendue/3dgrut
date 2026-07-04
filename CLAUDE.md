# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

通用项目说明（架构 / 安装 / 训练命令 / 数据集 / 导出格式）见
[README.md](README.md)；本文件只保留 **Claude 在此仓库工作时必须知道的项目特定约定**。

## A800 远程执行环境（重要）

Claude 可以**直接通过 `ssh a800-x2` 访问 A800 GPU 主机**执行训练 / 集成测试等 GPU 任务。不需要把"A800 上的任务"标记为待用户手动操作；不需要等待用户确认。plan（现以 [`v3_plan_revised.md`](v3_plan_revised.md) 为准）里写"A800 探测 / 5k smoke / 30k KPI"等任务都是 Claude 直接 ssh 跑。

- 远程主机别名：`a800-x2`（~/.ssh/config 已配置）
- 远程仓库路径：`/root/work/yusun/repo/3dgrut`
- 数据集路径：`/root/work/yusun/ncore-nurec/data/ncore/clips/...`
- 输出路径：`/root/work/yusun/ncore-nurec/output/...`
- 推荐 GPU：`export CUDA_VISIBLE_DEVICES=0` 或 `1`，看 `nvidia-smi` 选空闲卡
- 内存配置：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### ⚠️ conda env 激活（每次 ssh 必须）

**ssh non-interactive shell 不继承 conda PATH**——直接 `conda activate 3dgrut2` 或 `python ...` 会报 `conda: command not found` / `python: command not found`。每个 `ssh a800-x2` 命令开头都必须先 source conda init：

```bash
ssh a800-x2 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2 && cd /root/work/yusun/repo/3dgrut && python -c "..." '
```

或一次性导 env PATH（跑训练 / pytest 推荐，让 slangc 等 env 内工具都可见）：

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut2/bin:$PATH && cd /root/work/yusun/repo/3dgrut && python train.py ...'
```

历史踩坑（v2_plan.md L950-955）：仅设 `CUDA_VISIBLE_DEVICES` 但不 source conda init 会触发 `FileNotFoundError: 'slangc'`（slangc 装在 env 内）。

可用 conda envs（`conda info --envs`）：`base`（miniforge3）/ **`3dgrut2`（主开发环境）** / `j6`（~/.bashrc 默认）。

> ⚠️ **2026-06-29 实测环境漂移（CLAUDE.md 旧值已失效，本条为准）**：
> - **A800 conda env 已从 `3dgrut` 改名为 `3dgrut2`**（旧 `3dgrut` / `drivestudio` 不再存在）。所有 `ssh a800-x2` 命令一律 `conda activate 3dgrut2` / `export PATH=/root/miniforge3/envs/3dgrut2/bin:$PATH`。
> - **rsync mirror 严重过期**：`/root/work/yusun/repo/3dgrut` 的 `.git` 仍停在 `50d997e`（pre road/layer，2026-05 量级），跑新代码前**必须**把改动目录 rsync 到 HEAD（`threedgrut/ configs/ scripts/ threedgrut_playground/`），并用 `grep` 关键字符串验证（不能信 `git log`，rsync 不动 `.git`）。
> - `set -u`（严格模式）会让 `conda activate 3dgrut2` 因 unbound `NVCC_PREPEND_FLAGS` 报错——A800 上跑 conda activate 的命令**不要带 `set -u`**。
> - DataLoader `num_workers=24` 在 A800 上偶发 `pin_memory ConnectionRefusedError`；smoke 跑不通时先降 `num_workers` 排除。

执行模式：用 `Bash` 工具单条 `ssh a800-x2 'source /root/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2 && ...'` 心跳；长任务（≥ 5 min）用 `run_in_background=true`，Claude 会被自动通知完成；用 `tee /tmp/<run>.log` 同步 grep "PSNR" / 错误。

**触发同步 plan/architecture 时机不变**：远程任务跑完后，把实际 per-class PSNR / LPIPS / commit hash / iter speed 等写回 [`v3_plan_revised.md`](v3_plan_revised.md) § 6 Done Log + [`v2_architecture.md`](v2_architecture.md) § 7 关键不变量。

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
8. **A800 任务（P*.* 出口 / Phase X 出口）的 ✅ 必须以 metrics.json 实测数据 + commit hash 双重证据为前提**。Mac 本地完成的任务可以标 ✅ + 备注"(Mac)"，但 A800 出口任务 ✅ 必须含实测数字（per-class PSNR / LPIPS / SSIM / it/s）写入 [`v3_plan_revised.md`](v3_plan_revised.md) § 6 Done Log。
9. 把"伪完成"识别为"未完成"：训练 exit 0 + ckpt 写出 ≠ task ✅。必须 metric 数字达标 + 写进 Done Log + commit hash 入看板。
10. 跑挂了（exit ≠ 0 / 早期 RuntimeError）回头改代码时，**先写一个回归测试 pin 住这个 case**（如 4D vs 3D mask broadcast），再修代码 + Mac pytest，最后再 rsync + 重跑 A800。不要"改了就直接重跑 A800"，单测便宜，A800 贵。

## inceptio 本地 GPU 执行环境（RTX 4090，首选备用机）

A800 占用时，**Claude 可直接 `ssh inceptio` 使用本地 RTX 4090**（24GB VRAM）跑训练 / smoke / KPI。

- 主机别名：`inceptio`（~/.ssh/config 已配置，IP 10.8.31.113）
- 用户：`inceptio`
- 仓库路径：`~/repo/3dgrut2/`
- 数据路径：`~/work/data/<clip>/`、`~/ncore_data/`
- 输出路径：`~/work/output/`
- GPU：RTX 4090 24GB，Driver 590，CUDA 13.1（conda env 内用 cu128）

### ⚠️ conda env 激活（每次 ssh 必须）

```bash
ssh inceptio 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2 && cd ~/repo/3dgrut2 && python ...'
```

或导 PATH：

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && cd ~/repo/3dgrut2 && python train.py ...'
```

### ⚠️ 首次运行注意（已完成，记录备查）

1. **系统无 g++**：inceptio 未装 `build-essential`，通过 conda 安装并建软链接解决（已完成）：
   ```bash
   # 已执行，无需重复
   conda install -n 3dgrut2 -y -c conda-forge gxx_linux-64 gcc_linux-64
   ln -sf x86_64-conda-linux-gnu-c++ ~/miniforge3/envs/3dgrut2/bin/c++
   ln -sf x86_64-conda-linux-gnu-g++ ~/miniforge3/envs/3dgrut2/bin/g++
   ```
2. **VGG16 感知损失模型**：位于 `~/data/torch_cache/hub/checkpoints/vgg16-397923af.pth`，已复制到 `~/.cache/torch/hub/checkpoints/`（已完成）。
3. **JIT 编译**：首次已编译缓存至 `~/.cache/torch_extensions/py311_cu128/{lib3dgut_cc,lib_mcmc_cc}`，后续启动秒过。

### ⚠️⚠️ num_workers 必须按机器「系统内存」调（OOM 头号坑，2026-06-05 实测定位）

**`configs/base_gs.yaml` 默认 `num_workers: 24` 是为 A800（128 核 / 2TB 内存）调的。在小内存机器上直接用会 OOM**——被系统 OOM killer 杀掉（`dmesg`/`journalctl -k` 见 `Out of memory: Killed process python`），表现为训练跑到 ~iter 5000（约 40 min）后**静默退出无 Traceback**（SIGKILL 不留 Python 栈）。

**根因（实测 PSS 分解坐实，不是泄漏）**：
- 内存 ≈ **主进程堆（~13GB）+ num_workers × ~2-3GB**。每个 DataLoader worker 是 fork 出的数据集副本，Python 引用计数写对象头 → copy-on-write 被打破 → 每 worker 私有化 ~2GB。
- **2026-06-03 起 v3 baseline 配方打开了 LiDAR 深度监督**（`ncore_3dgut_mcmc_multilayer.yaml`: `load_lidar_depth_map=true` / `use_lidar_depth=true`），每帧多加载/解码一张深度图（1920×1080 float32 ≈ 8MB，按 256/worker 缓存）→ **per-worker 内存比旧配方重**（这就是为什么 5/27 旧 ThinkPad run 在 31GB 跑得下、而现在 62GB 反而 OOM 的差异）。
- 内存会**预热 ~13 min 后平台**（如 nw=10 稳定 39GB），**不是线性泄漏**；`RSS` 会因共享页被重复计数而虚高（11 进程 sum RSS 165GB 是假象），**以 `PSS`（`/proc/<pid>/smaps_rollup`）或 `free -g` used 为准**。

**经验值（深度监督 ON 时）**：
| 机器 | 系统内存 | 安全 num_workers | 实测 |
|---|---|---|---|
| A800 | 2TB | 24（默认） | OK |
| inceptio / 62GB | 62GB | **≤ 10**（→ ~39GB） | nw=24 OOM，nw=10 稳 |
| 32GB 机 | 32GB | ≤ 4-6 | — |

**用法**：小内存机一律 CLI 覆盖 `num_workers=10`（顶层 key，直接覆盖不带 `+`）。

> ⚠️ **2026-06-09 大g 决策（覆盖旧「不要关 use_lidar_depth」指令）**：**inceptio 上训练默认关 LiDAR depth + DepthAnythingV2 深度监督**（`use_lidar_depth=false` / `use_depth_prior=false` / `load_lidar_depth_map=false`），因为 inceptio 62GB 内存装不下深度图缓存——既是 num_workers OOM 的内存大头，深度图加载也是数据管线瓶颈、加了 depth 训练明显变慢。
>
> **原则（大g）**：**A800 内存多 → 加 depth 约束（lidar-on）；inceptio 内存不够、加 depth 又拖慢 → depth-off，并同步调小 `num_workers`。** 两台机各自统一配方，A/B 不跨机混搭。
> - **baseline 与实验一律统一 depth-off** → inceptio **内部** A/B 仍单变量可比。
> - 但与 A800 lidar-on 立锚的数字（如 cc_psnr_masked 25.79 / lane grad_corr 0.693 / 车 class_psnr 24.04）**不可直接跨机比** → 在 inceptio 上要先重立一条 **depth-off baseline 锚点**，A/B 以它为对照。
> - 旧「不要为省内存关 `use_lidar_depth`，否则 A/B 不可比」那句**仅在 A800（2TB 内存）成立**，inceptio 不适用。

**附带现象（同一根因）**：RTX 4090 上 GPU 利用率只有 ~30-50%（不是 100%），因为深度图 + sseg 的**数据管线喂不满快卡**（数据加载瓶颈，非 GPU 慢）。降 num_workers 省内存的代价就是数据并行度更低、GPU 更饿；这是内存 vs 速度的权衡，**不影响训练正确性 / 最终 metric**。本地 NVMe 已排除慢盘因素。

**OOM 防线**：跑长训练前可挂一个 RAM guard（每 30s 采 `free -g`，超阈值 `pkill` + 记日志），避免静默 OOM 浪费 40 min（2026-06-05 用过 `/tmp/p1_2_ramguard.sh`）。

**长任务启动用「proven inline nohup」模式**（`ssh inceptio "... && nohup python train.py ... > log 2>&1 & echo PID \$!"`），inceptio ssh 偶发抖动（exit 255）；`setsid bash 脚本 & disown` 这种复杂形式在抖动下容易半路夭折，inline nohup + 末尾 `echo PID` 最稳。

> ⚠️ **2026-06-22 实测补充（嵌套 driver 脚本场景，部分覆盖上面说法）**：跑「driver 脚本内部再起 train.py + eval」这类**嵌套长任务**时，上面的 inline nohup **会让 ssh 挂起超时 → exit 255 且任务没起来**（子进程继承 ssh 的 stdin/stdout fd，ssh 等 fd 关闭直到超时）。实测**唯一稳的启动式** ＝ `setsid` + 切断三个 fd + ssh `-n`：
> ```bash
> ssh -n inceptio 'cd ~/repo/3dgrut2-wt/<wt> && setsid bash scripts/<driver>.sh > /tmp/<run>.log 2>&1 < /dev/null & echo PID_$!'
> ```
> 四点缺一就 255/夭折：① `setsid` 脱离会话；② `< /dev/null` 切 stdin；③ `> log 2>&1` 切 stdout/err；④ ssh `-n`。**与上面「setsid 易夭折」旧说法相反**——旧说法对单条 `train.py` 成立，嵌套 driver 脚本必须 setsid。配套四个实测坑：① **高频 ssh 触发 sshd 限流**（连发多次全 255，单次冷却 ~30s 后恢复）→ ssh 之间留间隔、一次连接里做尽量多事；② **`pkill -9` 杀占大显存的进程会瞬时卡断当前 ssh**（255 真凶，反复踩）→ 别在启动命令里带 pkill，先单独清理或确认 GPU 空再启动；③ **`base64 -d` 内联传长脚本的命令易 255** → 传脚本走 **git**（`git push inceptio` + worktree `git reset --hard`）最稳，`scp` 次之；④ **`nvidia-smi` 偶发是 ssh 断点**，查 GPU/进程改用 `ps`/`pgrep`（注意 `pgrep -f` 会自匹配含关键字的 ssh 命令行本身，用 `[p]ython` 方括号技巧规避误报）。

### 🌐 访问外网（mihomo 代理）

inceptio 直连外网受限；要拉 HF 模型 / pip / git clone 等联网操作需先起 mihomo 代理：

```bash
# 1. 启动代理（后台），监听 127.0.0.1:7890
ssh inceptio 'cd /home/inceptio/repo/mihomo_p && nohup ./mihomo -f mihomo.yaml > /tmp/mihomo.log 2>&1 & echo PID $!'

# 2. 联网命令前在同一条 ssh 内导出代理 env（纯本地训练不需要）
ssh inceptio 'export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 && <要联网的命令>'
```

- 代理路径：`/home/inceptio/repo/mihomo_p/`，配置文件 `mihomo.yaml`。
- **只在需要联网的命令前导出 `http_proxy`/`https_proxy`**；数据/权重已在本地的训练不要带代理（避免本地回环也走代理）。

### 代码同步（Mac → inceptio）

```bash
rsync -az --exclude='.claude/worktrees' --exclude='.venv' --exclude='__pycache__' \
  /Users/etendue/repo/3dgrut2/ inceptio:~/repo/3dgrut2/
```

### ⭐ inceptio git worktree 工作流（推荐 — 每任务隔离，2026-06-10 大g 决策）

**A800 不稳（系统管理员清理 / conda env 丢 / 数据没上盘，已多次）→ 后续 GPU 任务一律转 inceptio。** 每个任务在 inceptio 的 git 仓库下 checkout 一个独立 git worktree 跑，完成即删——代码版本明确、多任务互不污染、不怕被清。inceptio `~/repo/3dgrut2` 是**真 git 仓库**（非 rsync mirror），`git worktree` 可用。

**一次性建 Mac→inceptio 的 git remote（本地 ssh，不走 GitHub/代理）**：
```bash
cd /Users/etendue/repo/3dgrut2 && git remote add inceptio inceptio:/home/inceptio/repo/3dgrut2
```

**每个任务的流程**：
```bash
# 1. Mac：push 任务分支到 inceptio（inceptio 在 main，push 非当前分支不冲突）
git push inceptio <branch>:<branch>

# 2. inceptio：从该分支 checkout 独立 worktree（仓库外路径 ~/repo/3dgrut2-wt/<task>）
ssh inceptio 'cd ~/repo/3dgrut2 && git worktree add ~/repo/3dgrut2-wt/<task> <branch>'

# 2b. ⚠️ 必做：补 submodule —— git worktree add **不 checkout submodule**，但
#     lib3dgut_cc JIT 编译需 thirdparty/tiny-cuda-nn（缺 → "Error building extension
#     'lib3dgut_cc'"）。从主仓库 rsync 过来（不需联网，inceptio 主仓库已有 submodule）：
ssh inceptio 'cd ~/repo/3dgrut2; WT=~/repo/3dgrut2-wt/<task>; for p in $(git config --file .gitmodules --get-regexp path | cut -d" " -f2); do rsync -a ~/repo/3dgrut2/$p/ $WT/$p/; done'

# 3. inceptio worktree 里跑（cd 进 worktree → import 自动解析 worktree 代码、非主仓库）
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && cd ~/repo/3dgrut2-wt/<task> && python train.py ... '

# 4. 完成：删 worktree
ssh inceptio 'cd ~/repo/3dgrut2 && git worktree remove ~/repo/3dgrut2-wt/<task>'
```

**注意**：
- 验证代码同步：`ssh inceptio 'cd ~/repo/3dgrut2-wt/<task> && git log --oneline -1'` 应 = Mac 分支 head。
- inceptio .git 可能含 Mac rsync 带来的 prunable worktree 引用 → `git worktree prune` 清理。
- inceptio 跑训练**一律 depth-off + `num_workers=10`**（内存铁律，见上）；与 A800 lidar-on 数字不可跨机比，须在 inceptio **重立 depth-off baseline 锚**再做 A/B。
- 后续改动：Mac 在 worktree 分支 commit → `git push inceptio <branch>` → inceptio worktree `git pull`（或 `git reset --hard origin/<branch>`），保持同步。

### 训练启动示例

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd ~/repo/3dgrut2 \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=5000 \
    path=~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json \
    trainer.sky_backend=mlp \
    out_dir=~/work/output \
    experiment_name=smoke_test'
```

### inceptio 无-aux NCore 数据：nre-tools 容器生成 aux（runbook + 坑，2026-07-01 实测）

新 inceptio ncore clip（只有 `*.ncore4-*.zarr.itar` + manifest、**无 aux/**）直接跑 4 层 multilayer 会崩在 road/dynamic 层 init（`trainer.py` → `get_road_lidar_points`/`get_dynamic_lidar_points` → `datasetNcore._get_semantic_lidar_points` 需 `aux.sseg`/`aux.lidar-sseg`）。**background 单层不需 aux 可直接跑**（基础 config `apps/ncore_3dgut_mcmc`，验证 camera/pose/pipeline）；4 层 multilayer 必须先生成 aux。

用**已在 inceptio 的 NGC 容器** `nvcr.io/nvidia/nre/nre-tools-ga:latest`（`docker images` 可见，docker 已登录 nvcr.io、`~/.ngc` 已配 apikey）。入口命令是 `ncore-aux-data`（**不带** `nre-tools` 前缀），**必须 `--gpus all`**（否则 `libcuda.so.1` 缺失 → 命令不注册）。**分两次跑**（见坑 1/2），`<D>` = 含 manifest+itar 的 clip 目录：

```bash
# run A：sseg + egomask（快 ~6min，多相机可 parallel），--no-lidar-seg-camvis 跳过慢的 lidar
docker run --rm --gpus all -v <D>:/workdir/data -v ~/.cache/torch:/home/.cache/torch \
  nvcr.io/nvidia/nre/nre-tools-ga:latest ncore-aux-data \
  --dataset-path=/workdir/data/<seq_stem>.json --output-dir=/workdir/data \
  --camera-id camera_front_wide_120fov --camera-id camera_cross_left_120fov --camera-id camera_cross_right_120fov \
  --segmentation-backend=mask2former --ego-mask --no-lidar-seg-camvis \
  --depth-backend=none --dinov2-backend=none --parallel-mode --workers-per-gpu=3 --zarr-store-type=itar --store-meta
# run B：lidar-seg（慢 ~5.8h 单核），--segmentation-backend=none 复用 run A 的 sseg
docker run --rm --gpus all -v <D>:/workdir/data -v ~/.cache/torch:/home/.cache/torch \
  nvcr.io/nvidia/nre/nre-tools-ga:latest ncore-aux-data \
  --dataset-path=/workdir/data/<seq_stem>.json --output-dir=/workdir/data \
  --camera-id camera_front_wide_120fov --camera-id camera_cross_left_120fov --camera-id camera_cross_right_120fov \
  --segmentation-backend=none --no-ego-mask --lidar-seg-camvis \
  --depth-backend=none --dinov2-backend=none --num-threads=8 --zarr-store-type=itar --store-meta
```

产物 `<seq_stem>.aux.{sseg,egomask,lidar-sseg,lidar-camvis}.zarr.itar` 落 clip 目录（= manifest 同级，datasetNcore 期望位置），root-owned（inceptio user 可读，训练 OK）。sseg 类别 = Cityscapes-20（模型 `mask2former_dinov2_nv_private12k`，权重内置容器、**无需联网**），与现有 multilayer 配方（road={0,1}/sky=10/dynamic={11-18}）天然对齐。

**四个实测坑（重踩浪费数小时，务必看）**：
1. **itar 中途绝不能 `docker stop`**：`.zarr.itar` 是 write-once，tar index header 在**整个 run 结束时**才写。半途停会让已完成的 sseg.itar 损坏 → 复用报 `IndexedTarStore: invalid index header, can't load indexed tar file`。要么一次跑完，要么按上面分两次（run A 完整 finalize 后 run B 才能 `--segmentation-backend=none` 复用）。
2. **lidar-seg 单核 ~105s/帧**（point-in-cameras + ensemble 是单线程 Python 逐点循环，GIL 锁死 1 核；GPU 0%、其余核闲、非 IO、内存充足——正是「哪都没饱和却很慢」）：串行 200 帧 ≈ 5.8h。`--lidar-seg-ensemble-cuda` 无效（瓶颈不在 ensemble），**容器内** `--parallel-mode --workers-per-gpu` 也无效（只并行**多 sensor**任务，单 lidar 的多帧不拆）。**解法 = 手动多 container 数据并行**（见下「⚡ lidar-seg 多 container 并行」小节，实测 12 段 5.8h→58min，6×，总 CPU 100%→1207%）。
3. **sseg 快 lidar-seg 慢 → 分两次**（坑 2 的应对）：run A 的 `--parallel-mode --workers-per-gpu=N` 对多相机 sseg **有效**（N ≤ 相机数；mask2former 每 worker ~4-6GB 显存，24GB 卡取 N≤3-4）；run B 复用 sseg 只补 lidar-seg。
4. **depth/dinov2 关掉提速**：depth-off 配方不需要，`--depth-backend=none --dinov2-backend=none` 省时间。

**⚡ lidar-seg 多 container 并行（实测 5.8h→58min / 6×，2026-07-01）**：坑 2 的正解。lidar-seg 每帧独立 + 显存才 ~1GB/容器 + 20 核闲 19 个 → 手动起 N 个 docker 各切一段 lidar 帧并行（容器本身不支持，手动做）。四个关键点，缺一即崩：
- **限帧**：`ncore-aux-data [aux-opts] sensor-frames --main-sensor-id lidar_top_360fov --start-frame X --stop-frame Y`。`sensor-frames` 子命令是「限帧范围 + 跑完整 aux」（不是切数据集）；start/stop 是 lidar 帧索引，stop 为 past-the-end（[X,Y)）。
- **复用全帧 sseg**：段内只跑 lidar（`--segmentation-backend=none --no-ego-mask --lidar-seg-camvis`），复用 run A 的**全帧** sseg——lidar-seg 的 point-in-cameras 用 lidar 帧时间戳查 camera sseg，段内自生成的 sseg 覆盖不到 → `KeyError: semantic segmentation not found`。
- **干净隔离**：每段一个独立目录（`ln -f <clip>/*.ncore4* $SEGi/` 硬链 raw + `cp` 全帧 `*.aux.sseg`/`*.aux.egomask` + manifest），output 各自目录。**绝不能共享 clip 目录**——并发容器互读对方半成品 itar 报 `invalid index header`。
- **合并**：N 段各出 `lidar-sseg`（0-D `|S<n>` PNG bytes）+ `lidar-camvis`（`(N_pts,1)` uint8 **数组**）itar → 合并脚本遍历 `/aux/<comp>/<sensor>/<ts>` 汇所有帧到一新 itar：`create_dataset(ts, shape=src.shape, dtype=src.dtype)` + `ds[...]=src[...]`（**用 src.shape 通用处理，别写死 `shape=()`**——camvis 是数组，写死 0-D 会 `ValueError: setting array element with a sequence`）。

N=12（≤20 核，~17 帧/段）实测各段 ~34min（12 段并行）、合并秒级、读回 200 帧正常 finalize → 串行 5.8h 降到 58min。全流程：run A 全帧 sseg+egomask（`--no-lidar-seg-camvis`，parallel N≤相机数）→ N 段并行 lidar-seg（限帧+复用 sseg+隔离）→ 合并 → multilayer 训练直接读。驱动 + 合并脚本模板见本次会话 scratchpad（`parallel_aux_train.sh` / `merge_lidar_aux.py`）。

## Vast.ai 远程执行环境（A800 占用时备用）

A800 被其他任务占用时，**Claude 可以自行起 vast.ai RTX 4090 实例**跑 V3 smoke / KPI。整套流程已在 2026-05-27 V3-L5/L8/L9 任务中跑通（详见 [`v3_plan.md`](v3_plan.md)（冻结历史，仅证据参考）§ 5 Done Log "V3-L5 + V3-L8 + V3-L9" 条目）。

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

### ⚡ 快速实验一律用 5 秒数据窗（2026-07-04 大g 决策）

**凡是机制验证 / 诊断 / A/B 快测（非正式锚点跑），一律把数据裁到 5 秒**，迭代速度约 ×4：

```bash
python train.py ... dataset.train.duration_sec=5.0 dataset.val.duration_sec=5.0
```

- 键位：`dataset.train.duration_sec` / `dataset.val.duration_sec`（`-1`=全量；配套 `seek_offset_sec` 选起始秒）。
- **快测数字与全量（20s）锚不可比**——5s 窗只用于「机制是否生效 / 相对好坏」判断；正式锚点、Done Log 入档数字仍须全量跑。
- **b6a9 clip 的车辆集中在后段**（前 ~10s 基本空路）：动态层相关实验用 `dataset.train.seek_offset_sec=15 dataset.train.duration_sec=5`（f185/19.9s 时段有 23 个活跃 cuboid）；road/bg 实验用默认前 5s 即可。
- 步数等比降：5s 窗帧数 ≈ 1/4，快测建议 5k-8k 步（30k 会过拟合小窗）。

**v2 多层训练统一使用 `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`**，这是 dynfix 7 层递归链 (`dynfix → 4dviz → exposure → sky → full → road → ncore_3dgut_mcmc`) 的字节等价扁平版（Hydra compose 递归 diff 0 差异，A800 1k smoke 验证通过，详见 v2_plan.md § 5 Done Log 2026-05-26 "Config 重构"条目）。**inceptio b6a9 多相机线用 `configs/apps/ncore_3dgut_mcmc_multilayer_inceptio.yaml`**（v5 任务A baseline：6-cam + use_opacity=false，锚数字见该 yaml 头注释）。

**后续所有训练（smoke / 5k / 30k / 全量）默认用 multilayer**：

```bash
# A800 标准启动（sky_backend=mlp 必须，nvdiffrast 不可用；env 现为 3dgrut2）
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut2/bin:$PATH \
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

### Optional — V3 pose_adjustment（学习 dynamic cuboid 的 pose）

默认 disable（v2 byte-identical）；启用方式：

```bash
# 方式 1：直接用 multilayer，CLI 打开
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
  trainer.pose_adjustment.enabled=true \
  trainer.pose_adjustment.lambda_t=1e-2 \
  trainer.pose_adjustment.lambda_r=1e-1 \
  ...

# 方式 2：用预设 yaml（lambdas 已固化为 DriveStudio 默认 1e-2 / 1e-1）
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer_poseopt \
  ...
```

三个用户接口：
- `trainer.pose_adjustment.enabled` — 总开关（默认 false）
- `trainer.pose_adjustment.lambda_t` — temporal smoothness 强度 (translation)
- `trainer.pose_adjustment.lambda_r` — temporal smoothness 强度 (rotation)

高级参数（lr / freeze_until_iter / pose-prior 占位）仍在 `trainer.learnable_pose.*` 内部 key，CLI 直接 override 即可（旧脚本也保留 backward-compat）。

## v3 开发工作流（actor-centric per-class）

v3 工作以 [`v3_plan_revised.md`](v3_plan_revised.md) 为**唯一执行依据**（actor-centric 重编号 plan，Phase 0–3 + asset-harvester，P*.* 任务编号）；架构差异图 / 文件清单 / 关键不变量仍维护在 [`v2_architecture.md`](v2_architecture.md)。

> **文档层级（按权威性，从高到低）**：
> - [`v3_plan_revised.md`](v3_plan_revised.md) —— **当前主线 plan**（Kanban + P*.* 任务级状态 + Done Log）。今后所有 v3 工作以本文档为准执行。
> - [`v2_architecture.md`](v2_architecture.md) —— 架构差异图（v1 vs v2/v3）+ 文件清单 + § 7 关键不变量。**v3 新模块继续在此登记**（如 V3-R1/R2 已登记 § 6 文件清单 + § 7 不变量）。
> - [`v3_plan.md`](v3_plan.md) —— **冻结的旧版 v3 阶梯**（错轴：novel-view PSNR≥30 全局主 KPI）。仅作历史 experience / Done Log 证据参考，**不再作为执行依据**。
> - [`v2_plan.md`](v2_plan.md) + 旧 7 层 yaml —— v2 分层高斯历史，仅供 commit 复现，不起新工作。

### 任务完成后必须同步更新文档

**每个 task（按 [`v3_plan_revised.md`](v3_plan_revised.md) 中 P*.* 编号）执行完成并 commit 后，必须在同一个或紧随的 commit 中更新：**

1. **`v3_plan_revised.md`**：
   - `§ 1.1 顶层看板（Mermaid Kanban）`：把任务从 Backlog/In Progress 移到对应列（Review/Done），标 ✅
   - `§ 1.2 任务级看板（P*.* 表格）`：状态列从 ⬜/🟡/🔵 改为 ✅，"改动/新增"列填实际 commit 短 hash
   - `§ 1.3 Phase 状态汇总 + per-class gap 表`：更新「任务数 (Done/Total)」；**Phase 0 实测后回填 per-class gap 三行真实数字**
   - `§ 6 Done Log`：追加一条 —— 日期 + commit hash + 实际改动摘要 + 关键验收数据（**per-class PSNR / LPIPS** / 测试数 / 耗时等）

2. **`v2_architecture.md`**（仅当 task 新增/修改模块或文件时）：
   - 在对应 mermaid 图中把该节点的 classDef 从 `:::todo` 改为 `:::done`，并在节点 label 中追加 commit 短 hash
   - 在 `§ 6.1/6.2 文件清单` 表格中把该任务从 ⬜ 标为 ✅
   - 不影响架构的纯测试任务可以只在 `§ 7 关键不变量` 表格中加一行验证锚点，无需改图

### 不更新文档 = 任务未完成

代码 commit 通过但文档未同步会让看板与实际状态脱节，导致后续任务依赖判断错误。把"更新 plan/architecture"视为 task 的 Step N+1（在跑测试和 commit 代码之后），与代码改动写在同一个 commit message 中（沿用项目 `docs(plan):` / `docs(arch):` 约定 + PR 号后缀 + Co-Authored-By Claude）：

```
feat(P1.2): <一句话改动>

<代码改动说明>

docs(plan): mark P1.2 done in v3_plan_revised.md kanban + Done Log
docs(arch): flip P1.2 nodes to :::done in v2_architecture.md
```

或拆成两个相邻的 commit 也可，但必须在 push/合并前完成。

## Mermaid 图表约定（防 `()` 解析报错 —— 已反复踩坑）

**铁律：mermaid 图（kanban / flowchart / 任何 diagram）的「节点 / 卡片标签」里，括号一律用全角 `（）`，绝不用半角 `()`。**

**原因**：mermaid 把半角 `(` 当作圆角节点语法 `id(text)` 的起始符，于是用 `()` 去「聚集 / 包住一段字符串」时解析器会把它误读成节点形状声明而报错（`Parse error` / `got 'PS'` 之类）。本仓库的 `v3_plan_revised.md`（看板 + 依赖图）、`v2_architecture.md`（架构图）反复因此渲染失败。

会炸 / 不会炸的边界：
- ❌ **kanban 卡片** `[文字]`（不带引号）—— `[P1.2 ...(stageA 已合)]` 里的 `(` 必炸。
- ❌ **flowchart 不带引号的标签** `A[文字]` / `A(文字)` —— 同理。
- ⚠️ **带引号的 flowchart 标签** `A["...(...)..."]` —— 新版 mermaid 多半能渲染，但 GitHub 等渲染器版本不一、`<br/>` + 中文混排时仍可能挑剔。**不要逐场景赌，一律全角 `（）` 最稳。**
- ✅ 例外：**sequenceDiagram 的消息 / Note 文本** `A->>B: batch (RGB+mask)`、`Note over X: ...(...)` 走的是另一套语法，半角 `()` 合法，**不用动**（避免对 `v2_architecture.md` 的时序图做无谓 churn）。

**正例**：`P0["Phase 0 ★ 测量（门）<br/>..."]`、kanban `[P1.2 track-pose 完整版（stageA 已合 main）]`
**反例**：`P0["Phase 0 ★ 测量(门)..."]`、kanban `[P1.2 ...(stageA 已合 main)]`

**提交前自查**（改完 `v3_plan_revised.md` 的 mermaid 看板/依赖图后跑一次，**应零输出**；该 doc 无时序图，所有 mermaid 块内半角 `(` 都是违规）：
```bash
awk '/```mermaid/{i=1;next} /```/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v3_plan_revised.md
```
（`v2_architecture.md` 同样可跑，但需人工排除 sequenceDiagram 消息行的合法 `()`，不能简单要求零输出。）
