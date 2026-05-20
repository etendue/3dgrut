# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**3DGRUT** is NVIDIA's research implementation of two novel 3D scene reconstruction and rendering techniques:
- **3DGRT** (SIGGRAPH Asia 2024): Ray tracing of volumetric Gaussian particles, supporting distorted cameras, rolling shutters, and secondary rays (reflections, refractions, shadows).
- **3DGUT** (CVPR 2025, Oral): Rasterization-based version using the Unscented Transform, enabling distorted camera support within a rasterization framework. Combined into the hybrid **3DGRUT** approach.

## Environment Setup

Requires Python ≥ 3.11, PyTorch, and a CUDA GPU (CUDA 11.8, 12.4, 12.6, 12.8, or 13.0).

```bash
# Recommended (UV-based)
./install_env_uv.sh            # Linux
.\install_env_uv.ps1           # Windows

# Legacy (Conda, CUDA 11.8 or 12.8 only)
./install_env.sh 3dgrut

# Docker
docker build --build-arg CUDA_VERSION=12.8.1 -t 3dgrut:cuda128 .
```

## Common Commands

### Training
```bash
# NeRF Synthetic dataset
python train.py --config-name apps/nerf_synthetic_3dgut dataset.path=<path> output_path=<out>

# COLMAP dataset
python train.py --config-name apps/colmap_3dgut dataset.path=<path> output_path=<out>

# Use 3DGRT backend (ray tracing)
python train.py --config-name apps/nerf_synthetic_3dgrt dataset.path=<path> output_path=<out>

# MCMC densification strategy
python train.py --config-name apps/ncore_3dgut_mcmc dataset.path=<path> output_path=<out>
```

### Rendering / Evaluation
```bash
python render.py --config-path <checkpoint_dir> --config-name config dataset.path=<path>
```

### Interactive Visualization
```bash
# Polyscope GUI
python playground.py --checkpoint <checkpoint_dir>

# Viser GUI (browser-based)
python playground.py --checkpoint <checkpoint_dir> --gui viser
```

### Code Formatting & Linting
```bash
bash formatter.sh          # Format code (black, isort, clang-format)
bash formatter.sh --check  # Check only (used in CI)
```
- Python: Black (line length 120, target Python 3.11), isort (black profile)
- C++/CUDA: clang-format version 18 (LLVM style, see `.clang-format`)

### Benchmarks
```bash
bash benchmark/nerf_synthetic.sh
bash benchmark/mipnerf360.sh
bash benchmark/scannetpp.sh
```

## Architecture

### Entry Points
- `train.py` — Main training script, uses Hydra for configuration
- `render.py` — Evaluation/rendering from a checkpoint
- `playground.py` — Interactive real-time visualization GUI

### Core Packages

| Package | Role |
|---|---|
| `threedgrut/` | Main framework: training, datasets, model, optimization |
| `threedgrt_tracer/` | OptiX-based ray tracing backend (CUDA/C++) |
| `threedgut_tracer/` | CUDA rasterization backend (Unscented Transform) |
| `threedgrut_playground/` | Interactive visualization GUI |

### Key Classes

| Class | File | Role |
|---|---|---|
| `Trainer3DGRUT` | `threedgrut/trainer.py` | Orchestrates the full training loop, optimization, densification, evaluation |
| `MixtureOfGaussians` | `threedgrut/model/model.py` | Core scene representation (positions, colors, scales, rotations, densities) |
| `Renderer` | `threedgrut/render.py` | Checkpoint loading and test-time rendering with metrics |
| `BaseStrategy` | `threedgrut/strategy/base.py` | Abstract base for Gaussian optimization (densification, pruning, cloning) |
| `GSStrategy` | `threedgrut/strategy/gs.py` | Standard 3DGS gradient-based densification |
| `MCMCStrategy` | `threedgrut/strategy/mcmc.py` | MCMC-based densification (3dgs-mcmc) |

### Configuration System

Uses **Hydra** for all configuration. Config files are in `configs/`:
- `configs/apps/` — Full pipeline presets (e.g., `nerf_synthetic_3dgut.yaml`, `colmap_3dgrt.yaml`)
- `configs/render/` — Render backend configs (`3dgrt.yaml`, `3dgut.yaml`)
- `configs/strategy/` — Densification strategy configs
- `configs/dataset/` — Dataset-specific configs
- `configs/paper/` — Configs for reproducing paper results

Override any config key from the CLI: `python train.py ... training.iterations=30000`.

### JIT Compilation Pipeline

The CUDA/Slang backends are compiled at runtime (not pre-built):
- `threedgrut/utils/jit.py` — JIT compilation via `torch.utils.cpp_extension`
- `threedgrt_tracer/setup_3dgrt.py` — Compiles OptiX ray tracer (requires OptiX SDK in `thirdparty/optix-dev/`)
- `threedgut_tracer/setup_3dgut.py` — Compiles CUDA rasterizer (uses `thirdparty/tiny-cuda-nn/`)
- `threedgrut/strategy/src/setup_mcmc.py` — Compiles MCMC CUDA kernels
- Slang (`.slang`) shaders are compiled to CUDA at runtime; configuration parameters (SH degree, kernel params) are embedded at compile time

### Data Flow

```
Input Images → Dataset Loaders (COLMAP/NeRF/ScanNet++/NCore)
                     ↓
             MixtureOfGaussians (scene representation)
                     ↓
    ┌────────────────┼─────────────────┐
    ↓                ↓                 ↓
3DGRT (OptiX)  3DGUT (CUDA)     3DGRUT (Hybrid)
ray tracing    rasterization    raster + ray tracing
    └────────────────┼─────────────────┘
                     ↓
         Optimization (SelectiveAdam)
                     ↓
         Strategy (GS densification / MCMC)
                     ↓
         Checkpoint + Export (PLY, INGP, USD/USDZ)
```

### Dataset Support

- **NeRF Synthetic** — `threedgrut/datasets/dataset_nerf.py`
- **COLMAP / MipNeRF360** — `threedgrut/datasets/dataset_colmap.py`
- **ScanNet++** — `threedgrut/datasets/dataset_scannetpp.py`
- **NCore v4** — `threedgrut/datasets/datasetNcore.py`

All datasets implement the `BoundedMultiViewDataset` abstract interface in `threedgrut/datasets/protocols.py`.

### Export Formats

- **PLY** — Standard Gaussian Splatting format
- **INGP** — Instant NGP format
- **USD/USDZ** — For NVIDIA Omniverse and Isaac Sim (`threedgrut/export/usd/`)

## GPU/CUDA Requirements

- Minimum compute capability: 7.0 (V100, A100)
- RT cores required for 3DGRT ray tracing performance (RTX series recommended)
- Supported CUDA: 11.8, 12.4, 12.6, 12.8 (default), 13.0 (experimental)
- 3DGRT depends on NVIDIA OptiX SDK (bundled in `thirdparty/optix-dev/`)
- `thirdparty/tiny-cuda-nn/` provides fast CUDA neural network primitives for 3DGUT

## A800 远程执行环境（重要）

Claude 可以**直接通过 `ssh a800-x2` 访问 A800 GPU 主机**执行训练 / 集成测试等 GPU 任务。不需要把"A800 上的任务"标记为待用户手动操作；不需要等待用户确认。所有 v2 plan 里写"A800 探测 / 5k smoke / 30k KPI"等任务都是 Claude 直接 ssh 跑。

- 远程主机别名：`a800-x2`（~/.ssh/config 已配置）
- 远程仓库路径：`/root/work/yusun/repo/3dgrut`
- 数据集路径：`/root/work/yusun/ncore-nurec/data/ncore/clips/...`
- 输出路径：`/root/work/yusun/ncore-nurec/output/...`
- 推荐 GPU：`export CUDA_VISIBLE_DEVICES=0` 或 `1`，看 `nvidia-smi` 选空闲卡
- conda env：`conda activate 3dgrut`
- 内存配置：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

执行模式：用 `Bash` 工具单条 `ssh a800-x2 << 'EOF' ... EOF` 心跳；长任务（≥ 5 min）用 `run_in_background=true`，Claude 会被自动通知完成；用 `tee /tmp/<run>.log` 同步 grep "PSNR" / 错误。

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
