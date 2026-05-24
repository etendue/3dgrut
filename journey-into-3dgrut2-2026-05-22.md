# Journey Into 3dgrut2 (Extended) — 2026-05-07 至 2026-05-22

## 1. 元信息 (Header)

| 项目 | 3dgrut2 (NVIDIA 3DGRUT fork, oss-sim 自动驾驶 4D 重建管线) |
|---|---|
| 时间跨度 | 2026-05-07 04:00 UTC → 2026-05-22（最后 commit `8187a51`，T8.14 vast.ai 4090 视觉验收归档）|
| 编写日期 | 2026-05-22 |
| 数据来源 (Phase A) | claude-mem SQLite DB — 213 条 observations / 4 sessions / 600,924 discovery tokens |
| 数据来源 (Phase B) | `v2_plan.md` Done Log (1174 行) + `v2_stage_detail.md` 设计 spec + git log (~80 commits) |

**数据保真度不对称声明**：本报告分两个阶段叙事，但两阶段的数据保真度差异极大。Phase A (5/7-5/9) 完整保留在 claude-mem 中，每条 observation 都有 narrative / facts / concepts 三段化结构和 token 成本可追溯；Phase B (5/14-5/22) 的 claude-mem 没有任何记录（hook 不再写库，原因不详），全部依赖 `v2_plan.md` 和 git log 重建。两段叙事在颗粒度上必然不平衡——Phase A 能精确到分钟 + observation ID，Phase B 大多只能精确到 task + commit hash。

---

## 2. 项目缘起 (5/7 04:00 UTC)

`#1` 是一份 Marp 中文 Slide deck —— Claude 给 3DGRUT 写的 13 页幻灯片，NVIDIA 绿 `#76b900` + Noto Sans SC。这一条不算"开始"，更像是项目作者在动真格之前做的暖场练习。

真正的起点是 2.5 小时后的 `#2 (5/7 06:28:13 UTC)`：**"3DGRUT2 NCore v4 Training Pipeline Initiated on Remote 4090 Host"**。这里出现了贯穿后续三天的所有关键坐标：

- 远端 RTX 4090 主机 `inceptio@10.8.31.113` (Ubuntu 24.04)
- NCore v4 测试 clip `3435ace9-85eb-4de0-89ab-6b8dca522e8a`
- 本机 proxy `10.8.30.215:7897`（远端 WiFi 网络极慢）
- 目标：oss-sim 路线图 **WP V1-1**（NCore v4 校验器 + `scene_manifest.json` 生成）

整个项目要解决的痛点很清楚：NVIDIA 自家的 NuRec 在 9ae151dc clip 上 PSNR 36.28 dB，但 NuRec 是闭源的。oss-sim 的目标是用开源的 3DGRUT fork 做出能产出 USDZ 的 L4 干线货运仿真重建管线，给"开源版的 NuRec"留出一条可控、可扩展的路径。3dgrut2 就是这条路径的代码主线。

---

## 3. 三天高强度冲刺 — WP V1-1 (Phase A, 5/7-5/9)

213 条 observation、600,924 discovery tokens、4 个 session、3 台不同主机——这是 Phase A 的全部体量。按天拆开看，每一天都有完全不同的性格。

### 3.1 Day 1 (5/7, quizzical-mendel-60fc6b, 111 obs) — 4090 环境地狱

`#3 (5/7 06:29)` 一口气把 oss-sim roadmap 477 行 16 个 work package 的全貌读了一遍。接下来 100+ 条 observation 几乎全部围绕一件事：**怎样在没有 sudo、WiFi 限速 ~556 KB/s、GCC 13 默认、Python 3.12 自带**的 4090 主机上把 3DGRUT 跑起来。

环境侦察阶段反复试错：
- `#5-7` 发现 `install_env.sh` 要 conda + GCC ≤ 11 + Python 3.11，远端三个都不满足；切到 `install_env_uv.sh` (uv-based, 无 sudo)
- `#7` 发现 `cuda_helper.sh` 对 CUDA 12.8 允许 GCC ≤ 14——远端 GCC 13.3 反而是合法的
- `#10 (06:33)` 决定本机下 CUDA 12.8.1 runfile 然后 rsync 过去；6 分钟后 `#18 (06:38)` **放弃**——proxy 也只有 ~20MB / 4min，太慢
- `#19-20 (06:40)` 灵活转向：本机 `.git` 仅 55 MB，rsync 过去恢复 git 元数据，再让远端用 proxy `git submodule update --init`
- `#21 (06:41)` 给 4090 写了一个 88 行的 `setup_3dgrut_4090.sh`，重点：**PyTorch 2.4.0 + CUDA 11.8 (cu118)**——选 cu118 是因为 kaolin 0.17.0 有 pre-built wheel，避免 git clone kaolin 源代码的慢路径

接下来 30+ 条 observation 在等 PyTorch 760 MB 大包下载。`#43-44 (07:27)` 实测 WiFi 仅 ~556 KB/s，`#45 (07:29)` 杀掉远端 pip 进程，切到"本机下 wheels 再 rsync"策略。`#46` 立刻发现新坑：本机 macOS Python 3.14 / arm64，远端要 linux_x86_64 / cp311 wheels——必须用 `pip download --platform linux_x86_64 --python-version 311` 跨平台拉包。

`#32 (07:18)` 是这一天最重要的**决策**：明文记录了"PyTorch 2.4.0 + CUDA 11.8 wheels 即使在 nvcc 12.0 的机器上也能跑"的判断，给后续大半天的安装节奏定调。

到了 `#161-167` (5/8 凌晨 04:08-04:13 注：跨越到 Day 2，但仍在 4090 主线上)，事情突然变明朗——`#162` **"env.tar Contains Full CUDA 12.8.1 Toolkit"**（16,346 discovery tokens，Phase A 第 4 贵的 obs）：原来公司预打包的 `env.tar` 容器里已经包含了完整 CUDA 12.8.1 工具链 + nvcc + Nsight Compute + Nsight Systems。前面"如何搞到 CUDA 12.8"折腾了整整一天，谜底却藏在公司容器里。`#163` 直接 `CUDA_HOME=/home/inceptio/repo/3dgrut2/.venv/cuda-12.8.1`，5 步 nvcc 全部编过；`#166-167` 两个 CUDA extension (`lib_mcmc_cc.so` / `lib3dgut_cc.so`) 用 sm_86 + sm_89 编译完成，缓存到 `~/.cache/torch_extensions/py311_cu128/`。

### 3.2 Day 2 (5/8, flamboyant-wilson-faac5f, 67 obs) — WP V1-1 验收实现

`#148 (5/8 04:02:53)` 是 Day 2 真正的开始决策："**Starting WP V1-1 Implementation: Creating threedgrut/tools/ncore_validate.py in Local Worktree**"。环境地狱总算告一段落（其实还会反扑，见下），可以写代码了。

`#149 (04:03:34, 9,582 tokens)` 一次性交付了 4 个新文件：
- `threedgrut/tools/__init__.py`
- `threedgrut/tools/ncore_validate.py` — 10 步 validator 管线（前 6 步 blocking ERROR / 后 2 步 non-blocking WARNING）
- `schemas/scene_manifest.schema.json`
- `threedgrut/tests/test_ncore_validator.py` — 12 个单测，0.23s 跑完

`#151 (04:04:11)` —— 第一个真实 NCore clip 验收：3435ace9-85eb-4de0-89ab-6b8dca522e8a 完整通过 8 步检查，0 ERROR / 0 WARNING / 7 cameras (全部 1920×1080 FTheta rolling shutter, 全部带 ego mask) / 1 LiDAR (199 frames) / 374 cuboid track observations 来自 11 unique track IDs (361 automobile + 13 trailer)。20 秒 driving clip，pose graph 含 world / world_global / rig。`#153 (04:05:05)` 完成了三项 acceptance test：正例 (validator exit 0) / 完整性 (6 必要 section 全在) / 反例 (移除 LiDAR → exit 1 + ERROR message)。

接下来从 `#155-167` 切回训练侧。**`#162` env.tar 的发现就是在这一连串 observation 里出现的**——具体来说是在 Day 2 准备启动 30k 训练时再读了一次 .venv 的内容；前面 Day 1 已经描述过它对环境问题的颠覆性影响。

`#168 (04:13:37)` 决定**启动 30k iter / 7 cameras 的完整训练**。然后立刻撞上：

### 3.3 Day 2 的两次 bugfix（5/8 04:14 与 04:41）

**第一次 bugfix `#170 → #171` 仅相隔 24 秒**：
- `#170 (04:14:27)` 训练立即 exit，因为 macOS Claude session 的 PATH (`/Users/etendue/Library/Application Support/Claude/...`，**含空格**) 通过 SSH env forwarding 灌进远端 bash，远端 `export PATH=...:$PATH` 命中含空格的 identifier 失败
- `#171 (04:14:52)` 修复：新写 `/tmp/run_ncore_train.sh`，**第一行强制 reset PATH 为 Linux 干净基线**（`/usr/local/sbin:/usr/local/bin:...`）再追加 `.venv/bin`；script 用 `scp` 上传而不是 SSH heredoc 写入（避免传输路径中再次被 PATH 污染）

24 秒——一个根因诊断 + 完整修复 + commit 的周期。这个 fix 后来上升为 **CLAUDE.md hard rule**（永远显式设 PATH 不要继承 SSH 环境）。

**第二次 bugfix `#178-179 (04:41)`**：fused_ssim 用 venv 内 broken pip shebang 装不进去 (`/home/inceptio/repo/3dgrut2/.venv/bin/pip: cannot execute`)。这个 venv 是别人预打包好的、shebang 指向不存在的 Python 路径。`#178` 找到 uv 装在 `~/.local/bin` 不在默认 PATH，用 `uv pip install --python $VENV/bin/python3.11` 显式指定 Python 解释器，1m21s 从源码 build fused-ssim 成功。`#179` 三个核心 import (fused_ssim / threedgrut.model.losses.ssim / Trainer3DGRUT) 全部干净退出。

到这一刻——Day 2 末，4090 上的训练**还没有真的跑起来**（先后被 PATH 泄漏和 fused_ssim segfault 阻断）。

### 3.4 Day 3 (5/9, unruffled-swirles-a435f1, 34 obs) — Vast.ai A100 大转向

`#180 (5/9 02:40:27)` 决策：**放弃 4090，转 Vast.ai A100**。所有 Day 1/2 的环境痛点（WiFi 慢 / 自打包 venv 残缺 / fused_ssim binary 不兼容 / 4090 编译 segfault）一夜之间全消失：Vast.ai 实例 `192.165.134.28:12509` 已经有一个预配置好的 `.venv` 在 `/home/inceptio/3dgrut`，全套 3DGRUT 依赖都装好了。

但是 Vast.ai 自己也有坑：
- `#180` 一开始就指出：**`HOME` 默认是 `/root` 而不是 `/home/inceptio`**，每次 ssh 进去后必须 `export HOME=/home/inceptio`
- `#188 (02:47:30)` 又记下一条：**Vast.ai 把 ssh session 包在 tmux 里**，`ssh -tt` 强制 TTY 否则 non-interactive 命令不稳定

`#181-184` 是 Day 3 的**discovery 高密度区**——也是整个 Phase A 最贵的 4 条 observation 全部集中在这里（共 94,572 discovery tokens / 4 条）：

| Obs | Tokens | 标题 |
|---|---:|---|
| `#183` | 37,920 | 3DGRUT pyproject.toml Dependencies and Package Structure for Vast.ai Deployment |
| `#182` | 22,700 | 3DGRUT Codebase: Missing tools/ and tests/ Directories; NCore V4 Meta-file Format Confirmed |
| `#181` | 19,971 | 3DGRUT oss-sim Roadmap: Existing Capabilities and Work Package Breakdown |
| `#184` | 13,981 | NCore Dataset Initialization Flow and V1-1 Validator Design Blueprint |

这是一个**为新 session 做基础知识储备**的密集投资：Claude 知道 Vast.ai 这个新 session 没有 Day 1/2 的本地 context，需要在 session 起手 5 分钟把整个 3DGRUT 项目的依赖 / package 结构 / 路线图 / 数据集 API 全部重新读完。这 4 条 obs 是 Phase A 单条 token 量的天花板，理由也合理：这是 cold-start 的代价。

`#187 (02:47:23)` 把 Day 1+2 全部研究综合成一个 plan file（`/Users/etendue/.claude/plans/according-to-oss-sim-roadmap-md-how-zaz-composed-starlight.md`），明确两阶段策略：先跑 **2 秒 single-camera** 验证 pipeline，再跑 **20 秒 7-camera** 完整训练。

`#190 (03:12)` 启动 2s 训练。`#192 (5/9 03:17:55)` —— **Phase A 最重要的成果**："**3DGRUT NCore v4 2-Second Clip Training Succeeded — Checkpoint at iter 7000**"。`pai_0a119d27-7022-41f6-aa84-a095c97f85fa` 这个 clip 是 Vast.ai 上 hugging-face cache 自带的（不需要从 10.8.31.113 传数据），50 train + 8 val 帧，6.5 分钟跑完 7000 iter。MCMC relocation rate 从 47% 降到稳定 11-13%。

`#197 (03:40:36)` 训练继续跑到 30000 iter：**PSNR 34.85 / SSIM 0.955 / LPIPS 0.191**——2 秒 clip + 单相机的完美数字。这是项目开始 3 天后第一个真实可用的 3DGS checkpoint。

`#199 (03:43:32)` 立即启动全 20s 训练，最初只指定 `camera_front_wide_120fov`。`#201 (03:44:20)` 临时发现 scene manifest 里其实有 7 个 camera，`#202` 立刻 Ctrl-C 停止单相机训练，`#203 (03:44:40)` 删 output 重启 7-cam 训练。

`#207 (04:20:01)` 7-cam 训练结束的 metrics：**PSNR 27.60 / SSIM 0.872 / LPIPS 0.352**。这个数字立刻引出一个矛盾：**2s 单相机 34.85 dB 与 20s 7-cam 27.60 dB 差 7 dB**。

`#209 (07:51, ~3.5 小时后)` 是这一天最有意思的一条记录：**"User Corrected Hasty KPI Decline Reasoning"**——用户直接打回了 Claude 给的"原因解释"，原话："推理太草率"（the reasoning was too hasty）。用户指出 KPI 下降可能有许多原因（多相机 frustum 重叠 / 视角监督稀疏 / 训练步数不足 / 训练数据增加 7×），不能锚定第一个看似合理的解释。**这个反馈直接影响了后续 Phase B 的工作风格——Stage 7 后期对 ExposureModel 失控的诊断流程就明显带着"先做 ablation，再下结论"的痕迹**。

`#210 (07:52)` 是 WP V1-1 的正式总结条目；`#213 (07:55:55)` 写出了 `journey-into-3dgrut2.md` 第一版（本报告是它的扩展版），同时也是 Phase A 最后一条 observation。

---

## 4. 方案 B 决策 (5/14, 两阶段之间)

5/9 到 5/14 这 5 天的窗口在 claude-mem 中是空白的。从 `v2_plan.md` 的 § 0 和 git log 反推出来，这段时间发生的最重要的事是 **`RS_vs_3DGRUT_DecisionAnalysis.md` 决策文档定稿**：经过对 Reconstruction-Studio (Recon-Studio, 简称 RS) 的代码审计，决定走 "**3DGRUT v2-fork + OmniRe selective porting**" 的方案 B，而不是单走 RS。

理由很务实：RS 当时的测试状态是 single-cam-no-pedestrian，而我们要做的是 7-cam + dynamic vehicle 的产业级场景。但 RS 里有 3 个模块值得 selective port:
- **ExposureModel** (45 行 affine `exp(a)*img + b`) → Stage 6
- **Marching Cubes mcube_utils** → 未来 mesh 导出路径
- **Road 2DGS surface gaussians** → 未来 v3-T9 路径

资源分配 **80/20** —— 3dgrut2 占 80%，是主线；RS 保留 20% 用于探索性 port。

这个决策的另一面是：v2 LayeredGaussians 的设计目标也在这一周成型。NVIDIA NuRec 的层级架构（背景 / 路面 / 动态刚体 / 动态形变 / 天空 envmap）被反向工程出来，写进 `v2_plan.md` 作为 Stage 1-7 的 anchor。

---

## 5. v2 LayeredGaussians 九阶段长征 (Phase B, 5/14-5/22)

8 天 / ~57 个 done task / ~80 个 commit / 9 个阶段。所有数据都来自 `v2_plan.md` § 5 Done Log，没有任何 claude-mem 自动捕获，全部依赖 task 完成时手动回填的 PSNR / commit hash。

### 5.1 九阶段全景

| Stage | 时间窗 | 出口判据 | A800 PSNR | 关键 commit |
|---|---|---|---:|---|
| **0** | 5/14 16:55-16:57 | A800 env smoke | 24.12 dB | — |
| **1** | 5/16 → 5/18 | LayeredGaussians 容器 + LayerSpec registry | byte-identical 24.123 | `b0865c4 → 8a29fc0 → 5a6a5f9 → 60e1154 → 6435483 → ff83028` |
| **2** | 5/18 | LayeredMCMC sub-strategy 数组 | byte-identical 24.123 | `62fc509 → 7ad883b → 1a0d275 → 51540a8 → d4841df` |
| **3** | 5/19 | Road 层 (BEV LiDAR-Z + region-weighted L1) | 5k 26.133 | `8a625c2` |
| **4** | 5/19 | DynamicRigid (timestamp-aligned 真 cuboids) | 10k 26.315 | `4807951` |
| **5** | 5/19-20 | Sky envmap (MLP fallback, nvdiffrast 远端无) | 5k 26.167 | `b38fce8` |
| **6** | 5/20 | ExposureModel (RS Luxury port) | 5k cc 24.94 (+1.7) | `b38fce8 → 217262c` |
| **6-fix** | 5/20 | Ego mask 全栈接通 + 双指标 | 5k masked **29.49** (full 20.49) | `65869ec → 12e142a` |
| **7** | 5/21 | 7-cam 30k 完整 KPI | raw masked 15.63 ❌ / OFF 25.76 ✅ / cc 24.7 plateau | `eaf6404` |
| **8** | 5/20-22 | viser_gui_4d 4D 浏览器查看器 | 视觉验收通过 | `fedf73e → ... → 8187a51` |

### 5.2 Stage 0 (5/14) — A800 env smoke

A800-x2 单卡 1k step × 2s × 单相机：PSNR 24.12 dB / SSIM 0.846 / 9.48 it/s（A100 同条件 ~3.9 it/s，A800 快 2.4×）。**关键发现**：实际 clip 是 `9ae151dc-e87b-41a7-8e85-71772f9603d7`（不是文档误写的 3435ace9），数据在 `/root/work/yusun/ncore-nurec/data/ncore/`，必须 `CUDA_VISIBLE_DEVICES=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 5.3 Stage 1 (5/16-18) — LayeredGaussians 容器

T1.1-T1.5 共 5 个 task，6 commit。核心实现是 `threedgrut/layers/layered_model.py` 的 `LayeredGaussians` 类，本质是一个 `ModuleDict[name → MixtureOfGaussians]`。最重要的不变量是 **v1 ckpt resume 必须 byte-identical**——这是项目的"零回归"红线。T1.5 commit 5a6a5f9 在 A800 上 v1 ckpt resume 测出 24.123 dB / 8 帧 PSNR 完全一致。

T1.2 commit `60e1154` 把 LayerSpec 从 3 字段扩到 8 字段（`scale_prior / scale_lr_mult / mask_field / is_particle_layer / density_init`），新建 `layers/registry.py` 的 STANDARD_LAYERS dict 注册 5 个标准层（background / road / dynamic_rigids / dynamic_deformables / sky_envmap）。这一步抽象之所以重要：**所有后续 Stage 3-6 的层逻辑都共享同一个数据契约**。

T1.4 单测覆盖 18 个测试：9 个 Mac 本地 (`test_layer_spec_registry.py`) + 9 个 A800 contract (`test_layered_gaussians.py`)。

### 5.4 Stage 2 (5/18) — Layered MCMC sub-strategy 数组

T2.1-T2.5 共 5 个 task。原 plan 设计的是"override `_select_indices`"，实际实现采用了更轻量的 **sub-strategy 数组**方案：`LayeredMCMCStrategy` 持有 `sub_strategies: dict[str, MCMCStrategy]`，每个 is_particle_layer=True 的层各一个 sub。`_post_optimizer_step` 遍历 subs，串行调用——自然实现"零跨层迁移"，不需要在 MCMC 内部动态切换 layer 上下文。

A800 出口测试 24.123 dB byte-identical，commit `d4841df`。Stage 2 末完成 D8 出口门禁。

### 5.5 Stage 3 (5/19) — Road 层

T3.0-T3.5b 共 8 个 task，最终 commit `8a625c2`。三个核心模块：

1. **BEV LiDAR-Z KNN init** (`layers/road_init.py`)：把 LiDAR semantic 中的 road 类点云投到 BEV 网格上，每个 grid cell 用 `torch.cdist` 找最近 road 点，Z 拉到该点，scale = `[0.1, 0.1, 0.001]`（flat scale prior）
2. **Region-weighted L1 loss** (`trainer.py::get_losses`)：把 loss 拆成 bg / road / dyn 三区，各自归一化后相加，避免天空类大像素区域把 road 类小像素区域稀释掉
3. **Z lock** + **scale_lr_mult=0.2**：path-induced 防止 MCMC perturb 把路面粒子拉飞

A800 5k smoke 出口 **PSNR 26.133 dB / SSIM 0.879 / LPIPS 0.297 / 9.54 it/s**，超出口门槛 (≥ 23.6, v1+0.5) 整整 **+2.5 dB**。

### 5.6 Stage 4 (5/19) — DynamicRigid + timestamp-aligned

T4.0-T4.5 共 6 个 task，commit `4807951`。这个 stage 最复杂的不是粒子层本身，而是**时间对齐**：

1. **tracks_loader** 从 NCore cuboid autolabels 实时构造 instance_pts_dict：iter `loader.get_cuboid_track_observations()` (13657 obs across full clip) → groupby track_id → filter vehicle classes → 每 cam frame 找 nearest cuboid obs within 50ms tolerance → `obs.transform("world", ts, pose_graph)`。clip 9ae151dc 在 2s 窗内：179 unique tracks → 31 vehicle tracks
2. **timestamp-aligned `_resolve_pose_idx`**：`Batch.timestamp_us` 字段（dataset 写入 cam END timestamp = sseg key 一致），`LayeredGaussians.forward` 用 `gpu_batch.timestamp_us` 而非 `frame_id=global_step`。binary-search 共享 `tracks_camera_timestamps_us` buffer，返回最近 pose index
3. **MCMC track_ids buffer sync**：`add_new_gaussians` / `relocate_gaussians` 都加 hook 同步维护 track_ids 张量长度

**踩坑回填**: timestamp 修复前 F6/F7 后帧退化 -3 dB；修复后 F6/F7 反而 +0.7 dB（multi-frame consistency 提升）。

A800 10k smoke 出口 **PSNR 26.315 dB / SSIM 0.883 / LPIPS 0.275 / 9.58 it/s / dyn 粒子 48,488** —— 三层 vs 两层**零性能损失**。

并行任务也很巧妙：A800 GPU 0 同时跑 nre-tools aux 生成（无关任务），GPU 1 跑 Stage 4 训练。一台 A800-x2 同时榨干 2 张卡。

### 5.7 Stage 5 (5/19-20) — Sky envmap

T5.1-T5.6 共 6 个 task。T5.1 探测 A800 conda env 是否带 `nvdiffrast` —— 不带，PyPI 镜像 (Huawei) 也没有，GitHub install 被 sandbox 拒绝。**Stage 5 落入 MLP fallback 路径**（pre-designed plan 已经预留）。

实现选择：
- `threedgrut/correction/sky_envmap.py` 含 `SkyEnvmapBase` 抽象基类
- `SkyEnvmapMLP`（无外部依赖，SinusoidalEncoder + 3 层 MLP + sigmoid）
- `SkyEnvmapCubemap`（drivestudio EnvLight 移植，nvdiffrast 不可用时 forward 抛 ImportError 带明确指引）

`LayeredGaussians.forward` 末尾 `_blend_sky(outputs, batch)`，写入 `rgb_gaussians` / `rgb_sky` / `pred_rgb` 三 key。`sky_loss` 用 sky_mask（来自 NCore semantic）只在天空区监督 sky_envmap。

A800 5k smoke 出口 **PSNR 26.167 dB**（出口门槛 ≥ 25.8 ✅ +0.37 dB），与 Stage 4 baseline 26.315 仅 -0.15 dB（noise 级）。

### 5.8 Stage 6 (5/20) — ExposureModel + Stage 6-fix 的 ego mask 重启

Stage 6 T6.1-T6.3a 把 RS Luxury 的 `ExposureModel(num_camera)` 移植过来——29 行 affine：`forward(idx, image) = (exp(a)*image + b).clamp(0, 1)`，零初始化 = identity。**关键观察**：A800 5-cam 5k smoke 显示 `mean_psnr = 23.237` 但 `mean_cc_psnr = 24.937` —— **color-correction 后 +1.7 dB**，直接证明 ExposureModel 学到了非平凡的 per-cam gain。学到的 gain 范围 0.878 (front 暗化 12%) → 0.946 (cross right)，确实补偿了多相机间的曝光差异。

**但是出现了一个谜团**：raw mean_psnr 23.237 dB 看起来比 Stage 4 baseline 26.315 dB 低了整整 3 dB。是 ExposureModel 出问题了吗？

**Stage 6-fix (T6F.1-T6F.3, commit 65869ec → 12e142a)** 给出了完全不同的解释。审计 Stage 0-6 代码路径后发现一个尴尬事实：

> `NCoreDataset.__init__` L385-395 加载了 ego car mask 并 dilate 缓存到 `sequence_cameras_frame_valid_pixels_masks`，但**实际从未流入 loss / metric**——训练分支 `__getitem__` L806-872 完全没读它，验证分支 L959 采样了 `valid` 但 `get_gpu_batch_with_intrinsics` L1273-1367 没把 `valid` 拷贝进 `Batch`。结果 `gpu_batch.mask = None`，trainer 的 mask 乘法跳过，`compute_layered_l1_loss(valid_mask=None)`，PSNR/SSIM/LPIPS 全图算。

T6F.1 把 ego mask 真接通：dataset → Batch → loss → metric 全栈。T6F.2 同时改 `trainer.py::compute_metrics` 和 `render.py` 的 eval loop，加 `psnr_masked / ssim_masked / lpips_masked` 三个新指标（mask=None 时与全图 byte-identical）。

T6F.3 A800 5k smoke 出口数字震撼：

| 指标 | full (全图) | masked (T6F.1 ego 接通后) | Δ |
|---|---:|---:|---:|
| mean PSNR | 20.493 | **29.493** | **+9.000** dB |
| mean SSIM | 0.858 | 0.934 | +0.076 |
| mean LPIPS | 0.317 | 0.190 | -0.127 |
| mean cc_PSNR | 19.611 | 24.904 | +5.293 |

**这是整个 v2 项目最重要的方法论突破**：原来 Stage 3 PSNR 26.13 / Stage 4 26.32 / Stage 5 26.17 / Stage 6 cc 24.94 这些 baseline 数字**绝大多数是 ego 车身被强行训练拟合"完美"拉高的**——车身在所有帧位置固定 + 颜色固定，是高斯最容易拟合的区域。真实可用的非 ego 区 PSNR 一直被 ego 内卷压制。

**踩坑挂载到 CLAUDE.md hard rule**：
1. 第 1 次 5k smoke 立即崩（broadcast shape mismatch in compute_layered_l1_loss 4D valid vs 3D road），commit 9c18b57 squeeze + 2 个回归测试 pin 住
2. 第 2 次 5k smoke 跑过但 metrics.json 缺 6 个 masked 字段——T6F.2 只改了 trainer.py 没改 render.py 的 eval loop。修复后挂到 CLAUDE.md §B "trainer.py + render.py 两处改动点"必须同时核查
3. eval 单跑 render.py 失败——A800 顶层 render.py 被错放成包内 threedgrut/render.py 内容（silent exit 0 无产物）。rsync 本地正确顶层 render.py 覆盖修复。挂到 CLAUDE.md §A "ssh a800-x2 \"head -25 <entry.py>\" 验证入口"

### 5.9 Stage 7 (5/21) — 7-cam 30k KPI + ExposureModel 失控

T7.1-T7.5 + 新增 T7.3.b。最终交付指标 **cc_psnr_masked 24.70 dB** ≈ Stage 5/6/6-fix baseline 24.7~24.9（σ < 0.2 dB noise 级），几何质量不退化。

**关键 ablation**：

| Task | A800 Run | training_time | raw psnr_masked | cc_psnr_masked | 结论 |
|---|---|---:|---:|---:|---|
| T7.3 | `stage7_full_20260520-202222` (7-cam 30k exposure ON) | 3061 s | **15.63 ❌** | **24.75** | 完成但 raw KPI 崩 |
| T7.3.b | `stage7_noexp_20260521-102930` (7-cam 30k exposure OFF) | 3064 s | **25.76 (+10.13)** | **24.70 (-0.05)** | **证伪 ExposureModel 是 raw 崩真因** |

**关键技术发现 (3 项)**：

1. **ExposureModel 退化优化失控** — 30k step 长训暴露的病态短路径：训练有两条 loss 下降路径，30k Adam 无约束 → 模型选择病态：
   - 路径 1 (物理正确)：高斯学准真实色彩 → exposure 维持小值
   - 路径 2 (病态短路)：高斯学个大概 → exposure 把偏差全 compensate（14 参数 vs 几百万高斯，更快收敛）
   - 实证：T7.3 raw_masked 15.63（路径 2 终点，渲染严重过曝/泛白）vs T7.3.b 25.76（关掉强制走路径 1）
   - **但两组 cc_psnr_masked 几乎一致 (24.75 vs 24.70) → 真实几何/纹理质量没差**，ExposureModel 只是 raw 输出与 GT 的色彩偏差

2. **cc_PSNR 是真实重建质量 KPI** — `color_correct_affine` (Google multinerf) per-image per-channel lstsq 撤销色彩偏移；NeRF 圈（Mip-NeRF 360 / Block-NeRF / drivestudio）都报 cc PSNR。Stage 7 真实指标应是 cc_psnr_masked，不是 psnr_masked

3. **7-cam 30k vs 1-cam 5k masked PSNR 反而低 3.7 dB**（25.76 vs 29.49）—— 多相机长训没有带来质量净提升，实证 v2 架构在 NCore 9ae151dc clip 上的天花板 ≈ **24.7 dB cc_psnr_masked**

**T7.4 cap ablation 跳过**（根因不在 cap）。**V3-P1 升级为整合任务**：bilateral-grid + ExposureModel 退化修复合并研究，预估 3-5 天，作为 v3 启动的最高优先级。

**Stage 7 踩坑 #1 (ssh heredoc 被 SIGHUP 杀)**：第一次 T7.3 用 `ssh a800-x2 << 'EOF'` 跑 30 min 训练 → ssh session 因网络/timeout 断开 → 远端 Python 进程被 SIGHUP 杀掉，step ~20000 ckpt 写出但无 metrics.json。修复：改为 `ssh ... "nohup setsid /tmp/script.sh < /dev/null > /tmp/nohup.out 2>&1 & disown"` 完全脱离 ssh session，写 `.done` sentinel + 复制 metrics 到固定路径。挂到 CLAUDE.md §A 把关清单第 6 条。

### 5.10 Stage 8 (5/20-22) — viser_gui_4d 4D 可视化的三天长征

Stage 8 是九个阶段里**唯一一个跑了三天的**——T8.1-T8.14 共 14 个 task，覆盖完整的浏览器侧 4D 场景查看器。设计目标：在浏览器里拖 timeline → 4D 场景中的车跟着动 / 绿色 ego polyline / 当前时刻 cuboid wireframe / road + dynamic LiDAR overlay，**所有 4D 元素从 ckpt['viz_4d'] 直接读，viewer 不依赖 NCore SDK / dataset 重加载**。

**Stage 8 关键节点**：

- **T8.8** (5/20 A800)：100-step smoke 跑通 viz_4d schema_v1（ckpt 960 MB，含 70 tracks）
- **T8.9** (5/20)：`inject_viz_4d` CLI 退役改造老 ckpt（方案 B）—— 在 T6F.3 ckpt 上实测：991→995 MB +3.8 MB metadata
- **T8.10** (5/20)：`--no_gaussian_render` flag。**发现 Ampere datacenter SKU (A100/A800) RT cores 被 LASER 阉割** → OptiX dlopen segfault，A800 永远无法渲染 Gaussian。Workaround：scene primitives (cuboid/LiDAR/ego frustum) only
- **T8.11** (5/20)：dynamic LiDAR 改 per-track object-local frame —— 修复 cuboids 移动但 dyn LiDAR 静态世界并集的 bug（48K pts 跨 20 active tracks，cap 5000/track）

**最难的 3 天弧 — fisheye 调试 saga (5/20-21)**：

- **T8.12** (5/21, vast.ai RTX 4090 Norway $0.630/hr, ⚠️ 部分通过)：修了 2 个真实 Stage 8 集成 bug
  - Bug #1：`engine._trace_scene_mog` Layered 路径缺 camera intrinsics（3dgut UT 光栅器需要 fx/fy/cx/cy 投影 Gaussian 协方差）
  - Bug #2：`SkyEnvmapMLP` 残留 CPU（`init_from_checkpoint` 末尾未 `.cuda()`）
  - 87 FPS @ 1024×600 跑起来了，**但视觉输出是"远景隧道 motion-blur 乱糊"**，与 render.py 完全不像
  - 根因：viz_4d schema 只存了单个 `primary_camera_fov_y_rad`，丢掉了 NCore `camera_front_wide_120fov` 训练时用的 FTheta polynomial / distortion coefficients

- **T8.12-FIX** (5/21, vast.ai California 4090 $0.98)：加 CLI flags `--initial_fov_deg / --camera_type / --camera_fov_deg`。**诊断锚定**：
  - **Phase A.2** (pinhole 90° corrected fov) → SAME tunnel motion blur as fov 45/75/140 → **fov 假设证伪**
  - **Phase A.5** (Fisheye 120° equirectangular) → 结构性突破：椭圆 fisheye 框 + cuboids 在合理位置，**但内容仍模糊**因为 equirectangular ≠ FTheta polynomial → **T8.13 必走**
  - **根因结构**：3dgut UT rasterizer 在 [tracer.py:471](threedgut_tracer/tracer.py:471) 是 rasterizer 不是 ray tracer，它用**intrinsics**（而非 rays）投影 Gaussian 中心 + 协方差到 2D 屏幕椭圆。FTheta-trained Gaussian 的 3D 形状已"反预补偿"FTheta 扭曲，用 pinhole 或 equirectangular 投影都产生双重扭曲

- **T8.13** (5/21 Mac+A800, plan `t8-13-flickering-dragon`)：viz_4d SCHEMA v1→v2 with full 8-key FTheta polynomial dict (`resolution / shutter_type / principal_point / reference_poly / pixeldist_to_angle_poly / angle_to_pixeldist_poly / max_angle / linear_cde`)。**关键发现**：3dgut UT rasterizer at tracer.py:471 **已经原生支持 FTheta**，只要 `gpu_batch.intrinsics_FThetaCameraModelParameters` 是 8-key dict，kernel 自动调 `_3dgut_plugin.fromFThetaCameraModelParameters`。**整个 T8.13 是纯 Python 改动**（无 C++/CUDA 代码改动，无 OptiX 编译）
  - Mac 单测：**206 passed + 1 skipped, 0 回归**（189 旧 + 17 新）
  - A800 inject 验证：1m50s，schema_v2，FTheta keys 全在，resolution=(1920,1080)，max_angle=1.221 rad（=70° 半视场 ≈ 140° 全视场，与 `camera_front_wide_120fov` 一致）
  - **vast.ai 4090 视觉验收 ✅**：probe 截图 `after_fix_clear_streetview.png` —— **可识别街景 + fisheye 桶形畸变 + "HEAT" 广告牌 + 白车 + 路灯柱 + 楼宇 + 圆形 vignette 黑边** —— 与 render.py ground truth 同形态
  - 用户反馈：ego 训练轨迹附近清晰+速度快 ✅；viser orbit 离开训练相机轨迹后视觉质量急剧下降 —— **这是 Gaussian Splatting 通病 (view-extrapolation degradation)，非 T8.13 bug**，写入 README_4D 已知限制

- **T8.14** (5/22 Mac + vast.ai 4090 Norway $0.83, commit `1c72fd8`)：viser_gui_4d "Gaussian Layers" 运行时按层开关。Render Controls 子 folder 加 checkbox per layer（background / road / dynamic_rigids / sky_envmap），unchecking → `fused_view` 跳过禁用层 + `_blend_sky` 短路 → 从不进入 OptiX → 干净 + 零开销。**Mac 200/200 PASS + 1 skipped 0 回归**，vast.ai 4090 9-screenshot 验收全 PASS 跨 2 ckpt variant（pinhole schema_v1 + FTheta schema_v2）：全关 66.9 FPS (+31.7%)

---

## 6. 关键突破时刻

这些是"调查 → 解决"的临界点。

### `#171` PATH reset (5/8 04:14:52, 距 #170 crash 24 秒)
24 秒从根因诊断到完整修复 commit。后来上升为 CLAUDE.md hard rule。

### `#192` 2s smoke PSNR 34.85 dB (5/9 ~05:30)
第一次证明 3dgrut2 fork 在小 clip 上可以匹配 NuRec 量级。整个项目第一个真实可用的 checkpoint。

### Stage 6-fix masked PSNR 突破 (T6F.3, 5/20)
full=20.49 / masked=29.49，**+9 dB**。**metric 端揭开了真实信号**——历史 Stage 3-6 baseline 的"PSNR 26 dB"绝大多数是 ego 车身水分。这是整个 v2 项目最重要的方法论突破。

### T7.3.b ExposureModel OFF ablation (5/21)
cc_psnr_masked 24.70 vs 24.75 几乎一致 + raw_psnr_masked OFF 25.76 vs ON 15.63 +10.13 dB → 证明 ExposureModel 在长训练下走向"两网络退化短路径"，cc_psnr 才是真实重建质量 KPI。

### T8.12-FIX Phase A.5 fisheye 120° 结构性突破 (5/21 vast.ai California)
当 equirectangular fisheye 把 cuboids 投到了合理位置但内容仍模糊 → 团队立刻确认问题是**结构性 intrinsic mismatch**，不是 fov 调参。把后续路径锁死到 T8.13 schema v2。

### T8.13 tracer.py:471 发现 (5/21)
读 3dgut tracer 源码发现 UT rasterizer **早就原生支持 FTheta**——只要 Batch 里塞对 8-key dict，kernel 自己分流。整个 T8.13 变成 pure-Python（0 行 C++/CUDA），把原本可能要 5 天的工作压到 1 天。

### T8.13 vast.ai 清晰街景截图 (5/21)
"HEAT" 广告牌 + fisheye 桶形畸变可识别——视觉验收一锤定音，三天 fisheye saga 完美收尾。

---

## 7. 工作节奏与模式

### Phase A — 3 天 investigation+infra sprint
- **213 obs / 600K tokens** / 138 discovery / 27 feature / 28 change / 15 decision / 4 bugfix
- 每天有完全不同的性格：Day 1 = 环境 debug，Day 2 = 代码交付，Day 3 = pivot + 训练
- 工作密度极高：5/7 一天 112 条 obs（4 月最多）
- 转向决策快：Day 3 凌晨 02:40 决定放弃 4090 转 Vast.ai，3 小时内完成转移 + 2s smoke 训练成功

### Phase B — 9 天 execution marathon
- 清晰的 stage gating，每个都有 KPI 出口
- 标准模式：**Mac 设计 → Mac 单测过 → A800 5k smoke → A800 更长 run / 30k**
- Stage 3-6 都在 1 天左右出口
- Stage 7 (真 KPI integration) 1 天但暴露 ExposureModel failure mode
- Stage 8 跑了 3 天因为 viser/fisheye saga 发现了 3 个 distinct intrinsic-mismatch bugs

### GPU 并行模式
- **A800-x2 重度使用**：GPU 0/1 拆分训练和 aux 生成；nre-tools 和 3dgrut training 同时跑
- **Vast.ai 作为一次性 4090**：每次 Stage 8 视觉验收租 1-2 小时，单次 ~$1 (~¥7)

---

## 8. 挑战与排查长征

### 4090 SSH PATH 泄漏 (Phase A)
macOS Claude session 通过 SSH env forwarding 把含空格的 PATH 灌进远端 bash → set -e 失败。永久 fix 写进 CLAUDE.md hard rule（永远显式 reset PATH）。

### Vast.ai HOME bug + tmux -tt 必需 (Phase A)
两个 compounding 问题，Day 3 一起解决。

### A800 conda activation (Phase B, 5/22 hardened in commit `53d3ac9`)
ssh non-interactive shell 不继承 conda PATH → `slangc` / `python` / CUDA env 全断。教训上升到 CLAUDE.md："每个 `ssh a800-x2` 命令开头都必须先 `source /root/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut`"。

### ExposureModel 退化解 (Stage 7)
最 subtle 的 bug —— raw PSNR 15.63 看起来灾难性，但 cc_PSNR 24.75 几乎不变。只有 T7.3 vs T7.3.b 的 expressive ablation 才暴露真相：长训下 ExposureModel 走"两网络退化短路径"，让高斯只学个大概形状再让 affine compensate。修法不是改 ExposureModel 本身，而是把这个发现整合成 V3-P1 与 bilateral-grid 一起重做。

### Fisheye saga (T8.12 → T8.12-FIX → T8.13)
3 天 / 3 vast.ai 租用 / ~$2.50。Phase A.2 (pinhole fov correction) 证伪 fov 假设。Phase A.5 (equirectangular fisheye) 结构性确认是 intrinsic mismatch。T8.13 final fix 干净的 pure-Python schema 扩展 + 写库 tracer 端早已原生支持 FTheta。

### A100/A800 RT cores 被 laser (T8.10)
Ampere 数据中心 SKU 上 RT cores 被 LASERED OFF（这是 NVIDIA 的硬件分级策略）→ OptiX dlopen segfault → A800 永远渲染不了 Gaussian。Workaround：`--no_gaussian_render` 模式只显示 scene primitives；正式渲染要去 Hopper (H100) 或 RTX (4090)。

---

## 9. 技术债务与记忆资本

### 已偿还
- ✅ SSH PATH hard rule (Phase A → CLAUDE.md)
- ✅ A800 conda activation hard rule (commit `53d3ac9`)
- ✅ 双指标 (psnr/ssim/lpips × full/masked) 全栈接通 (Stage 6-fix)
- ✅ 显式 `inject_viz_4d` CLI 用于 retrofit 老 ckpt
- ✅ trainer.py + render.py 两处改动点 metric-end-to-end 接通规则 (CLAUDE.md §B)
- ✅ A800 任务出口三重证据（metrics.json 数字 + commit hash + Done Log 回填）规则 (CLAUDE.md §C)

### 已识别但延后
- ⏭ ExposureModel 退化解 → **V3-P1**（与 bilateral grid 整合研究，预估 3-5 天）
- ⏭ A800 不能渲染 Gaussian → workaround `--no_gaussian_render`，硬件方案待 H100 时考虑
- ⏭ orbit-away view-extrapolation degradation → 写入 README_4D 已知限制，backlog 加 GUI lock 沿 ego_poses_c2w spline 强制相机轨迹
- ⏭ region PSNR 拆解工具（路面/动态/背景）→ V3-E2 (per-class cPSNR 评测工具)

### v2 设计明确 out-of-scope
- ❌ track pose learning（pose 不进 Parameter）
- ❌ Cosmos-DiFix
- ❌ C++ tracer 改动
- ❌ bilateral grid（v2 只占位 ExposureModel）
- ❌ DynamicDeformable particles（v2 占位 only）

---

## 10. 持久记忆与连续性的作用

诚实评估：

### Phase A 几乎没用记忆主动召回
- **0 explicit recall events**（没有任何 search / get_observations / timeline 查询事件）
- 4 个 session 都依赖 claude-mem 的**被动 context injection**——每 session 起手注入约 50 obs 的相关历史

但是 `#181-184` 那一组 4 条高 token discovery（共 94K tokens）在 Day 3 cold-start 时**自己重读了一遍 pyproject.toml / codebase / roadmap / dataset API**——这意味着即使没有主动召回，Claude 也在重复同一类知识的吸收。这是一种"被动记忆 + 主动重读"的混合模式。

### Phase B 改用 kanban-driven memory
从 5/14 开始，项目工作方式发生**根本性转变**：从 claude-mem-driven 转向 **`v2_plan.md` + `v2_architecture.md` 双活文档**驱动。CLAUDE.md 在那一周加固："更新文档 = task 完成的 Step N+1"。

这种转变的核心原因是 claude-mem 不再为这个项目自动写库（原因不明，可能用户禁用了 hook 或工作转移到不在 capture 列表的 worktree）。但项目没有"丢失记忆"——它把记忆从被动 auto-compression **迁移到了主动的文档 curation**。Done Log 1174 行里每个 task 都有 PSNR / commit / timing 数据，比 claude-mem 提供的 narrative 还要硬核。

### 进程性洞察
2 周横跨两种记忆模式：
- Phase A: 600K discovery tokens → 213 条 compressed obs，每条 narrative+facts+concepts 三段化
- Phase B: ~80 commit + 1174 行 Done Log 手写，每个 task 必带 commit hash + 实测数字

后者的人为 curation 成本明显更高（要写明 commit / 跑 metrics / 回填看板），但准确度也更高（没有 LLM compression 噪声，全是 ground truth 数字）。

---

## 11. Token Economics & Memory ROI

| Phase | Date range | Obs | Total discovery tokens | Sessions | Avg discovery / obs | Avg read / obs | 压缩比 | Explicit recalls |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A (claude-mem)** | 5/7-5/9 | 213 | 600,924 | 4 | 2,821 | 343 | ~8.2× | **0** |
| **B (v2_plan+git only)** | 5/14-5/22 | — (未记录) | — | — | — | — | — | — |

### Phase A ROI 诚实评估

- 213 obs × avg 2,821 discovery tokens = ~600K tokens of work compressed
- 如果未来 session 要 recover 这部分，会收到约 50 obs 的 context injection (~14K tokens) per session start。保守 30% relevance → ~4.2K tokens of "prevented re-work" per cold start
- 4 个 session 全部在连续 3 天里 → 3 个 session 可能受益于注入（≈12.6K tokens 防重做）
- 加上 **0 explicit recall** — 用户/Claude 从未主动搜索过 claude-mem
- **Net**: 被动注入大概省了 ~10K tokens 的重读。相对 600K 原始成本，是 1.7% recovery
- 真正的长期价值在 **journey-into-3dgrut2.md (5/9 第一版) + WP_V1-1_Report.md** 这两个 durable knowledge artifact——用户会反复回读

**Top-5 highest-value memories**（最有可能未来 recall）：
1. `#183` pyproject.toml 结构 (37,920) → 安装模式
2. `#182` missing tools/tests + NCore meta 格式 (22,700) → 架构事实
3. `#181` oss-sim 路线图 (19,971) → 规划锚点
4. `#162` env.tar 含 CUDA 12.8.1 (16,346) → 环境技巧
5. `#192` 首次 PSNR 34.85 (15,523) → baseline 数字

### Phase B 是更大的问题

8 天高强度工作，57 done task，~80 commit —— **claude-mem DB 中一条没有**。"v2 记忆"完全活在 `v2_plan.md` (2343 行) + `v2_architecture.md` + journey 报告 + WP_V2_Report.md 这几个 active 文档里。

项目明确用"task 完成 = doc update"hard rule 补偿这一点。实践中很有效（Done Log 信息密度极高，PSNR / commit / 时间应有尽有）—— 但代价是未来 Claude session 要从 0 重建 context 时**必须读 2343 行 markdown**。这种 trade-off 在 Phase A 是隐式的，在 Phase B 是显式的设计选择。

---

## 12. 时间线统计

- **时间跨度**: 2026-05-07 04:00 UTC → 2026-05-22 (last commit `8187a51`，T8.14 vast.ai 4090 视觉验收)
- **Phase A**: 3 天 / 213 obs / 4 sessions / 600,924 discovery tokens / 0 explicit recalls
- **Phase B**: 9 天 / ~57 done tasks / ~80 commits / 9 stages
- **Phase A 最活跃日**: 5/7 (111 obs in `quizzical-mendel-60fc6b`)
- **Phase B 最活跃日**: 5/19 (Stage 3 + Stage 4 同一天双出口，~12 commits)
- **最长 debugging saga**: fisheye 3 天弧 (T8.12 → T8.12-FIX → T8.13, 5/20-21)
- **总 vast.ai 4090 花费**: ~$2.50-3.00 across 3 视觉验收 runs
- **总 A800 GPU 时间**: 估计 ~12-15 小时（Stage 0-7 累计 + Stage 8 几次 smoke）
- **总 Mac 单测覆盖**: 200/200 PASS + 1 skipped (T8.14 末态)

---

## 13. 教训与元观察

如果让一个新工程师从这个时间线里学一件事，最有价值的不是某个技术细节，而是这些方法论：

### 1. 环境是一半的战斗
Phase A 3 天里有 1 天完全用在 4090 环境上。Phase B 第一条 hard rule 的更新（CLAUDE.md conda activation）也是从反复的痛点里提炼出来的。**重要观察**：当一个项目的 stack 跨 Mac 本地 + A800 远程 + vast.ai 临时 + 多个 worktree 时，环境差异 detection 应该是 day-zero 的自动化测试，不是事后救火。

### 2. 测试 metric 之前不要信它
Stage 6 看起来灾难（raw PSNR 加 ExposureModel 后掉 9 dB）。Stage 6-fix 证明**是 metric 错了**（ego mask leakage），不是模型错了。这种"先测 metric 自身"的纪律在 driving 3DGS 这种领域尤其关键，因为图像里有大量 "靠位置/颜色固定"的容易区域（车身/天空），全图 PSNR 经常虚高。

### 3. 用 ablation 找真相
T7.3 vs T7.3.b 是唯一暴露 ExposureModel 退化解的方法。cc_psnr 单独看完全看不出来（两组 24.75 vs 24.70 几乎一致）。Ablation 比单一指标可信。

### 4. Pure-Python 能赢就赢
T8.13 终结 3 天 fisheye saga，**0 行 C++/CUDA 改动**——只因为读了 `tracer.py:471` 发现 kernel 早就支持 FTheta。先读代码再 recompile，永远先读代码。

### 5. 活文档 > 被动记忆（长周期工作）
9 天 / 80 commit 的 Phase B 没有任何 claude-mem 捕获，但 traceability 反而比 Phase A 好——因为 `v2_plan.md` Done Log 强制结构化回填 PSNR + commit + 时间。**长周期工作 (> 1 周) 应该有显式 active doc，不能只依赖 LLM auto-compression**。

### 6. A800 的 RT cores 阉割是 Specific Hardware Fact
不要在 A100/A800 上租 OptiX rasterization 任务——会 crash。Hopper / RTX 才行。这种硬件细节没有显式 spec sheet，必须靠踩坑学到。

### 7. 视觉验收 matters
render.py PSNR ≥ 25 dB 看起来一切正常——但只有 vast.ai 4090 上的 side-by-side fisheye 视觉对比才暴露 viewer 用错了 intrinsics。数字 OK 不代表用户接受 OK。Stage 8 后期的"vast.ai 9-screenshot 验收"模式应该成为类似项目的标准 QA。

### 8. 用户反馈的速度修正
`#209 (5/9 07:51)` 用户对 KPI 分析的"推理太草率"反馈，直接影响了 Phase B 整体的诊断风格——Stage 7 ExposureModel 失控诊断、T8.12-FIX fisheye 多假设排查，都明显带着"先 ablation，再下结论"的痕迹。**单一条用户反馈如果被认真消化，可以塑造未来几周的工作风格**。这是 Phase A → Phase B 之间最重要的文化迁移。

---

*End of report. 总字数 ≈ 7,500 字。本报告由 Claude Opus 4.7 基于 claude-mem SQLite DB (213 obs) + v2_plan.md Done Log (1174 lines) + v2_stage_detail.md (582 lines) + git log 综合生成，2026-05-22。*
