# T8 viser_gui_4d Bug List

**最后更新：** 2026-05-25 19:33 GMT+8
**对应代码版本：** worktree `worktree-distributed-beaver` @ `4de6658` (训练稳定性修复后) — **9/9 bug 全部关闭** (Phase B + E 12 commits + 稳定性修复 1 commit)
**Plan 文档：**
- 旧：[`/Users/etendue/.claude/plans/v2-t8-13-t8-14-bug-happy-starfish.md`](/Users/etendue/.claude/plans/v2-t8-13-t8-14-bug-happy-starfish.md)
- B3 详细：[`/Users/etendue/.claude/plans/t8-viser-gui-4d-distributed-beaver.md`](/Users/etendue/.claude/plans/t8-viser-gui-4d-distributed-beaver.md)

---

## 运行环境

### Mac (开发 + scp relay)
- 仓库：`/Users/etendue/repo/3dgrut2/`
- venv：`./.venv/` (python 3.14, 仅跑 pytest)
- Mac 单测：`source .venv/bin/activate && python -m pytest threedgrut/tests/ -x`

### ThinkPad (yusun, RTX 4090 24GB)
- ssh 别名：`thinkpad`（~/.ssh/config）
- 网络：跟 A800 不能直连，必须 Mac 中转 (`scp -3` 或两段 scp)
- 仓库：`/home/yusun/repo/3dgrut2/`
- conda env：`3dgrut2` (`~/miniconda3/envs/3dgrut2/`)，torch 2.11.0+cu128
- viser 启动（用户最近一次成功命令）：
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3dgrut2 && \
  cd ~/repo/3dgrut2 && \
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python -u threedgrut_playground/viser_gui_4d.py \
    --gs_object /home/yusun/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt \
    --port 8090 --target_fps 10
  ```
- 当前 ckpt：`/home/yusun/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt` (954 MB, schema_v2 + FTheta + 70 tracks + 19.93s)
- Mac → ThinkPad SSH tunnel：`ssh -f -N -L 8090:localhost:8090 thinkpad`，浏览器开 http://localhost:8090

### A800 (训练 + ckpt 源)
- ssh 别名：`a800-x2`
- 仓库：`/root/work/yusun/repo/3dgrut`（rsync mirror，**`.git` 不同步**，submodule 链接破损）
- conda env：`3dgrut` (`/root/miniforge3/envs/3dgrut/`) — py311 + torch 2.1.2+cu118 + kaolin 0.17.0 + fused-ssim + slangtorch + mlp sky_backend fallback (nvdiffrast 内网装不上)
- env 备份（conda-pack）：`/root/work/yusun/envs/3dgrut-env-py311-cu118-20260523_182456.tar.gz` (8.6 GB) — ⚠️ 这份 backup 是补 fused-ssim **之前**的，下次重建需在装完所有补丁后重新 pack
- 数据集：`/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc-e87b-41a7-8e85-71772f9603d7/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json`
- 输出根：`/root/work/yusun/ncore-nurec/output/`
- 训练命令模板：
  ```bash
  ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH && \
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    cd /root/work/yusun/repo/3dgrut && \
    nohup python -u train.py --config-name apps/ncore_3dgut_mcmc_v2_full_4dviz \
      path=<dataset.json> \
      out_dir=<out_dir> \
      experiment_name=<label> \
      n_iterations=30000 \
      trainer.sky_backend=mlp \
      > <log> 2>&1 & disown'
  ```
- GPU：两张 A800-SXM4-80GB，driver 535.129.03 → CUDA 12.2 max（cu118/cu121 OK，cu128+ 不兼容）
- 长任务必须 `nohup setsid disown` + sentinel + 轮询，**不能用 ssh heredoc**（SIGHUP 杀进程）

---

## Bug 状态总览

| ID | 名称 | 现状 | 优先级 |
|---|---|---|---|
| B1 | Play 时视角不跟 ego | ✅ **已修** (commit 209886c) | — |
| B2 | cuboid 与 Gaussian 不重合 | ✅ **已修** (commit `2e12a1b` FTheta projector helper + Phase E.10 实测 wireframe 紧贴车) | — |
| B3 | dynamic_rigids toggle 无效 | ✅ **已修** (Phase B + E 全套，commits `7f8bb17` → `7d5be05`) | — |
| B4 | 训练只覆盖 ~2s 短 clip | ✅ **已修** (commit 1.9s → 19.9s 全时长 30k 重训) | — |
| B5 | inject CLI 不该是新 ckpt 必需步骤 | ✅ **已修** (commit `b1752b3` 修 uint64 + 1k smoke 实测 ckpt 自带 FTheta block 无需 inject) | — |
| B6 | viz_4d 中 ego trajectory 按 camera 拼接，非时间连续 | ✅ **已修** (commit `46e643f`) | — |
| B7 | Active cuboids checkbox 取消后 Play 仍 render | ✅ **已修** (commit `46e643f`) | — |
| B8 | Dynamic LiDAR checkbox 状态不对 | ✅ **已修** (commit `46e643f`) | — |
| B9 | FTheta extraction 在 numpy.uint64 上静默失败 | ✅ **已修** (commit b1752b3) | — |

---

## 详细 Bug 列表

### ✅ B1 — Follow Ego（已修）

**现象**：Play 时 viewer 自由相机不动，画面跟车飘但相机静止。
**根因**：`_on_time_change` 只更新 `h_ego_frustum` scene primitive，从未写过 `client.camera.position/wxyz`。
**修复**：commit `209886c`
- `Viser4DViewer.__init__` 加 `self._follow_ego_enabled: bool = False`
- `_build_visibility_gui` 末尾加 "Follow Ego" checkbox（默认 OFF）
- 新增 `_snap_clients_to_ego(t_us)` 方法
- `_on_time_change` 末尾按 flag 调用 snap
- 4 个 Mac 单测 PASS（`test_viser_gui_4d_follow_ego.py`）

**用户实测**：✅ 已确认。

---

### ✅ B2 — cuboid 与 Gaussian 不重合（已修）

**现象**：浏览器画面上 cuboid wireframe 与 Gaussian 渲染的真实物体（如白色卡车）在屏幕坐标上**偏移**，越靠边缘偏得越远。

**根因**：投影模型不一致 — Gaussian backdrop 走 FTheta polynomial（鱼眼，140° FOV），viser 的 `add_line_segments` cuboid 走 pinhole（Kaolin Camera fov）。

**修复**：commit `2e12a1b` (FTheta projector helper) + Phase E.10 `5bc878c` (validate_frame_0 验证)
- `threedgrut_playground/utils/ftheta_projector.py` — polynomial-based forward projection (`FthetaForwardProjector` + Shepperd 数值稳定 + `project_polylines` 含 polyline subdivision)
- viser_gui_4d.py 中 cuboid wireframe path 已接入 FTheta
- Phase E.10 frame-0 验证（`docs/T8_artifacts/E10_frame0_init/out_1_cuboids.png`）：蓝色 cuboid wireframe 紧贴前景银 SUV + 背景所有车的 box 都正确对位

**用户实测（B3 1k ckpt viser）**：✅ 各 active cuboid wireframe 都跟实车对齐，无明显偏移。

---

### ✅ B3 — dynamic_rigids toggle 无效（已修，Phase B + E 完整链路）

**现象**：浏览器 Gaussian Layers folder 勾掉 `dynamic_rigids` checkbox → 车辆 Gaussian **不消失**；勾掉 `background` 反而车辆**跟着消失** → 视觉证据是车在 bg 层而非 dyn 层。

**追溯到的 6 个独立 bug**（Phase A → E 全部修完）：

| # | 根因 | 修复 commit | 文件 |
|---|---|---|---|
| 1 | **MCMC per-layer scoped, dynamic_mask 无层归属约束** | `7f8bb17` (Phase B) | `threedgrut/model/bg_cuboid_loss.py` 新增 3D opacity penalty，把 bg 层在 cuboid 内的粒子 push 到 dead 阈值 |
| 2 | **NCore cuboid 旋转被丢弃** — `tracks_loader.py:195` 写死 `pose = np.eye(4)`，车辆 yaw 全部按 identity rot 处理 | `40875a5` (Phase E.2) | `tracks_loader.py:euler_xyz_to_rotation_matrix` 把 `bbox.rot` 解为 intrinsic XYZ Euler 写入 pose[:3,:3] |
| 3 | **fused_view rotation 没复合** — E.2 让 pose 含旋转后，per-particle local rotation 仍按 world 轴解释 → MCMC 用 6 米巨型 scale 补偿方向错位 | `2987d12` (Phase E.2.b) | `layered_model.py:_transform_means_and_active` 新增 `q_world = q_pose ⊗ q_local` 复合 |
| 4 | **Inactive 帧 pose 是 identity** — tracks_loader 用 `np.eye(4)` 初始化所有帧，inactive track 粒子被塌到 world 原点 | `368c87d` (Phase E.3) | `_transform_means_and_active` 返回 active_mask；fused_view 把 inactive 粒子 density 改 -50 → sigmoid≈0 |
| 5 | **`track_ids` buffer ckpt roundtrip 丢失** — `MoG.get_model_parameters` 不存这个 buffer，viewer 加载后 `_transform_means` 拿不到 per-particle owner | `1594396` (Phase E.4) | `LayeredGaussians.get_model_parameters` + `init_from_checkpoint` 在 wrapper 层序列化 track_ids |
| 6 | **Inference 自由相机无时间戳 fallback** — `timestamp_us<=0 and frame_id is None` 时不变换 dyn 位置 → 渲染崩 | `7d5be05` (Phase E.2.c) | per-track 各自找第一个 active 帧做 fallback pose |

**附加工具**：
- `7f8bb17` (B7): `dyn_clamp_to_cuboid` 在 `_post_optimizer_step` 末尾把 dyn positions 钳回 `|local|≤size/2`
- `7f8bb17` (B7): dataset 端 `dyn_mask_cuboid` 用 FTheta 投影替代 sseg
- `add202a` (Phase A): `scripts/diagnose_bg_in_cuboid.py` 量化 bg 层粒子误入 cuboid 比例
- `f446f43` (Phase E.1): `scripts/diagnose_dyn_per_cuboid.py` per-track alive_pct + outlier 距离
- `b00dddf` (Phase E.5+E.6): `threedgrut/model/class_psnr.py` per-cuboid PSNR 指标，trainer/render.py 双路径接入 metrics.json
- `5bc878c` (Phase E.10): `scripts/validate_frame_0.py` 把 cuboid wireframe / sseg / LiDAR init / Gaussian centers 同时投影到第一帧，验证 init 对齐

**KPI 验证**（A800 1k smoke `B3_E2b_1k_20260525_114457`）：

| 指标 | 30k baseline | 5k broken (no E.2.b) | **1k Fix (Phase E)** |
|---|---:|---:|---:|
| bg_inside_pct | 10.17 % | 5.72 % | (待 30k 重测) |
| dyn alive_pct | n/a | 22.0 % | **84 %** ✅ |
| dyn scale max | n/a | 6.88 m | **0.22 m** ✅ (-31×) |
| dyn outside_cuboid | n/a | 2281 m max | **0 m** ✅ |
| mean_class_psnr | 18.73 dB | 17.82 dB | **19.13 dB** ✅ +0.40 |
| automobile_psnr | 18.70 | 17.61 | **19.01** ✅ |
| heavy_truck_psnr | 18.52 | 20.16 | **20.26** ✅ |

**视觉验证（用户实测，ThinkPad viser + 1k ckpt）**：✅ 勾掉 `dynamic_rigids` checkbox → 车辆区域清空；勾掉 `background` checkbox → 车辆保留。**toggle 双向独立工作**。

**Frame-0 投影验证**（`docs/T8_artifacts/E10_frame0_init/*.png`）：cuboid wireframe 紧贴 SUV / sseg mask 完整覆盖车身 / dyn LiDAR 点聚集在车上 / bg 点避开 cuboid 内部 → init 阶段四套数据完全对齐。

**Mac 单测**：342 passed, 1 skipped, 0 regression（新增 88 个测试覆盖 E.1-E.6 + E.2.b/c + E.10）。

---

### ✅ B4 — 训练只覆盖 ~2s 短 clip（已修）

**现象**：旧 ckpt (`v2_ftheta_20260520_113746/ckpt_with_viz_4d_v2.pt`) viz_4d duration **1.9s, 51 ego frames, 31 tracks**，用户误以为 20s clip 但后 15s "模糊"。诊断 A.2 确认 metadata 范围就只到 ~2s。
**根因**：旧 ckpt 训练时 `dataset.train.duration_sec=2.0`（短 clip smoke 测试残留）。
**修复**：A800 重训 `bug4_v2_full_30k_20260523_184318`
- `n_iterations=30000` 全步
- `dataset.train.duration_sec=-1` 默认（全 20s clip）
- 全 5 个 camera 训练（front_wide / rear_tele / cross_left / cross_right / rear_left）
- 用时 64 min @ A800 单卡
- **新 ckpt：schema_v2, duration 19.93s, 2623 ego frames, 599 track frames, 70 tracks, FTheta 全字段**
- metrics: `mean_cc_psnr_masked: 24.65 dB`（v2 真实重建上限 baseline）

**用户实测**：✅ slider 真到 19.988s。

---

### ✅ B5 — inject CLI 不该是新 ckpt 必需步骤（已修）

**现象**：理论上 `trainer.save_checkpoint` 在 `viz_4d.enabled=true` 时自动调 `extract_4d_metadata`，写完整 schema_v2 + FTheta。但旧训练后 ckpt 里 `FTheta present: False`，必须额外跑 `python -m threedgrut.viz.inject` 才能补 FTheta。

**根因**：B9 同源（uint64 异常）的连锁反应—`extract_4d_metadata` 在训练 save 路径下静默落到 `ftheta_dict=None`，schema_v2 占位但 FTheta 8-key 没填。

**修复**：commit `b1752b3` 修了 numpy.uint64 → int64 cast。

**实测验证（Phase E 1k smoke `B3_E2b_1k_20260525_114457`）**：✅ 新训练 ckpt `B3_E2b_1k.pt` 直接含完整 `viz_4d` block (schema_v2 + 8-key FTheta + 70 tracks)，**无需 inject** 就能直接被 viser_gui_4d 加载并正确渲染。`[T8.13-DIAG]` 启动日志确认 `tracks_poses dict: 70 tracks`、`FTheta intrinsics 已加载 (resolution=(1920, 1080), max_angle=1.221rad)`。

---

### ✅ B6 — viz_4d ego trajectory 按 camera 拼接，非时间连续（已修）

**现象**：旧版 Play 时**视角在 0-5s 是前视，5-10s 切到后视**…完全不是真实 ego 飘动。

**根因**：`threedgrut/viz/metadata.py:_extract_ego` 按 `camera_ids` 循环 concat 所有相机的 timestamps + poses，没去重也没按时间排。`n_ego_frames: 2623 = 524 × 5 camera` 而非 unique 525。

**修复**：commit `46e643f` (B6+B7+B8 同 PR)
- `_extract_ego` 只取 primary camera 的帧（或按 timestamp dedupe + sort）
- 单测 mock 5 camera × 524 帧 → 验证输出 ≤ 525 unique stamps

**用户实测（B3 1k ckpt）**：✅ ego trajectory 流畅连续，Play 不再跨相机跳切。

---

### ✅ B7 — Active cuboids checkbox 取消后 Play 仍 render（已修）

**现象**：勾掉 Visibility "Active cuboids" checkbox 后，**Play 一推进帧，cuboid 又重新出现**在画面里。

**根因**：`_update_active_cuboids` 每帧 remove + re-add `h_cuboid_lines`，新 line_segments 默认 visible=True，没读 `self.show_cuboids.value`。

**修复**：commit `46e643f` (B6+B7+B8 同 PR) — `_update_active_cuboids` 加 `prev_visible` preserve 逻辑，镜像 `_update_dynamic_lidar` 模式。

**用户实测（B3 1k ckpt）**：✅ 勾掉 Active cuboids 后 Play 不再重新出现 wireframe。

---

### ✅ B8 — Dynamic LiDAR 初始 checkbox 不勾但点云已显示（已修）

**现象**：初始状态 Visibility folder 里 "Dynamic LiDAR" checkbox 未勾选，但浏览器画面上 LiDAR 点云已显示。

**根因**：`_update_dynamic_lidar` 中 `prev_visible` 在首次调用时 `h_dyn_pts is None` → 硬编码 fallback `True`，忽视 `show_dyn_pts.value`。

**修复**：commit `46e643f` (B6+B7+B8 同 PR) — `prev_visible` 默认值从硬编码 `True` 改为 `bool(getattr(self, 'show_dyn_pts', None) and self.show_dyn_pts.value)`。

**用户实测（B3 1k ckpt）**：✅ 初始 checkbox 状态跟实际渲染一致。

---

### ✅ B9 — FTheta extraction 在 numpy.uint64 上静默失败（已修）

**现象**：训练 + inject 后 ckpt 里 `viz_4d.ego.primary_camera_intrinsics_FTheta = None`，schema_v2 占位但 FTheta 8-key 全空 → viser 自动落回 pinhole approximation，T8.13 修的 fisheye 投影完全失效。
**根因**：[`threedgrut/viz/metadata.py:124`](threedgrut/viz/metadata.py)
```python
"resolution": _to_cpu_int64(torch.as_tensor(params.resolution)).numpy(),
```
NCore FTheta `params.resolution` 是 `numpy.uint64`（e.g. `[1920, 1080]`），torch **没有 native uint64 dtype** → `torch.as_tensor()` 抛 TypeError → 外层 try/except 静默吞 → `ftheta_dict=None`。
**日志证据**：`[WARNING] FTheta intrinsics extraction failed: can't convert np.ndarray of type numpy.uint64 ... ; ftheta_dict=None`

**修复**：commit `b1752b3`
```python
"resolution": np.asarray(params.resolution, dtype=np.int64),
```
跳过 torch round-trip，直接 numpy cast。
**Mac 单测**：26/26 PASS（test_viz_4d_metadata + test_inject_viz_4d + test_ftheta_intrinsics）
**A800 实测**：重 inject 后 FTheta 8 keys 全齐，max_angle=1.221 rad = 70° 半视场。

**待补**：单测覆盖 NCore FTheta `params.resolution` 是 uint64 的情况，防回归。

---

## 提交计划

按优先级合并：
1. ✅ **PR #1**：B6 (ego trajectory dedupe) + B7 (cuboid visibility preserve) + B8 (LiDAR init state) — commit `46e643f`，三 bug 同 PR
2. ✅ **PR #2**：B3 完整修复链（Phase B + E.1-E.10 + E.2.b + E.2.c）— 12 commits 从 `add202a` → `7d5be05`
3. ✅ **PR #3**：B2 FTheta cuboid overlay — commit `2e12a1b` (forward projection helper) + Phase E.10 验证 wireframe 与 FTheta backdrop 对齐
4. ✅ **B5 自动验证**：1k smoke `B3_E2b_1k.pt` 实测无需 inject 即含 FTheta block（commit `b1752b3` 修复链打通）

## 9-bug 最终状态 ✅ 9/9 全闭合

| ID | 现状 | 完成 PR |
|---|---|---|
| B1 Follow Ego | ✅ | `209886c` |
| B2 cuboid 与 Gaussian 不重合 | ✅ | `2e12a1b` + Phase E.10 验证 |
| **B3 dynamic_rigids toggle 无效** | ✅ | `7f8bb17` → `7d5be05` (12 commits) |
| B4 训练只覆盖 ~2s 短 clip | ✅ | `bug4_v2_full_30k_20260523_184318` |
| B5 inject CLI 必需 | ✅ | `b1752b3` + 1k smoke 自动产 FTheta ckpt 验证 |
| B6 ego trajectory 按 camera 拼接 | ✅ | `46e643f` |
| B7 Active cuboids checkbox 取消后再现 | ✅ | `46e643f` |
| B8 Dynamic LiDAR 初始状态错误 | ✅ | `46e643f` |
| B9 FTheta extraction uint64 静默失败 | ✅ | `b1752b3` |

**T8 viser_gui_4d bug list 全部关闭** — 用户在 B3_E2b_1k ckpt + Phase E 代码上实测确认 9 个 bug 行为均符合预期。

## 验证步骤模板

ThinkPad 浏览器验证：
1. SSH tunnel：`ssh -f -N -L 8090:localhost:8090 thinkpad`
2. 浏览器 http://localhost:8090
3. 查终端 `/tmp/viser_4d.log` 看 `[T8.13-DIAG]` 四段诊断 + `[BUG3-DIAG]` callback 触发记录
4. 截图记录到 `docs/T8_artifacts/<bug_id>_<state>.png`
