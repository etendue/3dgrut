# T8 viser_gui_4d Bug List

**最后更新：** 2026-05-24 10:52 GMT+8
**对应代码版本：** main @ `b1752b3` (FTheta uint64 fix) + `d965b00` (Bug 3 guard removal) + `209886c` (Follow Ego)
**Plan 文档：** [`/Users/etendue/.claude/plans/v2-t8-13-t8-14-bug-happy-starfish.md`](/Users/etendue/.claude/plans/v2-t8-13-t8-14-bug-happy-starfish.md)

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
| B2 | cuboid 与 Gaussian 不重合 | ⚠️ **待修** (FTheta vs pinhole 投影) | P1 |
| B3 | dynamic_rigids toggle 无效 | ⚠️ **代码侧已修但行为仍异常** | P1 |
| B4 | 训练只覆盖 ~2s 短 clip | ✅ **已修** (commit 1.9s → 19.9s 全时长 30k 重训) | — |
| B5 | inject CLI 不该是新 ckpt 必需步骤 | ⚠️ **新 bug** | P2 |
| B6 | viz_4d 中 ego trajectory 按 camera 拼接，非时间连续 | ⚠️ **新 bug** | P1 |
| B7 | Active cuboids checkbox 取消后 Play 仍 render | ⚠️ **新 bug** | P2 |
| B8 | Dynamic LiDAR checkbox 状态不对 | ⚠️ **新 bug，需澄清** | P3 |
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

### ⚠️ B2 — cuboid 与 Gaussian 不重合（待修，P1）

**现象**：浏览器画面上 cuboid wireframe 与 Gaussian 渲染的真实物体（如白色卡车）在屏幕坐标上**偏移**，越靠边缘偏得越远。
**根因（已确认）**：投影模型不一致
- Gaussian backdrop 由 engine 走 **FTheta polynomial 多项式投影**（鱼眼，桶形畸变，140° FOV）
- viser 自己画的 `add_line_segments` cuboid 走 **pinhole 投影**（Kaolin Camera fov）
- 两套投影对同一个 3D 点会算出不同的 2D 像素位置
- 诊断块 A.4 段已警告此点

**修法（Phase D-a）**：
- 新增 `threedgrut_playground/utils/ftheta_projector.py`，用 polynomial helper 在 Python 端把 cuboid 3D 边投到 FTheta 2D 像素
- 渲染到 RGBA overlay 图，叠加在 Gaussian backdrop 之上
- 关闭 viser line_segments 的 cuboid/frustum/track 路径
- 工作量：~0.5d

**用户实测**：✅ 未修（FTheta ckpt 上仍偏移）。

---

### ⚠️ B3 — dynamic_rigids toggle 无效（代码修了但行为仍异常，P1）

**现象**：
- 浏览器 Gaussian Layers folder 勾掉 `dynamic_rigids` checkbox → 车辆/行人 Gaussian **不消失**
- 勾掉 `background` checkbox → 车辆/行人 **跟着背景一起消失**
- 终端日志 `[BUG3-DIAG]` 行确认 callback 真的触发，`enabled_layer_names` 真的被改

**已做修复**：commit `d965b00`
- `engine.py:1050` 移除了 `len(self.scene_mog.tracks_poses) > 0` guard，让所有 LayeredGaussians 都走 forward → fused_view → enabled_layer_names 过滤
- `viser_gui_4d.py` toggle callback 加 `[BUG3-DIAG]` 运行时打印

**新根因（已确认）**：训练时**动态车辆的 Gaussians 被分到了 `background` 层**，不是 `dynamic_rigids` 层。dynamic_mask 训练监督不够强 / 初始化分配不充分，导致车的密度高斯进了 background。

**新 ckpt（全时长 30k 重训）状态**：
- 诊断显示 `dynamic_rigids` 仍有 200k 粒子，70 tracks 都有 frame_info coverage
- 但视觉验证：勾掉 dynamic_rigids 车依然在 → 车的 Gaussian 仍然主要在 background

**待修方向**：
- 训练侧：strengthen dynamic_mask 监督（更高权重）/ adjust dynamic_rigid 初始化策略
- 评估工具：写诊断脚本统计 background 层粒子在 cuboid 内的占比（验证假设）
- 可能跟 V3-P1 ExposureModel / bilateral grid 优化合并研究

**用户实测**：⚠️ 视觉无变化。

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

### ⚠️ B5 — inject CLI 不该是新 ckpt 必需步骤（新 bug，P2）

**现象**：理论上 `trainer.save_checkpoint` 在 `viz_4d.enabled=true` 时自动调 `extract_4d_metadata`，写完整 schema_v2 + FTheta。但本次训练后 ckpt 里 `FTheta present: False`，必须额外跑 `python -m threedgrut.viz.inject` 才能补 FTheta。

**根因**：是 B9 同源（uint64 异常）的连锁反应——`extract_4d_metadata` 在训练 save 路径下静默落到 `ftheta_dict=None`，schema_v2 占位但 FTheta 8-key 没填。

**已做修复（部分）**：commit `b1752b3` 修了 numpy.uint64 → int64 cast。但新训练**没用这版代码**，所以本次 ckpt 仍要 inject。

**待验证**：
- 下次训练应直接产出含 FTheta 的 ckpt，不需要 inject
- 加一个单测：模拟 NCore FTheta camera_model → extract → 验证 ftheta_dict 8 keys 全齐

**优先级 P2**：现 ckpt 用 inject workaround 可用，下次训练验证 b1752b3 修复链是否打通。

---

### ⚠️ B6 — viz_4d ego trajectory 按 camera 拼接，非时间连续（新 bug，P1）

**现象**：用户实测 Play 时**视角在 0-5s 是前视，5-10s 切到后视，10-15s 切其它相机，15-20s 又回到后视**。trajectory 不连续，完全不是真实 ego 飘动。

**根因（已确认）**：[`threedgrut/viz/metadata.py:_extract_ego` L156-179](threedgrut/viz/metadata.py)
```python
for camera_id in dataset.camera_ids:
    frame_indices = dataset.camera_train_frame_indices[camera_id]
    cam_ts = ...frame_indices, end_idx
    all_ts.append(cam_ts)
ts_np = np.concatenate(all_ts)  # ← 按 camera 顺序拼接，不按时间排序
```
同样 `dataset.get_poses()` 也按 camera_ids 循环 → 拼出来的 `ego_poses_c2w` 跟 `frame_timestamps_us` 是 [front_wide_524帧][rear_tele_524帧][cross_left_524帧][cross_right_524帧][rear_left_524帧]，**而不是按时间排序的真 ego trajectory**。

**新 ckpt 诊断证据**：`n_ego_frames: 2623 = 524 + 525 + 525 + 525 + 524`（5 camera 各全帧）。但 ego 在物理上同一时刻只有 **一个** pose，本应 dedupe → ~525 unique time stamps。

**修法**：
- `_extract_ego` 改成只取 primary camera 的帧（或者按 timestamp dedupe + sort）
- 单测：mock 5 camera × 524 帧的 dataset，验证输出 `ego_poses_c2w.shape[0] == 524`（不是 2623）
- 文件：[`threedgrut/viz/metadata.py:143-191`](threedgrut/viz/metadata.py)

**用户实测**：⚠️ 视角切换体验非常突兀。

---

### ⚠️ B7 — Active cuboids checkbox 取消后 Play 仍 render（新 bug，P2）

**现象**：勾掉 Visibility "Active cuboids" checkbox 后，**Play 一推进帧，cuboid 又重新出现**在画面里。

**根因（已确认）**：[`threedgrut_playground/viser_gui_4d.py:_update_active_cuboids` L686-704](threedgrut_playground/viser_gui_4d.py)
```python
def _update_active_cuboids(self, frame_idx):
    pts, cols = self._build_cuboid_edges(frame_idx)
    if self.h_cuboid_lines is not None:
        self.h_cuboid_lines.remove()   # ← 删旧
        self.h_cuboid_lines = None
    if pts.shape[0] == 0:
        return
    self.h_cuboid_lines = self.server.scene.add_line_segments(...)
    # ↑ 重新 add 默认 visible=True, 没读 self.show_cuboids.value
```
对比 `_update_dynamic_lidar` L546-570 是有 `prev_visible = self.h_dyn_pts.visible` 保留 + re-apply 的，**cuboid 路径漏写了这个 preserve 逻辑**。

**修法**：在 `_update_active_cuboids` L692 / L700 加：
```python
prev_visible = (self.h_cuboid_lines.visible 
                if self.h_cuboid_lines is not None 
                else bool(getattr(self, 'show_cuboids', None) and self.show_cuboids.value))
... # remove + add ...
self.h_cuboid_lines.visible = prev_visible
```
镜像 `_update_dynamic_lidar` 的 L556-570 模式，1 行改动即可。

**优先级 P2**：很小很明显，跟 B6 同一个 PR 即可。

---

### ⚠️ B8 — Dynamic LiDAR 初始 checkbox 不勾但点云已显示（新 bug，P2）

**用户实测（2026-05-24 已澄清，含截图）**：
- 初始状态 Visibility folder 里 "Dynamic LiDAR" checkbox **未勾选**（默认 False）
- 但浏览器画面上 **LiDAR 点云已经在显示**（路面上密集白色方块）
- 这是个初始状态不一致的 bug，不是 toggle 失效

**根因（已确认）**：[`viser_gui_4d.py:_update_dynamic_lidar` L546-570](threedgrut_playground/viser_gui_4d.py)
```python
prev_visible = (self.h_dyn_pts.visible
                if self.h_dyn_pts is not None
                else True)                # ← 第一次调用时 h_dyn_pts 为 None,
                                          #    硬编码 True 忽视了 show_dyn_pts.value
```
而 `_on_time_change(t_us_first, source="init")` 在 `__init__` 末尾被显式调用，会立刻调 `_update_dynamic_lidar` → 第一次进来 `h_dyn_pts is None` → `prev_visible=True` → 点云被加进 scene 且 visible=True，跟 checkbox 默认 False 不同步。

**修法**：把 `prev_visible` 默认值从 `True` 改成 checkbox 实际值
```python
prev_visible = (self.h_dyn_pts.visible
                if self.h_dyn_pts is not None
                else bool(getattr(self, 'show_dyn_pts', None) and self.show_dyn_pts.value))
```

**优先级 P2**：影响初始体验，跟 B7 同一类（"per-frame re-add 不保留 visibility flag"）。**B7 和 B8 应同一 PR 修**。

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
1. **下一个 PR**：B6 (ego trajectory dedupe) + B7 (cuboid visibility preserve) + B8 (待澄清, 同源 fix) — 都改 metadata.py / viser_gui_4d.py，一致主题
2. **再下一个 PR**：B2 (FTheta cuboid overlay) — 工作量较大（~0.5d），独立 PR
3. **持续**：B3 训练侧研究（与 V3-P1 合并）
4. **验证**：B5 在下次训练（不需要 inject）

## 验证步骤模板

ThinkPad 浏览器验证：
1. SSH tunnel：`ssh -f -N -L 8090:localhost:8090 thinkpad`
2. 浏览器 http://localhost:8090
3. 查终端 `/tmp/viser_4d.log` 看 `[T8.13-DIAG]` 四段诊断 + `[BUG3-DIAG]` callback 触发记录
4. 截图记录到 `docs/T8_artifacts/<bug_id>_<state>.png`
