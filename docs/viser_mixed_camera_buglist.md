# viser_gui_4d Mixed-Camera Bug List

**最后更新：** 2026-07-14 13:14 CST

**发现环境：** inceptio RTX 4090，C4 11-camera checkpoint，`3dgrt` renderer

**主入口：** `threedgrut_playground/viser_gui_4d.py`
**对应实施计划：** [`docs/superpowers/plans/2026-07-14-viser-mixed-camera-bugfix.md`](superpowers/plans/2026-07-14-viser-mixed-camera-bugfix.md)

---

## 1. 背景与结论边界

C4 将 9-camera 配方扩为 11-camera，新增：

- `camera_front_fisheye` — NCore `FThetaCameraModel`
- `camera_back_rear_fisheye` — NCore `FThetaCameraModel`

其余 10 台可用相机均为带 rational distortion 的 `OpenCVPinholeCameraModel`，不是无畸变 ideal pinhole。

用户在 C4 viser 检查中发现：

1. `render.py` 鱼眼图具有明显径向畸变，而 viser 中画面看起来被平面化；
2. Follow Camera 播放时画面逐帧跳动；
3. 多台相机呈现“中心较清晰、外围显著模糊”；
4. cuboids / trajectories 在部分相机中完全错位，`camera_front_wide_120fov` 下相对温和。

经代码与数据检查，确认当前 viewer 存在 mixed-camera projection/state 缺陷。因此：

> 在本 buglist 关闭前，mixed-camera viser 不能作为 checkpoint 鱼眼投影、外围清晰度或 overlay 对齐的可靠验收工具。C4 模型质量只能先参考 `render.py` 原生相机输出与 metrics；不得用当前 viser 截图单独决定 C4 晋级或放弃。

---

## 2. 运行证据

### 2.1 Camera model 实测

| Camera | Model |
|---|---|
| `camera_front_fisheye` | FTheta |
| `camera_back_rear_fisheye` | FTheta |
| 其余 10 台 | OpenCVPinhole rational distortion |

### 2.2 Camera pose 连续性实测

原始 `c2w` rotation 连续，没有 calibration pose 爆跳：

| Camera | median Δrotation | max Δrotation | median Δtranslation | max Δtranslation |
|---|---:|---:|---:|---:|
| front wide | 0.122° | 0.392° | 1.15 m | 3.56 m |
| front fisheye | 0.119° | 0.529° | 1.15 m | 5.90 m |
| rear fisheye | 0.119° | 0.404° | 1.17 m | 6.64 m |
| cross left | 0.119° | 0.391° | 1.15 m | 3.67 m |

相机流正常约 10 Hz，但存在 200–600 ms 缺帧。当前 Follow Camera 取 nearest pose，无 interpolation，因此高速行驶时普通帧约跳 1.15 m，缺帧位置可跳 5–7 m。

### 2.3 启动日志矛盾

C4 viewer 启动时同时出现：

- metadata primary camera 无 FTheta，先打印“无 FTheta intrinsics，走 pinhole approximation”；
- `_load_multi_cam_poses` 随后正确识别两台 `ftheta=YES`；
- `--initial_cam_id camera_front_fisheye` 又打印 engine FTheta 已锁定；
- 浏览器 Camera dropdown 却显示 `camera_front_wide_120fov`。

这证明 GUI camera id、client pose 与 engine projection state 没有单一真源。

---

## 3. Bug 状态总览

| ID | 名称 | 当前状态 | 优先级 | 当前结论 |
|---|---|---|---|---|
| MC-1 | initial camera 被 GUI primary camera 覆盖 | ✅ 已修复 | P0 | CLI initial camera、dropdown、pose 与 projection state 已统一；有单测和 C4 启动验收 |
| MC-2 | camera switch 未原子更新 projection state | ✅ 已修复 | P0 | `CameraRenderState` + 单一 apply 路径；mixed switch matrix 通过 |
| MC-3 | FTheta → pinhole 不清 stale FTheta state | ✅ 已修复 | P0 | 切到 OpenCVPinhole 时显式清空 FTheta fields；transition test 通过 |
| MC-4 | OpenCVPinhole intrinsics/rays 不随 dropdown 更新 | ✅ 已修复 | P0 | intrinsics、per-pixel rays 与 native resolution 随 camera 原子切换 |
| MC-5 | overlay compositor 不能从 `None` 动态创建 | ✅ 已修复 | P0 | OpenCVPinhole → FTheta 可动态创建 compositor；world-space trajectory cache 与初始 model 解耦 |
| MC-6 | 非 FTheta distorted pinhole overlay 投影公式 | 🔴 重新打开 | P1 | `PinholeForwardProjector` 已证实把 6 项 OpenCV rational radial coefficients 错当普通 polynomial，且忽略 thin-prism / SDK validity gate；wide 外围可数值发散，需按 NCore SDK parity 重写 |
| MC-7 | ego frustum 绑定 `_initial_cam_id` 而非当前 dropdown camera | ✅ 已修复 | P1 | frustum 重新解析当前 selected camera state；单测通过 |
| MC-8 | Follow Camera nearest-frame 阶梯跳动 | 🟡 核心问题已修，数据缺帧仍存在 | P1 | nearest snap 已改为 translation lerp + rotation SLERP；200–600 ms source gap 无法凭空恢复，只显示 warning，仍需人工评估大 gap 观感 |
| MC-9 | Camera dropdown / engine state 缺少可见诊断 | ✅ 已修复 | P1 | Camera status 显示 camera/model/resolution/pose interpolation/Δt/overlay/warning |
| MC-10 | “中清边糊”/外围眩晕 | 🟡 高置信根因已定位，待 A/B | P1 | OpenCV rational wide camera 约 35–37% 外围 pixels 为 finite inverse rays、但 forward projection `valid=False`；dataset 未屏蔽这些 supervision pixels。PAI/FTheta 以成对 polynomial + `max_angle` 共用有效域，无同类症状。详见 [`inceptio_opencv_rational_peripheral_blur_analysis_2026-07-14.md`](inceptio_opencv_rational_peripheral_blur_analysis_2026-07-14.md) |
| MC-11 | mixed-camera 自动回归覆盖缺失 | ✅ 已修复 | P0 | 新增 mixed transition、resolution、overlay lifecycle、frustum、interpolation 回归测试 |

> 本表记录 `0159698..66f668a` mixed-camera + rig-origin 修复及 C4 实机验收证据。若人工操作可稳定复现与表中“已修复”相冲突的现象，对应条目应立即重新打开，而不是用现有单测结论否定人工观察。

---

## 4. 详细 Bug

### MC-1 — initial camera 被 GUI primary camera 覆盖

**现象**

启动参数指定：

```text
--initial_cam_id camera_front_fisheye
```

日志显示 engine 已切到 front fisheye，但浏览器 dropdown 显示 `camera_front_wide_120fov`。

**根因**

`Viser4DViewer.__init__` 早期调用：

```python
self._snap_clients_to_camera(self._initial_cam_id, self._t_us_current)
self._current_dropdown_cam = self._initial_cam_id
```

随后 `_build_visibility_gui()` 又按 metadata primary camera 初始化：

```python
initial_cam = self.meta.ego_primary_camera_id
self._current_dropdown_cam = initial_cam
```

后者覆盖前者。

**影响**

- dropdown 文本不代表实际 projection state；
- Follow Camera 开启后会突然切回 primary camera；
- 用户误以为在看 fisheye，实际状态可能混合。

**验收**

启动指定任意合法 camera id 后，以下四者必须一致：

1. dropdown value；
2. `_current_dropdown_cam`；
3. client pose 的 source camera；
4. active projection state's camera id。

---

### MC-2 — camera switch 未原子更新 projection state

**现象**

切换相机后，画面、FOV、resolution、overlay 和 dropdown 可能在不同 callback 中更新，出现短暂或持续的混合状态。

**根因**

当前 `_snap_clients_to_camera()` 同时承担：

- timestamp lookup；
- camera pose snap；
- FTheta state 部分更新；
- compositor 部分重建；
- FOV 更新；
- render dirty flag。

OpenCVPinhole state、旧 state 清理、GUI state 则分散在其他位置或完全缺失，没有一个完整的 `CameraRenderState`。

**影响**

任何 mixed-camera 顺序都可能触发状态残留，例如：

```text
front_wide → front_fisheye → front_tele → rear_fisheye
```

**验收**

增加纯函数 resolver 与单一 apply 方法；每次切换后 active state 的 camera id、model type、intrinsics、rays、resolution、FOV、overlay projector 必须来自同一个 camera entry。

---

### MC-3 — FTheta → pinhole 未清理 stale state

**根因代码**

`_snap_clients_to_camera()` 只有：

```python
if new_ftheta is not None:
    self.ftheta_intrinsics = new_ftheta
```

当新相机不是 FTheta 时，没有显式清空：

```python
self.ftheta_intrinsics = None
self.ftheta_render_wh = None
```

**影响**

选中 OpenCVPinhole 相机后，`fast_render()` 仍可能传入旧 `fisheye_intrinsics`。engine 分支优先判断 FTheta，导致当前 pinhole pose 配旧 FTheta projection。

**验收 sequence**

```text
FTheta A → OpenCVPinhole B
```

切换后：

- `ftheta_intrinsics is None`；
- engine 收到 `fisheye_intrinsics=None`；
- render resolution 与 B 一致；
- active overlay projector 不是 FTheta。

---

### MC-4 — OpenCVPinhole intrinsics/rays 不随 dropdown 更新

**根因**

`opencv_pinhole_intrinsics` 和 `opencv_pinhole_rays` 只在 viewer construction 时、且仅针对 `initial_cam_id` 初始化。`_snap_clients_to_camera()` 完全不更新二者。

**影响**

所有非初始 OpenCVPinhole 相机可能：

- 使用旧相机 rational distortion 参数；
- 使用旧相机 per-pixel rays；
- 使用当前相机 pose；
- 使用近似 scalar FOV 控制浏览器相机。

此外，`opencv_pinhole_rays` 是按该相机原始 `W×H` 预计算的，但
`update()` 目前只在 FTheta 模式锁定训练分辨率；OpenCVPinhole 仍读取通用
resolution slider。若 slider 与 rays shape 不一致，会造成 shape mismatch，或迫使后续
代码在错误分辨率下解释固定 rays。calibrated OpenCVPinhole 必须与 FTheta 一样锁定
到 state resolution，不能继续使用自由缩放的 ideal-pinhole 路径。

angular mismatch 在光轴附近最小、外围最大，符合“中间清晰、外围模糊”的症状。

**验收 sequence**

```text
front_wide → front_standard → front_tele → left_wide
```

每一步 engine 收到的 OpenCVPinhole dict/rays object identity、resolution 和 camera id 均必须切换；不得沿用上一步数组。

`update()` 的 render `W×H` 必须等于当前 OpenCVPinhole rays 的 `W×H`；resolution
slider 在 FTheta 和 OpenCVPinhole 两种 calibrated 模式均不可改变实际 render shape。

---

### MC-5 — FTheta overlay compositor 不能动态创建

**根因代码**

compositor 仅在 constructor 的初始 FTheta 模式创建。之后切相机时：

```python
if self._overlay_compositor is not None:
    self._overlay_compositor = Viser4DOverlayCompositor(...)
```

如果初始 primary camera 非 FTheta，则 compositor 初始为 `None`；后来切到 fisheye 仍不会创建。

同一 constructor 条件还控制 `_overlay_static_ego_polylines` /
`_overlay_static_track_polylines` 的填充。即使只修 compositor 从 `None` 创建，若静态
cache 仍按初始 camera model 条件构建，后切入 FTheta/OpenCVPinhole image-space overlay
也会缺失 trajectories。因此 world-space overlay cache 必须与初始 projection model
解耦，只要 metadata 存在就无条件建立。

**影响**

- Gaussian backdrop 可走 FTheta；
- cuboid / trajectory 却继续走 browser-side pinhole 3D primitive；
- 光轴附近看起来“温和”；
- 外围及 fisheye 下线框完全脱离物体。

**验收 sequence**

```text
OpenCVPinhole → FTheta
```

切换后必须创建 FTheta projector compositor，并移除/隐藏旧 browser line segments 与 3D labels。

---

### MC-6 — OpenCVPinhole distorted overlay 仍走 ideal pinhole

**现象**

即使不使用 FTheta，多台 wide OpenCVPinhole 相机的 cuboids / trajectories 在外围也对不上，front wide 中央区域相对温和。

**根因**

当前 image-space overlay 只支持 `FthetaForwardProjector`。OpenCVPinhole camera 仍使用 viser browser 的 ideal pinhole projection，忽略 NCore rational radial/tangential/thin-prism distortion。

仓库已有 `PinholeForwardProjector`，但没有接入 viewer compositor。

**影响**

- distorted wide camera 外围 overlay 漂移；
- narrow front standard/tele 较温和；
- 同一 cuboid 在不同相机中一致性差。

**验收**

所有带 NCore calibration 的相机都走 image-space overlay：

- FTheta → `FthetaForwardProjector`；
- OpenCVPinhole → `PinholeForwardProjector`；
- 只有无 calibration 的 legacy camera 才保留 browser line segments fallback。

---

### MC-7 — ego frustum 使用旧 initial camera

**根因**

`_update_ego_frustum()` 优先读取 `_initial_cam_id`，而不是 `_current_dropdown_cam`。

**影响**

dropdown 切到侧向或后向相机后，frustum 仍可能代表初始 front camera，形成明显辅助信息错位。

**验收**

相机切换后 frustum pose 与当前 active camera state 使用同一个 c2w；未选择相机时才 fallback 到 metadata ego pose。

---

### MC-8 — Follow Camera 使用 nearest pose，产生阶梯跳动

**根因**

`_snap_clients_to_camera()` 用 nearest timestamp index，直接写离散 `c2w`。播放 timeline 是连续 wallclock timestamp，但 camera pose 没做 SE(3) interpolation。

**影响**

- 普通 10 Hz 帧：约 1.15 m 一跳；
- 200–600 ms 缺帧：约 3–7 m 一跳；
- rotation 虽连续，translation 台阶仍造成画面明显跳跃。

**修复原则**

- translation：linear interpolation；
- rotation：short-arc quaternion SLERP；
- timeline 范围外：clamp endpoint；
- 相邻 pose gap 超阈值时：仍可插值，但 UI/status 明确显示 gap warning；不得悄悄伪装为高频真实 pose。

**验收**

合成两帧 pose 在中点 timestamp 返回 50% translation 与 50% rotation；真实 C4 播放不再 10 Hz 阶梯跳，缺帧区显示 source gap。

---

### MC-9 — 缺少 active projection diagnostics

**现象**

用户只能看到 Camera dropdown 和 Renderer，无法判断实际 engine 使用的是：

- FTheta；
- OpenCVPinhole rational；
- ideal pinhole fallback；
- 哪个 resolution；
- 哪一帧 pose / 时间差；
- overlay projector 是否一致。

**验收 UI**

在 Camera 控件下增加只读状态：

```text
Active camera: camera_front_fisheye
Model: FTheta
Render: 1920×1080
Pose source: frame 84 / Δt=12.4 ms / interpolated
Overlay: FTheta image-space
```

若发生 fallback、缺 calibration 或大 gap，状态必须显示 warning，而不是只写远端 log。

---

### MC-10 — 中心清晰、外围模糊归因不明

**状态：待隔离，不先假定是代码 bug。**

可能来源：

1. checkpoint 在极端视角的真实欠约束；
2. FTheta / rational distortion state 串台；
3. viewer 与 `render.py` 使用不同 per-pixel rays；
4. free-camera pose 与训练相机 pose 不完全一致；
5. Gaussian anisotropy / grazing angle 在外围投影下的真实退化。

**验证方法**

固定同一：

- checkpoint；
- camera id；
- camera frame timestamp；
- c2w；
- resolution；
- camera model parameters。

对比 `render.py` native-camera output 与 viewer headless output，并分别计算：

- center disk：半径 ≤ 0.35 × half-diagonal；
- peripheral annulus：0.65–0.95 × half-diagonal；
- full-frame MAE / PSNR；
- center/periphery gap。

**判定**

- viewer 与 render parity 后仍外围模糊：模型/数据问题；
- render 清晰、viewer 外围差：viewer projection/ray bug；
- 两者都差但 viewer 更差：两类问题叠加。

---

### MC-11 — mixed-camera state transition tests 缺失

现有测试覆盖：

- 单一 FTheta projector；
- 单一 Pinhole projector；
- constructor-time FTheta overlay；
- Follow Ego；
- 单 camera FOV。

但没有覆盖最关键的状态序列：

```text
OpenCVPinhole → FTheta → OpenCVPinhole → FTheta
```

也没有断言：

- stale state 被清理；
- compositor 从 `None` 创建；
- GUI initial camera 不覆盖 CLI；
- OpenCVPinhole rays 按 camera 更新；
- frustum 使用当前 camera；
- interpolation 行为。

这是旧单相机测试全部通过、mixed-camera 仍回归的根本测试盲区。

---

## 5. 修复顺序与 Gate

| Phase | 内容 | Gate |
|---|---|---|
| A | 建立 `CameraRenderState` 单一真源 + mixed transition 单测 | 所有 model/state 切换无 stale field |
| B | 动态 image-space overlay，覆盖 FTheta + OpenCVPinhole | 12-camera raw-image overlay 与 backdrop 同投影 |
| C | Follow Camera SE(3) interpolation + gap diagnostics | 合成单测 + C4 播放无 10 Hz 阶梯跳 |
| D | render/viewer projection parity 工具 | center/periphery 指标可复现、可归因 |
| E | inceptio C4 视觉验收 | 两鱼眼畸变正确；overlay 多相机对齐；无状态串台 |

P0 Gate：Phase A 与 B 未完成前，不进行 C4 viewer 视觉决策。

---

## 6. 最终验收矩阵

至少验证以下 camera：

| Camera | Model | 重点 |
|---|---|---|
| front fisheye | FTheta | 径向投影、外围 cuboid 曲线、切换后 state |
| rear fisheye | FTheta | 强 ego mask、后向 pose、overlay |
| front wide | OpenCVPinhole | rational distortion、当前温和基准 |
| cross left | OpenCVPinhole wide | 外围 overlay、侧向 pose |
| left wide | OpenCVPinhole wide | rational pole / 非有限边界 |
| front standard | OpenCVPinhole narrow | 与 wide 对照 |
| front tele | OpenCVPinhole narrow | FOV/rays 切换、不得继承 wide |

必须运行切换序列：

```text
front_wide
→ front_fisheye
→ front_tele
→ back_rear_fisheye
→ cross_left
→ front_wide
```

每一步检查：camera id、model、resolution、pose source、overlay type、画面投影与 cuboid 对齐。

---

## 7. 非目标

本轮不处理：

- C4 checkpoint 本身是否最终晋级；
- Gaussian 表示侧的尖刺/anisotropy 优化；
- 训练 camera loss weighting；
- rolling-shutter per-row pose 精确模拟；
- viewer free-camera 的任意 fisheye novel-view UI 设计。

这些须在 viewer parity 修复后另行判断，避免把可视化工具缺陷误归因给模型。

---

## 8. 2026-07-14 修复状态

实现提交范围：`0159698..66f668a`（目标分支：`main`）

| Phase | 状态 | 证据 |
|---|---|---|
| A CameraRenderState + 原子切换 | ✅ | mixed transition 测试；FTheta/OpenCV 字段互斥清理 |
| B calibrated image-space overlay | ✅ | FTheta + OpenCVPinhole projector integration；C4 多相机目视 |
| C Follow Camera SE(3) interpolation | ✅ | translation lerp + quaternion SLERP；播放 status 显示插值 alpha |
| D radial parity comparator | 🟡 | 比较器与单测完成；缺 deterministic UI-free viewer PNG dump，未声称真实 parity 数字 |
| E inceptio C4 视觉验收 | ✅ | `docs/T8_artifacts/C4_mixed_camera_viewer_fix_validation.md` |

验证结果：latest focused `55 passed`；Mac full `1008 passed, 2 skipped`；inceptio focused
`39 passed`。C4 切换 fisheye/wide/tele/rear-fisheye/cross-left/wide 无 stale state，
Follow Camera 使用插值 pose；Ego trajectory 改用 pose-graph `rig` origin，C4 首帧 camera
`z=2.4504` 改为 rig `z=0.0029`，局部 road median 差约 −0.13 m。残留的近场块状
splat、floaters、远景模糊归为模型质量
现象；在 exact viewer frame dump 补齐前，不声称 center/periphery native parity 已量化。
