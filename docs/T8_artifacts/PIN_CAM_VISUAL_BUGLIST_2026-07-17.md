# PIN-CAM visual buglist — 2026-07-17

## 审计范围

- Checkpoint: `pin_cam_visual_fullfix_frontwide_20s_30k/.../ours_30000/ckpt_30000.pt`
- Viewer: `viser_gui_4d`, `3dgrt`, Follow Camera
- 主审计帧: frame 118, time 12,626,792 us
- 相机: front-wide、cross-left/right、left/right-wide、back-rear-wide、rear-left/right
- Layer 隔离: all-on、sky-only、sky-off、background-only、road-only、background+road
- Novel-view 横移: front-wide 世界坐标 Y 从 4.3 m 改到 5.3 m / 6.3 m（+1 m / +2 m）

浏览器截图只用于目测定位，不用于像素级 KPI。精确 parity 仍需 UI-free dump 与原生 render 对齐。

## 总结

当前 checkpoint **不是多相机 checkpoint**。远端 run 的 `parsed.yaml` 明确只有：

```yaml
dataset:
  camera_ids:
  - camera_front_wide_120fov
```

`metrics.json.per_camera` 也只有 `camera_front_wide_120fov`（24 帧）。因此“前向正常、侧向和后向严重退化”的一级原因是：侧后向视图距离唯一训练相机约 90°–180°，属于无监督外推。它不是本轮 inverse-ray 修复的回归，也不能用来代表已训练侧后相机的质量。

不过本次审计同时确认了 4 个需要在多相机训练前/后处理的真实问题，以及 1 个 viewer 防误判问题。

## Bug list

### PIN-VIS-001 — 当前 checkpoint 缺少侧向/后向训练覆盖

- 严重度: **P0 / 多相机验收阻塞**
- 状态: **5s/5k multi-camera mechanism smoke passed；20s/30k visual acceptance pending**
- 归属: experiment configuration / training coverage
- 现象:
  - front-wide 可用；cross-left/right 明显变软并有 floaters。
  - left/right-wide 基本不可用，画面被大尺度模糊 road/ground blob 占据。
  - back-rear-wide、rear-left/right 出现重复树木、地面拉丝、半透明漂浮物和大面积空洞。
- 硬证据:
  - `parsed.yaml` 只有 `camera_front_wide_120fov`。
  - `metrics.json.per_camera` 只有 front-wide，没有任何侧向/后向 KPI。
- 判断:
  - 这不是“多相机训练后侧后向质量差”，而是“单前向模型被 viewer 放到未训练方向观察”。
- 修复进展（2026-07-18，commit `0994f21`）:
  - 使用标准 inceptio 9-camera 配方完成 5 秒数据窗 / 5k iteration smoke，depth-off、`num_workers=10`、front-tele weight 2.0。
  - 训练完成 `5000 steps / 394.30 s / 12.68 it/s`；checkpoint 明确打包 9 cameras，内置 Test Set 和独立 `render.py` 均成功写出 `metrics.json`。
  - `mean_psnr_masked=23.155`、`mean_cc_psnr_masked=20.326`、`road_crop_psnr=26.197`、automobile class PSNR `18.765`。这些数字只用于确认机制链路，不能与 20 秒 / 30k 锚点横比。
  - per-camera `cc_psnr_masked`: front-wide `22.57` / cross-left `18.73` / cross-right `17.57` / left-wide `15.67` / right-wide `16.92` / back-rear-wide `20.46` / rear-left `20.33` / front-standard `24.90` / front-tele `24.82`。
  - 结论：侧/后相机已真实进入训练、checkpoint 和 eval，不再是单前向 checkpoint 的 90°–180° 无监督外推；但 5 秒/5k 不构成最终视觉质量验收。
- 建议:
  1. 在解决 PIN-VIS-003 的 side-wide validity domain 后，再启动约定的 9-camera（或明确的环视 camera set）训练。
  2. 输出必须包含每相机 PSNR/SSIM/LPIPS；不能继续只看 overall/front-wide。
- 验收:
  - `metrics.json.per_camera` 覆盖全部训练相机。
  - 同帧原生相机位姿检查中，侧后向不再出现整屏级空洞、blob 或重复场景结构。

### PIN-VIS-002 — sky_envmap 在未监督方向产生大幅 bias

- 严重度: **P1**
- 状态: **Compositor + gradient fix landed；5s/5k retrain/eval passed；20s/30k acceptance pending**
- 归属: sky representation / supervision coverage / compositing
- 现象:
  - back-rear-wide all-on 时，天空有成片灰紫色横条、重复亮斑和物体状浮层。
  - 关闭 sky 后，横条消失并暴露黑色空洞；说明 bias 由 sky 层注入。
  - sky-only 的 rear 输出不是连贯天空，而是灰色底、水平纹理带、道路/物体状孤岛。
  - front-wide sky-only 在非天空方向也输出山体、地面和高亮 blob；这些区域在正常视角依靠 Gaussian opacity 遮住，但在稀疏/未覆盖方向会通过空洞泄漏。
- 机制证据:
  - `compute_sky_loss()` 只在 `sky_mask` 像素监督 `rgb_sky`。
  - `_blend_sky()` 会对每条 ray 查询 sky MLP，并在所有 `1 - alpha` 区域合成。
  - 单前向相机只约束了很小一部分球面方向；rear/side direction 的 MLP 输出没有训练约束。
- 根因补充（9-camera 5s/5k，2026-07-18）:
  - 在 checkpoint 已真实训练 9 台相机后，left-wide 同一帧的 sky-only 仍把黄色道路导流线画成整块 envmap，排除“只是单相机 angular coverage 不足”。
  - `_blend_sky()` 缺少世界坐标地平线门控，导致 world-Z 向下的 ray 也查询/合成 sky；同时主 RGB/SSIM/L2 通过 `pred_rgb` 在非天空像素继续反传到 sky MLP，把道路与建筑颜色写进 envmap。
- 修复（commit `39de34f`）:
  - 合成前将 ray 转到 world frame，以 normalized world-Z 构造 sky visibility；`z<=0` 严格为 0，地平线上方约 1.15° 使用短 ramp 避免硬缝。
  - `outputs["rgb_sky"]` 保留原始可微输出供专用 sky loss 使用；主图合成在 `sky_mask=0` 区域改用 detached sky，使非天空 RGB/SSIM/L2 只能推动 Gaussian/opacity，不能再污染 sky MLP。
  - 同一 left-wide / frame 94 / sky-only 前后目检：修复前为灰底 + 大块黄色道路；修复后地平线以下全黑，仅图像顶缘保留真实向上 sky ray 的小片 envmap。打开 Gaussian layers 后道路仍由 road/background 正常呈现，证明没有把实体层误删。
  - 回归测试覆盖 downward ray、camera-to-world rotation、non-sky gradient isolation；Mac `56 passed, 1 skipped`，inceptio CUDA `57 passed`。
  - 9-camera 5s/5k 真实 train→内置 Test Set→独立 `render.py` 全链路通过：`5000 steps / 408.93 s / 12.23 it/s`；外部 eval `mean_psnr_masked=22.991`、`mean_cc_psnr_masked=20.092`、`mean_cc_ssim_masked=0.658`、`mean_cc_lpips_masked=0.547`、road-crop PSNR `26.036`、automobile PSNR `18.604`。
  - 对同一 5s/5k inverse-prefix baseline，cc PSNR `-0.233 dB`、road-crop `-0.161 dB`、automobile `-0.160 dB`，SSIM 近乎不变、LPIPS 微升；这是把非物理 sky 补洞变为黑色/由 Gaussian 接管的机制代价，短窗未见灾难回退，但只作为 smoke，不替代 20s/30k 守护线。
- 建议:
  1. 首先用多相机 sky mask 扩大方向覆盖。
  2. 为未监督方向增加 validity/confidence；低置信方向不要直接用任意 MLP 输出填充 Gaussian 空洞。
  3. 增加方向平滑/低频先验，或为无覆盖方向使用稳定 neutral fallback。
  4. 增加 per-camera sky-only 可视化和 sky-region KPI。
- 验收:
  - 9 个训练相机的 sky-only 输出只呈现连续天空外观，无道路、建筑或横向复制纹理。
  - all-on 中关闭/开启 sky 不应改变非天空实体的颜色或引入大面积色偏。

### PIN-VIS-003 — left/right-wide validity certificate 失败并回退 legacy gate

- 严重度: **P1 / 多相机训练前置阻塞**
- 状态: **Fixed + geometry verified；等待多相机训练视觉验收**
- 归属: OpenCVPinhole validity-domain contract
- 日志证据:
  - `camera_left_wide_90fov`: certificate failed，legacy gate retained。
  - `camera_right_wide_90fov`: certificate failed，legacy gate retained。
  - `camera_front_tele_30fov`: 同样失败。
  - cross-left/right、back-rear-wide、rear-left/right 和 front-wide 均得到 `max_valid_r2`。
- 风险:
  - side-wide 若直接加入下一次训练，可能再次使用旧 `0.8 < icD < 1.2` 有效域，与已经修复的 front-wide 行为不一致。
  - 当前 left/right-wide 的极差画面主要由“未训练视角”解释，不能据此把全部退化归因于 legacy gate。
- 修复（2026-07-18）:
  - validity certificate 改为从光轴开始的最大安全连续 radial prefix；遇到远端 rational pole / fold 时，在第一个精确安全边界前裁剪，而不是整台相机回退 legacy gate。
  - 训练监督使用同一个 `max_valid_r2` 屏蔽证书外 inverse rays，保证 inverse-ray loss domain 与 renderer forward gate 一致。
  - 若未来某台相机仍无法生成证书，训练侧会显式应用与 renderer legacy gate 对应的 forward-valid mask，不再静默产生监督域不一致。
- 实测证据（inceptio b6a9，1920×1080）:
  - `camera_front_tele_30fov`: `max_valid_r2=0.094846842`，certified coverage `99.9970%`，训练 mask 额外移除 63 px。
  - `camera_left_wide_90fov`: `max_valid_r2=4.078572345`，certified coverage `98.9068%`（legacy `64.5587%`），训练 mask 在 ego mask 后额外移除 7,815 px。
  - `camera_right_wide_90fov`: `max_valid_r2=3.640740797`，certified coverage `98.1092%`（legacy `64.3357%`），训练 mask 在 ego mask 后额外移除 12,933 px。
  - 修复版 viewer 在 8090 启动时 9 台 OpenCVPinhole 相机全部输出显式 `max_valid_r2`，无 certificate fallback；聚焦测试本地 116/116、inceptio 47/47 通过。
  - 后续 9-camera 5s/5k smoke 的 train、内置 Test Set、独立 `render.py` 三条真实数据路径均无 certificate fallback；两个 metrics 出口全部覆盖 9 台训练相机。
- 建议:
  1. 对这三台相机单独检查 rational denominator、真实图像角点半径和 pole 位置。
  2. 明确采用可证明安全的裁剪半径或 per-calibration validity mask；不要静默回退。
  3. 在训练前记录每相机有效像素覆盖率和 center-row/column span。
- 验收:
  - 所有参加训练的 OpenCVPinhole 相机都有显式 validity domain。
  - 日志中无静默 legacy fallback；若必须裁剪，需输出覆盖率并由视觉验收接受。

### PIN-VIS-004 — background 层包含 road/lane 内容，横移后与 road 层重叠

- 严重度: **P2 / minor**
- 状态: **A1+A2 candidate mechanism passed, but 5s KPI/perf gate failed；default unchanged**
- 归属: semantic layer separation / initialization / regularization
- 现象:
  - background-only 在 nominal front-wide 中仍清楚包含双黄线、白色车道线和部分路面 radiance。
  - Y +1 m / +2 m 后，这些“属于 background 的地面内容”仍作为独立几何出现。
  - +2 m 时打开 road，background 与 road 同时贡献车道线和路面结构，产生宽化、拉丝和重影；road-only 自身也显示明显的 off-track 拉伸。
- 配置证据:
  - `trainer.bg_road_penalty.enabled: true`，但只使用软 penalty。
  - `bg_road_slab_exclude.enabled: false`，没有启用硬的 road slab 排除。
- 现有机制证据:
  - 仓库已有 A1+A2 hard clamp：A1 按 road BEV 薄板钳制 background density，A2 将 background 中心投影到当前训练相机并按 `road_mask` 钳制错误深度的 road floaters。
  - 历史 3k 单变量 A/B 已取得 lane grad-corr `+0.13`、lane-band PSNR `+3.6 dB`、road-crop PSNR `+2.0 dB`，且 OpenCVPinhole smoke 已验证；当前标准 9-camera 配方仍将两个开关保持 OFF，故本次重影不是“无实现”，而是修复未进入该 checkpoint 配方。
  - 下一步必须在当前 9-camera / inverse-ray / sky-fix 基线上打开 A1+A2 做 5s 单变量复验，再决定是否进入 20s/30k；不能仅凭旧 clip 数字直接改默认值。
- 当前 9-camera 5s/5k 单变量复验（完成）:
  - call #1: A1 clamp `433,176`、A2 road-mask projection clamp `162,053`；call #500 为 `352,777 / 76,256`；call #1000 已降到 `714 / 696`，证明两个 gate 都真实接管了错误 background，而不是配置空跑。
  - call #5000 稳定为 A1 `63` / A2 `129`，没有重新爆发；checkpoint、内置 Test Set、独立 `render.py` 均成功。
  - 但训练从 `408.93 s / 12.23 it/s` 退到 `1479.61 s / 3.38 it/s`，耗时 `3.62x`；NumPy/CPU A2 在 100 万 background、9-camera 上的实际代价远高于历史“约 25% slowdown”。
  - 对 sky-fix baseline：`mean_cc_psnr_masked -0.208 dB`、road-crop PSNR `-0.356 dB`、road-crop LPIPS `+0.0035`（变差）、mean class PSNR `-0.020 dB`；cross-left/back-rear 微升，其余多数相机下降，front-tele `-0.706 dB`。
  - viewer 同帧 layer isolation 仍显示：front-wide frame 36 的 background-only 在 nominal 与 Y+2m 都覆盖大面积灰色 road surface；A1+A2 去掉了部分高对比 road ownership，但基于 Gaussian **中心**的 clamp 没有阻止 road 外部/大 footprint splat 覆盖 road pixel，视觉验收同样未通过。
  - 结论：A1+A2 只完成了部分 takeover 清理，当前 9-camera 配方同时未通过图层纯度、road KPI 与性能门，**不得直接改默认 YAML**。下一版需要 footprint-aware/pixel-aware ownership（不只检查中心），并做 active-only/GPU 投影降本、补足 road 层接管后的容量，再重跑 5s→20s/30k。
- 建议:
  1. 在 background 初始化/增密阶段排除 road semantic pixels 与 road XY/Z slab。
  2. 评估启用 `bg_road_slab_exclude` 和 projection-aware exclusion，而不是只依赖 `lambda: 0.1` 的软 penalty。
  3. 加入 background-road opacity/occupancy overlap 指标。
- 验收:
  - background-only 不再出现连续车道线和大面积路面纹理。
  - front-wide 横移 ±1 m / ±2 m 时，background+road 不出现双线、双边缘或重复地面物体。

### PIN-VIS-005 — viewer 暴露未训练相机但没有 trained/untrained 提示

- 严重度: **P2 / 诊断误导**
- 状态: **Fixed by checkpoint camera-contract filtering**
- 归属: viewer UX / checkpoint metadata
- 现象:
  - camera dropdown 从 manifest 暴露全部 9 台相机。
  - Camera status 只显示模型、分辨率和 pose，不说明当前 checkpoint 只由 front-wide 训练。
  - 这会把预期的超大角度外推误判为相机渲染回归。
- 建议:
  - checkpoint/viz metadata 保存实际 `dataset.camera_ids`。
  - dropdown 对未训练相机标记 `UNTRAINED / extrapolation`，Camera status 给出显式 warning。
- 修复证据（9-camera smoke）:
  - manifest 可发现 12 台相机，但 viewer 日志明确输出 `bound 9 active checkpoint camera contracts` 与 `9 cameras available for dropdown`。
  - dropdown 只暴露 checkpoint 内 9 台训练相机；front/back fisheye 与 held-out rear-right 不再伪装成已训练视角。
- 验收:
  - 选择不在 checkpoint 训练集合中的相机时，UI 必须显示醒目 warning。

## Investigation note

- frame 118 的 `camera_cross_right_120fov` 还出现 `nearest delta-t=166.1 ms`、`source gap=400.0 ms` warning。它可能影响动态物体的时序对齐，但无法解释当前整幅静态场景退化；应在真正的 multi-camera checkpoint 上单独验证，暂不升级为独立 confirmed bug。

## 推荐处理顺序

1. **先处理 PIN-VIS-003**：确保要加入训练的 side-wide 相机不再走 legacy validity gate。
2. **跑真正的 multi-camera baseline**：解决 PIN-VIS-001，并产生完整 per-camera KPI。
3. **在 multi-camera baseline 上复测 sky**：判断 PIN-VIS-002 剩余多少；必要时再改 sky validity/regularization。
4. **最后处理 layer purity**：针对 PIN-VIS-004 做 background/road 单变量 A/B。
5. **补 viewer warning**：PIN-VIS-005 成本低，可与上述任务并行完成。
