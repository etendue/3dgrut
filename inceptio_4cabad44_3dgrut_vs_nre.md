# Inceptio 4cabad44 — NRE vs 3dgrut Static Reconstruction Comparison

> Date: 2026-06-24
> Operator: Eason.L (大g) + Claude Code
> Data: `/home/inceptio/ncore_data/inc_4cabad44_v2_20s_finalmask/`
> Clip: 6-cam OpenCVPinhole rational distortion, 199 frames / 19.95s, 重卡高速干线场景
> Status: 已知 limitation — 3dgrut multi-cam × OpenCVPinhole rational 训练 fail

## 1. 目标

复现 NRE 在 inceptio 4cabad44 finalmask clip 上的静态场景重建（baseline 28.99 dB），并用 3dgrut 对同一数据训一遍做横向对比。

## 2. 数据

| 项 | 值 |
|---|---|
| Clip ID | `inceptio_4cabad44-6d56-4c2e-999f-8db32983849c` |
| 格式 | NCore V4 separate-sensors profile |
| 时长 | 19.95s (199 frames) |
| 传感器 | 6 cam (1918×1078, OpenCV Pinhole + 6-coef rational distortion) + 1 LiDAR (lidar_top_360fov, 4 物理雷达拼接, 128 rings × 4096 cols) |
| Ego mask | ✅ 已生成（rear_left/right 5.8-5.9% nonzero, 其他相机空） |
| cuboids | ❌ 空 (inceptio converter 当前版本未接入 ppn_fusion) |
| aux 数据 | sseg / depth / egomask / lidar-camvis / lidar-sseg 已生成 (~2 hours on RTX 4090) |
| 相机 distortion 强度 | k1=0.48-1.40, k4=0.53-1.76 (rational mode); front_tele 30° 极端 k3=-138, k6=-153 |

## 3. NRE Baseline (✅ 成功)

| 项 | 值 |
|---|---|
| Config | `apps/prod/Hyperion-8.1/car2sim_6cam` |
| Steps | 40000 |
| Subsample | 2 (训练分辨率 959×539) |
| Duration | ~75 min on RTX 4090 |
| 镜像 | nvcr.io/nvidia/nre/nre-ga:latest |
| **test/psnr (overall)** | **28.99 dB** |

### Per-class cpsnr (NRE)

| Class | cpsnr (dB) | Class | cpsnr (dB) |
|---|---|---|---|
| sky | 44.47 | building | 35.31 |
| road | 40.42 | wall | 34.34 |
| egocar | 38.92 | sidewalk | 33.98 |
| vegetation | 36.44 | pole | 33.22 |
| terrain | 36.12 | person | 31.63 |
| fence | 34.63 | traffic sign | 30.19 |
| | | traffic light | 27.84 |
| **car (动态)** | **25.58** | **truck (动态)** | **21.06** |

静态类全部 30-44 dB；动态类（car/truck）显著低（无 cuboids → 鬼影/擦除）。

### NRE 视觉验证

抽 cam00 t=6s 红色重卡帧做 input vs pred 对比：
- INPUT GT：远山 + 护栏 + 车道线 + 右侧红色重卡半挂车
- PRED：远山/护栏/车道线全部精准还原，**红色卡车完全被擦除**（动态噪声压制，符合预期）
- cpsnr/truck=21.06 主要来自这种"擦除"造成的整帧像素错位

## 4. 3dgrut 三轮训练 — 配方与结果

跑了 3 个配方，**前两轮 6cam 都垮，单 cam 救活**：

| 配方 | config | n_cams | n_iter | downsample | mean PSNR (raw) | 视觉 | 时长 |
|---|---|---|---|---|---|---|---|
| **A.** multilayer 4-layer | `ncore_3dgut_mcmc_multilayer` (3 layer: bg/road/sky_envmap, 去 dynamic_rigids) | 6 | 30k | 1.0 | **20.20** | ❌ 整张糊 | 47min |
| **B.** single-layer | `ncore_3dgut_mcmc` (v1 MoG, no layers) | 6 | 30k | 1.0 | **20.99** | ❌ 整张糊 | 36min |
| **C.** single-camera | `ncore_3dgut_mcmc` | **1** (front_wide) | 30k | 1.0 | **28.44** | ✅ 清晰 | 22min |

C 直接逼近 NRE 28.99 dB。

### 5-up 视觉对比（cam0 t≈5s）

| 上行 | INPUT GT | NRE 6cam **28.99** | 3dgrut **single-cam 28.44** |
|---|---|---|---|
| 渲染质量 | — | 远山/护栏/车道线/标志全 sharp | 同样 sharp，几乎平起平坐 |

| 下行 | 3dgrut multi 6cam **20.20** | 3dgrut single 6cam **20.99** | (blank) |
|---|---|---|---|
| 渲染质量 | 整张糊，远山/车道线几乎丢 | 更糊 + 中心放射状条纹 | — |

存档：
- 对比图：`/tmp/inc4cab_vis/4up_cam0_t3s.png` (NRE input vs pred + 3dgrut multi/single 6cam)
- 5up grid：`/tmp/inc4cab_vis/5up.png` (加 single-cam baseline)
- single-cam val mp4 (25 帧 × 5 fps)：`/tmp/inc4cab_vis/sc_pred.mp4`
- canvas dump (viser, single-cam ckpt)：`/tmp/inc4cab_vis/viser_sc_canvas.png`

## 5. 根因诊断：3dgrut multi-cam × OpenCVPinhole rational distortion fail mode

### 实验隔离的事实

| 实验 | n_cams | camera model | PSNR | 备注 |
|---|---|---|---|---|
| NRE 6cam | 6 | OpenCVPinhole rational | **28.99** | ✅ |
| 3dgrut PAI 6cam (CLAUDE.md T4.5 历史) | 6 | **FTheta** | 26.31 | ✅ |
| 3dgrut inceptio 6cam (multilayer + single-layer) | 6 | OpenCVPinhole rational | 20.20 / 20.99 | ❌ |
| 3dgrut inceptio 1cam (front_wide) | **1** | OpenCVPinhole rational | **28.44** | ✅ |

变量交叉：
- single-cam OpenCVPinhole OK → 排除 OpenCVPinhole 数学问题、distortion CUDA clamp
- PAI 6cam FTheta OK → 排除 3dgrut multi-cam pipeline 本身
- world frame normalized 验证（rig 起 [0,0,0]，cam c2w float32, 跨度 ~43m）→ 排除 transform / coordinate frame

唯一 fail 组合 = **6 cam × OpenCVPinhole rational distortion in 3dgrut**。

### 代码层验证（chasing the candidate root cause）

详细 grep + Read 链路（all OK，证明不是这些）：
- `cameraProjections.cuh:72-117`：CUDA `projectPoint(OpenCVPinholeProjectionParameters&)` 完整支持 6-coef rational + tangential + thin_prism distortion
- `bindings.cpp:35, 125`：`fromOpenCVPinholeCameraModelParameters` 工厂函数注册
- `tracer.py:449`：Python 端调 factory 传 `intrinsics_OpenCVPinholeCameraModelParameters` 真字典
- `datasetNcore.py:1742`：NCoreDataset 正确设 batch.intrinsics_OpenCVPinholeCameraModelParameters
- `Batch.intrinsics` 简单 4-tuple 字段：NCore path 不设，避开 tracer.py:435 dummy 全 0 distortion 分支
- inceptio T_world 已 normalized（rig 起点 [0,0,0]）+ 6 cam c2w float32 + cam 位置合理

### 最可能根因（runtime instrumentation 才能定，未做）

**3dgrut MCMC strategy + 6 cam OpenCVPinhole rational 在跨 cam 时 likelihood 跳变**：

- MCMC 每 ~100 iter relocate gaussians，需要算每个 cam viewpoint 下 gaussian 的 likelihood
- OpenCVPinhole rational `icD = (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶)` 在边缘 r² 大时数值跳变（CUDA `kMinRadialDist=0.8 / kMaxRadialDist=1.2` clamp 触发 fallback path）
- 6 个 cam 的 distortion 系数极端不同（rear_left k4=0.53 / front_tele k4=1.30 / cross_right k4=1.76），同一个 gaussian 在不同 cam 视角下的 likelihood map 跨 cam 不平滑
- MCMC relocate decision 跨 cam 反复 perturb → gaussians 永远不能 settle
- 单 cam 时所有 likelihood 来自一个 viewpoint，无 cross-cam 冲突 → 收敛到 28.44 dB
- PAI FTheta 是 polynomial mapping，跨 cam 视角 likelihood 平滑过渡 → 6cam 也 OK 到 26.31 dB
- NRE 内部 multi-cam 训练对 rational distortion 鲁棒（PAI/Hyperion 团队为同类数据设计过）

## 6. NCore reader 数据验证发现

启动 viser viewer 时撞到的 viewer bug：

| 文件 | 行号 | bug | fix commit |
|---|---|---|---|
| `viser_gui_4d.py` | `_load_metadata` fallback (L1824-1840) | `NCoreDataset(...)` 默认 `n_val_image_subsample=4` 跟 1918×1078 不整除 → assert | `8d86961` |
| `viser_gui_4d.py` | `_load_multi_cam_poses` (L1702-1719) | 同款 bug | `1052015` |

inceptio finalmask 是 **1918×1078**（非 1920×1080），ncore SDK 默认 n_val_image_subsample=4 触发 `1918 % 4 != 0` assert。两处 fallback 都加 `n_val_image_subsample=1`。

## 7. viser_gui_4d 对 v1 MoG ckpt 的 GUI 限制

`viser_gui_4d.py` 注册 Camera dropdown / Frame slider / Follow Camera 等控件**仅对 LayeredGaussians ckpt**。`ncore_3dgut_mcmc.yaml` (single-cam) 训出来是 **v1 MixtureOfGaussians (no layers)**，T8.13-DIAG 显示：
```
scene_mog type: MixtureOfGaussians (v1 MixtureOfGaussians, no layers)
viser checkboxes registered: []
```

→ 验证 single-cam ckpt 视觉只能看 **eval val PNG / mp4**（不能用 viser 交互看）。multilayer ckpt 触发完整 GUI panel 但渲染垃圾（PSNR 20）。

## 8. 复现命令

### aux 生成 (~2h)

```bash
docker run -d --name inc4cab_aux --gpus all --shm-size=32g \
  -v <CLIP_DIR>:/workdir/dataset \
  -v <AUX_DIR>:/workdir/output \
  -v /home/inceptio/.cache/torch:/home/.cache/torch \
  nvcr.io/nvidia/nre/nre-tools-ga:latest \
  ncore-aux-data \
    --dataset-path=/workdir/dataset/<SEQ>.json \
    --output-dir=/workdir/output \
    --segmentation-backend=mask2former --ego-mask \
    --depth-backend=depthanythingv2 \
    --dinov2-backend=none \
    --lidar-seg-camvis --num-threads=8 --store-meta \
    --camera-id camera_front_wide_120fov --camera-id camera_front_tele_30fov \
    --camera-id camera_cross_left_120fov --camera-id camera_cross_right_120fov \
    --camera-id camera_rear_left_70fov --camera-id camera_rear_right_70fov
# 完成后 cp aux/* 到 dataset 目录
```

### NRE 训练 (~75min, 28.99 dB)

```bash
docker run --rm --gpus all --shm-size=8g \
  -v <CLIP_DIR>:/workdir/dataset \
  -v <OUT_DIR>:/workdir/output \
  -v /home/inceptio/.cache/torch:/home/.cache/torch \
  nvcr.io/nvidia/nre/nre-ga:latest \
    --config-name=apps/prod/Hyperion-8.1/car2sim_6cam \
    mode=trainval \
    out_dir=/workdir/output \
    dataset.path=/workdir/dataset/<SEQ>.json \
    dataset.lidar_ids=[lidar_top_360fov] \
    "dataset.samplers.batch_sampler.camera_pixel_sampler.subsample=2" \
    "dataset.n_val_image_subsample=2" \
    "dataset.n_train_sequential_image_subsample=2" \
    "++trainer.max_steps=40000" \
    "++checkpoint.artifact.mesh.ground.enabled=false" \
    logger=dummy
# Lyra 报告坑：subsample 必须 ≤2（1918 % 4 != 0）+ ground mesh 必须关（0 road points crash）
```

### 3dgrut single-cam (~22min, 28.44 dB)

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && cd ~/repo/3dgrut2 \
  && nohup python train.py --config-name apps/ncore_3dgut_mcmc \
    n_iterations=30000 \
    num_workers=10 \
    path=<JSON> \
    "dataset.camera_ids=[camera_front_wide_120fov]" \
    "dataset.lidar_ids=[lidar_top_360fov]" \
    out_dir=<OUT> \
    experiment_name=inc4cab_singlecam_fullres \
    > /tmp/log 2>&1 & echo PID $!'
# 注：multi-cam 6 cam 会 fail；render.py:1447 json bug 让 metrics.json 0 bytes 但 ckpt 正常
```

## 9. 已知 limitations / 后续

### Known limitations

- **`render.py:1447` UnboundLocalError 'json'**：3 轮 3dgrut 训练 metrics.json 全 0 bytes（崩在 render_all 写盘瞬间，per-frame PSNR 从 log 抽）
- **3dgrut multi-cam × OpenCVPinhole rational distortion fail mode**：本报告主结论。当前 workaround = 单 cam 训练
- **viser_gui_4d 对 v1 MoG ckpt 几乎无 GUI panel**：只对 LayeredGaussians 提供完整 controls
- **inceptio converter 没接 cuboids**：当前 `converter.py` 写 `store_observations([])`，需要走 thinkpad 最新版 (commit 526c5b5, 含 `obstacles.py` parse_ppn_fusion) 重转才有 cuboid

### Open questions (后续 deep investigation)

1. **3dgrut multi-cam fail 是 MCMC 还是 sampling 问题**：runtime instrumentation log MCMC relocate decision 在跨 cam 切换时是否震荡，或用 2cam baseline (front_wide+front_tele) 看是否随 cam 数单调下降
2. **opencv → ftheta 转换是否能救 multi-cam**：thinkpad `calibration.py:opencv_pinhole_to_ftheta()` 把 OpenCVPinhole 拟合成 FTheta polynomial（120° front_wide 极端处 ~75px 误差）。如果 ftheta + 6cam 跑能上 25+ dB，证实 rational distortion 是 fail 元凶
3. **`kMinRadialDist=0.8 / kMaxRadialDist=1.2` clamp 边界**：放宽到 [0.5, 2.0] 测一下 6cam OpenCVPinhole 训练能否 recover

## 10. 关键 commits

| Commit | 改动 |
|---|---|
| `8d86961` | fix(viz_4d): `_load_metadata` fallback 加 `n_val_image_subsample=1` |
| `1052015` | fix(viz_4d): `_load_multi_cam_poses` 加 `n_val_image_subsample=1` |

两 commit 都在 branch `claude/awesome-haslett-0cc964` 上，已 push 到 inceptio remote。
