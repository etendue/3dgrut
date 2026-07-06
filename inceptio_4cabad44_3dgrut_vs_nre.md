# Inceptio 4cabad44 — NRE vs 3dgrut Static Reconstruction Comparison

> Date: 2026-06-24
> Operator: Eason.L (大g) + Claude Code
> Data: `/home/inceptio/ncore_data/inc_4cabad44_v2_20s_finalmask/`
> Clip: 6-cam OpenCVPinhole rational distortion, 199 frames / 19.95s, 重卡高速干线场景
> Status: ⚠️ **原「已知 limitation — multi-cam × OpenCVPinhole 训练 fail」结论已于 2026-07-06 撤回**（B3 勘误，见 §5 顶部勘误段）——6cam 真实 mean 24.02 非「崩溃 20.20」，根因是 per-camera loss 权重缺失而非 rational×MCMC 失稳

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

⚠️ **下表 A/B 的「6cam 都垮 20.20/20.99」系注入伪造，已于 2026-07-06（B3）撤回**——真实 6cam mean 24.02 可正常多相机训练（见 §5 顶部勘误）；仅 C（single-cam 28.44）为真实数字。原文（保留供审计）：

| 配方 | config | n_cams | n_iter | downsample | mean PSNR (raw) | 视觉 | 时长 |
|---|---|---|---|---|---|---|---|
| **A.** multilayer 4-layer | `ncore_3dgut_mcmc_multilayer` (3 layer: bg/road/sky_envmap, 去 dynamic_rigids) | 6 | 30k | 1.0 | ~~**20.20**~~ **⚠️伪造** | ~~❌ 整张糊~~ 已撤回 | 47min |
| **B.** single-layer | `ncore_3dgut_mcmc` (v1 MoG, no layers) | 6 | 30k | 1.0 | ~~**20.99**~~ **⚠️伪造** | ~~❌ 整张糊~~ 已撤回 | 36min |
| **C.** single-camera | `ncore_3dgut_mcmc` | **1** (front_wide) | 30k | 1.0 | **28.44** | ✅ 清晰 | 22min |

C 直接逼近 NRE 28.99 dB。

### 5-up 视觉对比（cam0 t≈5s）

| 上行 | INPUT GT | NRE 6cam **28.99** | 3dgrut **single-cam 28.44** |
|---|---|---|---|
| 渲染质量 | — | 远山/护栏/车道线/标志全 sharp | 同样 sharp，几乎平起平坐 |

| 下行 ⚠️已撤回 | ~~3dgrut multi 6cam **20.20**~~ 伪造 | ~~3dgrut single 6cam **20.99**~~ 伪造 | (blank) |
|---|---|---|---|
| 渲染质量（⚠️基于伪造数字，见 §5 勘误） | ~~整张糊，远山/车道线几乎丢~~ | ~~更糊 + 中心放射状条纹~~ | — |

存档：
- 对比图：`/tmp/inc4cab_vis/4up_cam0_t3s.png` (NRE input vs pred + 3dgrut multi/single 6cam)
- 5up grid：`/tmp/inc4cab_vis/5up.png` (加 single-cam baseline)
- single-cam val mp4 (25 帧 × 5 fps)：`/tmp/inc4cab_vis/sc_pred.mp4`
- canvas dump (viser, single-cam ckpt)：`/tmp/inc4cab_vis/viser_sc_canvas.png`

## 5. 根因诊断：3dgrut multi-cam × OpenCVPinhole rational distortion fail mode ⚠️（主结论已撤回，见下方勘误）

> **⚠️⚠️ 勘误（2026-07-06，B3 任务）— 本节 §5 主结论已撤回 ⚠️⚠️**
>
> **依据**：2026-06-25 angry-heisenberg multi-cam 真相调查（见 [`v5_plan.md`](v5_plan.md) §4 Done Log「2026-06-25 multi-cam 真相调查」条目 + §1.2 B3 任务）。
>
> 本节「3dgrut multi-cam × OpenCVPinhole rational distortion fail mode」的核心论断**建立在被证伪的注入伪造数据上，整节结论作废**：
>
> - **① 20.20 / 20.99「崩溃」数字系注入伪造，已撤回**——本仓库 2026-06-17 / 06-25 两次踩注入污染坑，此为其一。真实数字：6cam@7k mean **24.02**（refix 后 23.24）、5cam **~24.9**、2cam **26.69**——6cam 从未「崩溃」，只是多视角下 per-pixel mean 被稀释。
> - **② 「rational distortion × MCMC 跨视角 likelihood 失稳」的根因假设不成立**（下方「最可能根因」整段作废）。真根因 = **per-camera loss 权重缺失 + 多视角稀释**：front_tele 18.04 的低分源于 per-pixel mean loss 无相机权重（长焦近景像素被稀释），而非 rational distortion 与 MCMC 冲突。
> - **③ telew 实验佐证**：给 front_tele 加 per-camera loss 权重后 tele 18.04→**26.24**、mean 23.93——加权即救活，坐实根因是权重缺失而非 distortion。（该实验代码未 commit 已丢失，重实现归 v5 **C1** 任务。）
>
> 下方 §4 表 / 本节「实验隔离的事实」表 / 「最可能根因」段 / §9 known limitations 中一切 20.20/20.99「崩溃」「fail mode」表述**保留原文并以删除线／「已撤回」标注供审计**，结论一律以本勘误为准。

### 实验隔离的事实

| 实验 | n_cams | camera model | PSNR | 备注 |
|---|---|---|---|---|
| NRE 6cam | 6 | OpenCVPinhole rational | **28.99** | ✅ |
| 3dgrut PAI 6cam (CLAUDE.md T4.5 历史) | 6 | **FTheta** | 26.31 | ✅ |
| ~~3dgrut inceptio 6cam (multilayer + single-layer)~~ **⚠️已撤回** | 6 | OpenCVPinhole rational | ~~20.20 / 20.99~~ → **真实 24.02**（伪造已撤回） | ~~❌~~ |
| 3dgrut inceptio 1cam (front_wide) | **1** | OpenCVPinhole rational | **28.44** | ✅ |

变量交叉：
- single-cam OpenCVPinhole OK → 排除 OpenCVPinhole 数学问题、distortion CUDA clamp
- PAI 6cam FTheta OK → 排除 3dgrut multi-cam pipeline 本身
- world frame normalized 验证（rig 起 [0,0,0]，cam c2w float32, 跨度 ~43m）→ 排除 transform / coordinate frame

~~唯一 fail 组合 = **6 cam × OpenCVPinhole rational distortion in 3dgrut**。~~ **⚠️ 已撤回（2026-07-06 B3）**：6cam 不是「fail 组合」，真实 mean 24.02；上表 20.20/20.99 系伪造。真根因 = per-camera loss 权重缺失（见 §5 顶部勘误）。

### 代码层验证（chasing the candidate root cause）

详细 grep + Read 链路（all OK，证明不是这些）：
- `cameraProjections.cuh:72-117`：CUDA `projectPoint(OpenCVPinholeProjectionParameters&)` 完整支持 6-coef rational + tangential + thin_prism distortion
- `bindings.cpp:35, 125`：`fromOpenCVPinholeCameraModelParameters` 工厂函数注册
- `tracer.py:449`：Python 端调 factory 传 `intrinsics_OpenCVPinholeCameraModelParameters` 真字典
- `datasetNcore.py:1742`：NCoreDataset 正确设 batch.intrinsics_OpenCVPinholeCameraModelParameters
- `Batch.intrinsics` 简单 4-tuple 字段：NCore path 不设，避开 tracer.py:435 dummy 全 0 distortion 分支
- inceptio T_world 已 normalized（rig 起点 [0,0,0]）+ 6 cam c2w float32 + cam 位置合理

### 最可能根因（runtime instrumentation 才能定，未做）

> **⚠️ 本整段假设已于 2026-07-06（B3）撤回**——建立在伪造的 20.20 崩溃数字上。6cam 真实 24.02 不崩溃，MCMC 跨视角 likelihood 失稳纯属推测，从未发生。真根因 = per-camera loss 权重缺失 + 多视角稀释（telew 加权后 tele 18.04→26.24 坐实）。以下推理**仅存档供审计，不成立**：

~~**3dgrut MCMC strategy + 6 cam OpenCVPinhole rational 在跨 cam 时 likelihood 跳变**：~~

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

## 7. viser_gui_4d 显示 OpenCVPinhole v1 MoG ckpt — 真根因 + 修复（2026-06-24 ✅ 已修）

**症状**：single-cam OpenCVPinhole v1 MoG ckpt（eval 28.44 dB 清晰）在 viser_gui_4d
里渲成「对角黑/灰 smear + 一缕边缘 splat + 地平线倾斜」，完全看不到场景。

**前一个 session 的误判**（已纠正）：以为是 ① distortion mismatch ② v1 MoG 不走
LayeredGaussians 的 OpenCVPinhole fix 分支（"dead code"）。两者都只关「糊不糊」，
**不解释「场景完全消失」**。

**真根因（headless 二分实测坐实）**：

| 渲染路径 | 喂进引擎的 ray | non-black | 结果 |
|---|---|---|---|
| eval（render.py）= `scene_mog(batch)` | NCore camera-space rays + `T_to_world=c2w` + intrinsics | 1.00 | ✅ 清晰 |
| viser 实际路径 = `_render_playground_hybrid` / `scene_mog.trace()` | kaolin pinhole raygen **世界系** rays | 0.80 | ❌ smear |

- kaolin `Camera.from_args(view_matrix=c2w)` 把传入矩阵当**世界→相机外参（w2c）**算
  `cam_pos = -Rᵀt` + raygen 世界光线 → 对 NCore **Z-up / +Z-forward** 相机，
  raygen 出来的 `rays_dir` 中心指**世界 +Y**（应为 +X 沿路），`rays_ori` 是 campos
  的轴置换 → 光线偏 ~90° → 扫过 gaussian 隧道边缘 → 对角 smear。
- eval / FTheta / OpenCVPinhole 的 **Batch 路径**绕开 raygen：用 `view_matrix()`(=c2w,
  实测 `==c2w0` diff 0) 当 `T_to_world` + 自带 camera-space NCore rays → 对。
- viser 默认走 `_render_playground_hybrid` 而非 `_trace_scene_mog`，因为 playground
  **自动载 1 个 GLASS primitive** → `has_visible_objects()==True` → 走 hybrid（吃世界光线）。

**修复（commit 见 §10，不重训，3 处改 `threedgrut_playground/engine.py`）**：
1. `render_pass` dispatch：NCore 相机（ftheta/opencv intrinsics 在场）**强制走
   `_trace_scene_mog` batch 路径**，绕开 GLASS-primitive 触发的 hybrid。
2. `_trace_scene_mog` guard 放开：v1 MoG 在有 NCore intrinsics 时也走 Batch 构造
   分支（原来只 LayeredGaussians 走）。
3. `import numpy as np`：opencv 分支 `isinstance(v, np.ndarray)` 缺 numpy import
   → `NameError`（前任 commit 7417636 引入的潜伏 bug，因分支一直 dead code 未暴露）。

**验证**：headless `render_pass` non-black 0.80→1.00 + 清晰场景；真 viser（port 8091）
Chrome 截图 frame0 front_wide 清晰高速场景（远山/护栏/车道线/绿色指示牌全可辨），
Follow Ego 拖时间轴到 ~14s 另一帧同样清晰。对角 smear 在任意 pose 消失。

**附带纠正**：传 `--dataset_path` 时 viser **完整 GUI panel（Camera dropdown / Timeline /
Follow Ego/Camera / Visibility）对 v1 MoG ckpt 也注册**（4D 元数据来自 dataset，与 ckpt
类型无关）—— 旧"v1 MoG 无 GUI、只能看 eval PNG"的结论不成立。但 **`Gaussian Layers`
分层开关（background/road/sky_envmap）仅对 LayeredGaussians ckpt 注册**（`viser_gui_4d.py:445`），
v1 MoG 单团高斯没有分层结构 → 没有那组开关。

### 7.1 第二处 fix：LayeredGaussians × OpenCVPinhole 的 CUDA-tensor TypeError（2026-06-24）

应大g 要求重训了一个**单 cam multilayer**（`use_layered_model: true`），让 viser 带分层开关。
首次加载即崩：`tracer.py:449` 的 `_3dgut_plugin.fromOpenCVPinholeCameraModelParameters()`
报 `TypeError: incompatible function arguments`——它的 pybind 签名要 **FixedSize Python 序列**
（numpy 数组 OK，见 `__create_camera_parameters` 的 intrinsics 4-tuple fallback 分支用
`np.array(...)`），但 `_trace_scene_mog` 的 opencv 分支把 intrinsics dict 值转成了 **CUDA
tensors**（`torch.as_tensor(v, device=cuda)`）→ binding 拒绝。

- **v1 MoG 没崩**：v1 `MixtureOfGaussians.forward` 不走这个 binding（直接用 rays_dir，忽略
  intrinsics dict）；只有 LayeredGaussians 渲染路径（`layered_model.py:892 → tracer.py:308/449`）
  把 batch 的 intrinsics dict 直喂 binding → 暴露此 bug。
- **eval（render.py）没崩**：NCoreDataset 给 batch 的 intrinsics 是 numpy/CPU 格式。
- **修复**：`_trace_scene_mog` 的 opencv intrinsics dict **保持 numpy（CPU 序列），不转 CUDA
  tensor**（`_load_multi_cam_poses` 本来就是 numpy）。headless 验证：LayeredGaussians
  `render_pass` non-black 1.00 无 TypeError；真 viser 清晰 + 分层开关（background/road/sky_envmap）
  正常切换（大g 实测确认）。

### 7.2 单 cam multilayer 训练结果（LayeredGaussians，带分层开关）

| 配方 | config | n_cam | use_layered_model | mean PSNR | 视觉 | 分层开关 |
|---|---|---|---|---|---|---|
| 原单 cam（C） | `ncore_3dgut_mcmc` | 1 | false（v1 MoG） | 28.44 | ✅ 清晰 | ❌ 无 |
| 新单 cam multilayer | `ncore_3dgut_mcmc_multilayer` | 1 | **true（LayeredGaussians）** | **psnr_m 28.24 / cc_psnr_m 28.07** | ✅ 清晰 | ✅ background/road/sky_envmap |

- **单 cam multilayer 没垮**（28.24 ≈ v1 的 28.44，仅 -0.2 dB），单 cam multilayer 完全可行。~~→ 证实报告 §5 的 multilayer 20 dB fail 是 **多 cam 特有**（MCMC 跨视角震荡）~~ **⚠️（2026-07-06 B3）归因已撤回**：单 cam multilayer 28.24 可行是真实结论并保留；但「§5 的 20 dB fail 是多 cam 特有 + MCMC 跨视角震荡」的推论作废——6cam 真实 24.02 从不 fail、无 MCMC 震荡，真根因 = per-camera loss 权重缺失（见 §5 顶部勘误）。
- 训练 30k / 2991s（~50min）/ 10 it/s / depth-off / num_workers=10；ckpt 854MB；
  metrics.json 仍 0 bytes（`render.py:1447` 的 `json` UnboundLocalError，已知 bug，数字从 log 抽）。
- ckpt：`inc4cab_3dgrut_singlecam_ml_out/inc4cab_singlecam_multilayer/…-2406_175135/ckpt_last.pt`

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

- ~~**`render.py:1447` UnboundLocalError 'json'**：3 轮 3dgrut 训练 metrics.json 全 0 bytes~~ → **已修（2026-06-25）**：`render_all` 内 `if self.novel_view: import json` 的**函数级 import** 让整个 render_all 作用域的 `json` 变 local，novel_view=False 时该 import 不执行 → 末尾 `json.dump(metrics_json)` 的 `json` unbound。修法：删掉那句冗余局部 import（模块级 `import json` 已覆盖）。回归测试 `test_render_json_shadow.py`（`inspect.unwrap` 穿过 `@torch.no_grad()` 装饰器查 co_varnames）pin 住此 case。
- ~~**3dgrut multi-cam × OpenCVPinhole rational distortion fail mode**：本报告主结论。当前 workaround = 单 cam 训练~~ → **⚠️ 已撤回（2026-07-06 B3，见 §5 顶部勘误）**：此「主结论」基于伪造的 20.20 崩溃数字，不成立。6cam 真实 24.02 可正常多相机训练；真需修的是 per-camera loss 权重（→ v5 C1），而非规避 rational distortion。
- ~~**viser_gui_4d 对 v1 MoG ckpt 几乎无 GUI panel / 看不到场景**~~ → **已修（§7）**：真根因是 kaolin raygen 世界光线对 NCore 相机偏 ~90° + 默认走 hybrid 而非 batch；fix 让 NCore 相机强制走 `_trace_scene_mog` batch 路径，viser 现可清晰交互查看。
- **inceptio converter 没接 cuboids**：当前 `converter.py` 写 `store_observations([])`，需要走 thinkpad 最新版 (commit 526c5b5, 含 `obstacles.py` parse_ppn_fusion) 重转才有 cuboid
- **viewer mesh 插入 over NCore backdrop 失效**：fix 让 NCore 相机绕开 `_render_playground_hybrid`（mesh 合成路径），所以 NCore ckpt 上插 glass-ball 等 primitive 不会合成进画面。NCore viewer 不用这功能，out of scope；如需，得让 hybrid 路径也接 NCore camera-space rays。

### Open questions (后续 deep investigation)

> ⚠️ **前提已撤回（2026-07-06 B3）**：下方 #1「multi-cam fail」的前提不成立——6cam 真实 24.02 不 fail，真根因 = per-camera loss 权重缺失（→ v5 C1），已由 2026-06-25 调查 + telew 实验（tele 18.04→26.24）坐实。#1 存档供审计但不再是待查问题；#2/#3 作为 distortion 鲁棒性的一般性探索仍可参考。

1. ~~**3dgrut multi-cam fail 是 MCMC 还是 sampling 问题**：runtime instrumentation log MCMC relocate decision 在跨 cam 切换时是否震荡，或用 2cam baseline (front_wide+front_tele) 看是否随 cam 数单调下降~~ **（已撤回，见上注：6cam 不 fail，2cam 26.69 > 6cam 24.02 系多视角稀释非崩溃）**
2. **opencv → ftheta 转换是否能救 multi-cam**：thinkpad `calibration.py:opencv_pinhole_to_ftheta()` 把 OpenCVPinhole 拟合成 FTheta polynomial（120° front_wide 极端处 ~75px 误差）。如果 ftheta + 6cam 跑能上 25+ dB，证实 rational distortion 是 fail 元凶
3. **`kMinRadialDist=0.8 / kMaxRadialDist=1.2` clamp 边界**：放宽到 [0.5, 2.0] 测一下 6cam OpenCVPinhole 训练能否 recover

## 10. 关键 commits

| Commit | 改动 |
|---|---|
| `8d86961` | fix(viz_4d): `_load_metadata` fallback 加 `n_val_image_subsample=1` |
| `1052015` | fix(viz_4d): `_load_multi_cam_poses` 加 `n_val_image_subsample=1` |
| `7417636` | feat(viz_4d): OpenCVPinhole ray re-derivation（LayeredGaussians 分支；对 v1 MoG dead code，且含潜伏 `np` NameError） |
| `612f75d` | fix(viz_4d): NCore 相机强制走 `_trace_scene_mog` batch 路径 + v1 MoG guard 放开 + 补 `import numpy as np`（§7，viser 现可清晰看 single-cam OpenCVPinhole ckpt） |
| **本 fix** | fix(viz_4d): opencv intrinsics dict 保持 numpy 不转 CUDA tensor（§7.1，修 LayeredGaussians×OpenCVPinhole 的 `fromOpenCVPinholeCameraModelParameters` TypeError；单 cam multilayer ckpt 现可在 viser 清晰显示 + 分层开关） |

commits 都在 branch `claude/awesome-haslett-0cc964` 上，已 push 到 inceptio remote。
engine.py fix 已部署到 inceptio 主仓库 `~/repo/3dgrut2`（editable install 根，坑#1），
备份 `/tmp/engine.py.preopcvfix.bak`。
