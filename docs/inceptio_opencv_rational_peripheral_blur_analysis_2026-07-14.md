# Inceptio OpenCV Rational 相机“中间清晰、外围眩晕”分析报告

**日期：** 2026-07-14
**状态：** 根因链高置信；修复方案待最小 A/B 验证
**数据：** Inceptio b6a9 12-camera clip
**主要模型：** C3 R6t 9-camera 30k checkpoint
**对照数据：** NVIDIA Physical AI（PAI）FTheta cameras

---

## 1. 问题与新增观察

在 Inceptio OpenCVPinhole 相机的重建结果中，画面呈现明显的径向质量差异：

- 图像中部相对清晰；
- 越靠外围越模糊、拉伸，主观上有“眩晕/涂抹”感；
- `camera_front_standard_55fov` 和 `camera_front_tele_30fov` 不明显；
- 其它 wide/cross/side/rear pinhole 相机较明显。

新增关键对照：

> NVIDIA Physical AI dataset 使用 FTheta camera model，在同一 3DGRUT/3DGUT 代码框架中没有观察到同类“中部清晰、外圈模糊”问题。

该对照说明问题不是“Gaussian Splatting 对所有广角相机天然外围模糊”，也不是 3DGUT 多相机框架的普遍缺陷，而更可能与 Inceptio OpenCV rational camera model 的有效域契约有关。

---

## 2. 结论摘要

### 2.1 主结论

没有发现常见的 intrinsic transport 错误：

- `resolution`、`principal_point`、`focal_length` 的 W/H、x/y 顺序正确；
- 6 项 radial coefficients 原样传递；
- tangential 和 thin-prism coefficients 原样传递；
- 3DGUT CUDA kernel 与 NCore SDK 使用同一个 OpenCV rational numerator/denominator 公式；
- 两边使用相同的 radial validity gate：`0.8 < icD < 1.2`。

但发现一个更根本的 camera-model contract 不一致：

> 对广角 OpenCVPinhole 相机，NCore iterative inverse `pixels_to_camera_rays()` 会为大量外围 pixels 生成 finite unit rays；然而相同 rays 经过 NCore/3DGUT forward projection 时会因超出可信 rational distortion domain 被判 `valid=False`。当前 dataset 只过滤 non-finite rays，没有过滤这些 forward-invalid pixels，因此仍用它们参与 photometric supervision。

该矛盾导致：

```text
外围 pixel 有 GT + 有 finite ray + 参与 loss
                 ↓
Gaussian forward projection 对该 ray/domain 判 invalid
                 ↓
中心区域可稳定优化，外围区域难以建立一致像素对应
                 ↓
中间清晰，外围逐渐涂抹/眩晕
```

### 2.2 FTheta 为什么没有同类问题

PAI FTheta 使用一组成对、显式定义的映射：

- `pixeldist_to_angle_poly`：pixel radius → ray angle；
- `angle_to_pixeldist_poly`：ray angle → pixel radius；
- `max_angle`：统一定义 forward/inverse 的有效视锥。

3DGUT FTheta kernel 在 reference polynomial 是 inverse direction 时，会用 Newton iterations 反演同一 polynomial pair，再以 `theta < max_angle` 判断有效性。因此 forward projection 与 inverse ray generation共享同一角度域，而不是 OpenCV rational 路径中“inverse 仍产 ray、forward 却因 radial trust gate 判 invalid”的状态。

PAI 的既有实测也支持这一点：

- 9ae PAI clip 的 5 台实测 camera 全部是 FTheta；
- FTheta cuboid projection 5-camera 验证通过，34 个可见 cuboid 无漏画/误画；
- 3DGRUT PAI 6-camera 历史锚约 26.31 dB；
- PAI 5 秒 5k 回归 A/B 为 22.499 vs 22.587 dB，FTheta 路径无 non-finite guard 命中；
- 未观察到 Inceptio rational wide camera 同样整齐的 `r≈0.7` 外围退化边界。

因此 PAI/FTheta 是一个有力的代码级对照：

> 同一 Gaussian 表示、同一 3DGUT renderer，FTheta 使用统一 polynomial angle domain 时没有该症状；OpenCV rational wide camera 的 inverse/forward trust domain 不一致时出现症状。

---

## 3. Camera 参数与代码链路核查

### 3.1 参数路径

OpenCVPinhole 参数经过：

```text
NCore CameraSensor.model_parameters
  → camera_model.get_parameters()
  → NCoreDataset._get_camera_model_parameters_for_resolution()
  → Batch.intrinsics_OpenCVPinholeCameraModelParameters
  → threedgut_tracer/tracer.py
  → _3dgut_plugin.fromOpenCVPinholeCameraModelParameters()
  → 3DGUT CUDA camera projection kernel
```

实际字段：

```text
resolution[2]
principal_point[2]
focal_length[2]
radial_coeffs[6]
tangential_coeffs[2]
thin_prism_coeffs[4]
```

`model_params.transform(image_domain_scale, new_resolution)` 负责分辨率缩放；distortion coefficients 与 resolution 无关，保持不变。

### 3.2 Rational 公式一致

NCore SDK 与 3DGUT kernel 均使用：

\[
icD =
\frac{1+k_1r^2+k_2r^4+k_3r^6}
     {1+k_4r^2+k_5r^4+k_6r^6}
\]

并加入 tangential 与 thin-prism：

\[
\Delta x = p_1(2xy)+p_2(r^2+2x^2)+r^2(s_1+r^2s_2)
\]

\[
\Delta y = p_1(r^2+2y^2)+p_2(2xy)+r^2(s_3+r^2s_4)
\]

两边都有：

```cpp
validRadial = (icD > 0.8) && (icD < 1.2)
```

因此 3DGUT kernel 没有把 radial coefficient 顺序读错，也没有把 rational 模型误写成普通 polynomial。

---

## 4. 真实标定的视角分组

所有相机分辨率均为 `1920×1080`。根据 NCore `pixels_to_camera_rays()`，图像边角最大 ray angle 为：

| Camera 组 | 典型边角 ray angle | 观察 |
|---|---:|---|
| `front_tele_30fov` | 约 17° | 外围眩晕不明显 |
| `front_standard_55fov` | 约 35° | 外围眩晕不明显 |
| front/cross/rear wide、rear-left/right、side wide | 约 67–69° | 外围眩晕明显 |

这说明症状与真实 ray angle/非线性程度相关，而不是 camera id 或 Viser dropdown 的偶发现象。

所有缓存 rays 都是单位向量，且在现有 `repair_nonfinite_rays()` 后 finite。唯一已知极点是：

```text
camera_left_wide_90fov：1 pixel/frame non-finite rational pole
```

现有 guard 已修复并 mask 该 pixel，但这只覆盖极少数 NaN/Inf，不覆盖大量 finite-but-forward-invalid 的外围 rays。

---

## 5. Pixel→ray→pixel Round-trip 证据

测试链：

```text
pixel
  → NCore pixels_to_camera_rays()
  → NCore camera_rays_to_pixels()
```

必须读取 `camera_rays_to_pixels().valid_flag`，不能只看返回的数值坐标。对于超出可信 distortion domain 的 ray，SDK 会给一个指示越界方向的裁剪坐标，但明确标记 `valid=False`。

### 5.1 全图有效比例

| Camera | 全图 forward-valid 比例 |
|---|---:|
| front standard | 100.000% |
| front tele | 99.997% |
| 其它广角 pinhole | 约 62.9–64.6% |

### 5.2 按归一化图像半径分桶

广角 pinhole 的共同结果：

| 归一化半径 | Round-trip forward-valid 比例 |
|---|---:|
| `r < 0.4` | 100% |
| `0.4 ≤ r < 0.7` | 100% |
| `0.7 ≤ r < 0.9` | 约 65–69% |
| `0.9 ≤ r < 1.1` | 约 32–34% |
| `r ≥ 1.1` | 接近 0% |

稳定有效域的拐点约为：

```text
normalized radius r ≈ 0.7
```

### 5.3 具体例子

`camera_front_wide_120fov`：

```text
input pixel = [1919, 540]
pixels_to_camera_rays → finite unit ray，约 60°
camera_rays_to_pixels → [3163, 540], valid=False
```

同一个 camera model 的 inverse 生成了 ray，但 forward 不承认该 ray 能在可信域内投回原 pixel。

---

## 6. Dataset supervision 缺口

当前 `datasetNcore.py` 只对以下情况做修复/屏蔽：

```python
repair_nonfinite_rays(...)
```

它过滤：

- NaN；
- Inf；
- rational pole 的极少数非 finite ray。

但没有构造：

```python
roundtrip = camera_model.camera_rays_to_pixels(all_rays)
forward_valid = roundtrip.valid_flag
```

也没有将其并入：

```python
camera_valid_pixels_ego_mask
```

因此约 35–37% 的广角外围 pixels 虽然 forward-invalid，仍可能参与：

- RGB L1；
- SSIM；
- image-space regularization；
- eval metric。

这构成当前最具体的训练契约缺口。

---

## 7. Native render 径向质量证据

分析对象是 `render.py` 保存的 UI-free native `renders/*.png` 与 `gt/*.png`，不是 Viser 浏览器截图，因此可排除：

- Viser perspective background plane；
- browser scaling；
- canvas crop/rotation；
- GUI presentation。

C3 R6t 9-camera 30k 的径向结果：

| Camera | Center PSNR | Edge PSNR | Edge−Center |
|---|---:|---:|---:|
| front standard | 22.99 | 24.70 | +1.71 |
| front tele | 24.04 | 25.32 | +1.28 |
| front wide | 22.76 | 20.24 | −2.52 |
| cross left | 21.84 | 17.34 | −4.50 |
| cross right | 21.83 | 17.46 | −4.37 |
| left wide | 18.41 | 16.12 | −2.29 |
| right wide | 26.61 | 15.92 | −10.69 |
| back rear wide | 19.87 | 18.85 | −1.02 |
| rear left | 20.67 | 19.47 | −1.20 |

梯度保持率同样向外围下降。例如 front-wide：

```text
center predicted/GT gradient ratio ≈ 0.311
edge predicted/GT gradient ratio   ≈ 0.056
```

因此外围不只是几何畸变，而是预测纹理和边缘被明显抹平。

---

## 8. 为什么 FTheta 代码路径更一致

### 8.1 Inverse ray generation

FTheta pixel ray 使用：

\[
\theta = P_{r\rightarrow\theta}(r_{pixel})
\]

\[
d = [\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta]
\]

代码路径：

```text
NCore FThetaCameraModel.pixels_to_camera_rays()
或 viewer helper ftheta_pixels_to_camera_rays()
```

### 8.2 Forward Gaussian projection

3DGUT kernel 根据 `reference_poly`：

- reference 为 `ANGLE_TO_PIXELDIST`：直接评估 forward polynomial；
- reference 为 `PIXELDIST_TO_ANGLE`：用 Newton iterations 反演该 reference polynomial；
- 最终统一使用 `theta < max_angle` 作为有效域。

代码位置：

```text
threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh
```

### 8.3 关键差异

| 项 | OpenCV rational pinhole | FTheta |
|---|---|---|
| inverse | iterative undistort | 显式 pixel-distance→angle polynomial |
| forward | rational numerator/denominator | 显式 angle→pixel-distance polynomial或对 reference 做 Newton 反演 |
| 有效域 | forward 额外检查 `0.8 < icD < 1.2` | `theta < max_angle` |
| 当前风险 | inverse 可产 ray，但 forward trust gate 可拒绝 | forward/inverse 共用同一 polynomial pair 与 angle cone |
| 本项目观察 | wide 外围约 35–37% forward-invalid | PAI 未见同类径向失效症状 |

因此 PAI/FTheta 的正常表现与本次根因在代码上能够直接联系起来。

---

## 9. 独立发现：Viser Pinhole overlay projector 公式错误

文件：

```text
threedgrut_playground/utils/pinhole_projector.py
```

当前实现错误地把 6 项 radial coefficients 当成普通多项式：

\[
1+k_1r^2+k_2r^4+k_3r^6+k_4r^8+k_5r^{10}+k_6r^{12}
\]

正确应为 rational numerator/denominator，并包含 thin-prism 与相同 validity gate。

与 NCore SDK 对照时：

- standard/tele 外围也出现数百到数千 pixel overlay 误差；
- wide camera 的自定义 projector 数值可发散到图外；
- 该 bug 会影响 cuboid、trajectory、label 的 image-space overlay；
- 它不解释 Gaussian backdrop 本身的外围模糊，因为 backdrop 使用的是正确的 SDK rays + 3DGUT rational kernel。

因此需要把两个问题分开：

1. **Backdrop 中清边糊**：dataset forward-valid supervision domain 缺口；
2. **Overlay 外围错位**：自定义 pinhole projector 的 rational 公式错误。

---

## 10. 已排除、降级与保留项

### 已排除/显著降级

- Intrinsic W/H 或 x/y 颠倒；
- focal/principal point 未缩放；
- radial coefficients 传输顺序错误；
- 3DGUT kernel 把 rational 写成 polynomial；
- 3DGUT 与 NCore 使用不同 trust gate；
- rays 未归一化；
- Viser presentation 是 native blur 主因；
- 大量 NaN/Inf rays。

### 仍可能是次级因素

- 固定模型容量在 9 cameras 间摊薄；
- Gaussian covariance 在高角度 nonlinear Jacobian 下更难优化；
- 多相机 exposure/temporal synchronization；
- 外围内容本身更斜视、低纹理、遮挡更多；
- UT 的 `ut_require_all_sigma_points_valid` 当前与官方配方不同。

这些因素可能造成剩余 blur，但无法单独解释 `r≈0.7` round-trip validity 拐点与 standard/tele 对照。

### 旧结论防误用

2026-06-25 的“6-cam × rational distortion 导致 MCMC 全局失稳”曾建立在伪造/错误的 20.20/20.99 指标上，已于 B3 撤回。本文不复活该结论。

本文的新结论是更窄、更可验证的：

> OpenCV rational wide camera 的外围 inverse rays 与 forward-valid projection domain 不一致，而 dataset 未屏蔽 forward-invalid supervision pixels。

它不等价于“rational camera 整体不可训练”；中心/有效域、单相机和 standard/tele 均可正常训练。

---

## 11. 建议验证与修复计划

### 11.1 A/B-1：Forward-valid supervision mask（最高优先级）

初始化每颗 OpenCVPinhole camera 时：

```python
rays = camera_model.pixels_to_camera_rays(all_pixels)
roundtrip = camera_model.camera_rays_to_pixels(rays)
forward_valid = roundtrip.valid_flag.reshape(H, W)
```

并入现有 valid mask：

```python
camera_valid_mask &= forward_valid
```

要求：

- 只对 `OpenCVPinholeCameraModel` 启用；
- FTheta 路径保持字节等价；
- 记录 per-camera valid coverage；
- train 与 eval 使用一致 mask；
- 保留 non-finite repair 作为安全防线。

### 11.2 5 秒快测协议

遵循仓库约定：

```text
dataset.train.duration_sec=5
dataset.val.duration_sec=5
```

Arm A：当前 9-cam baseline。
Arm B：加入 forward-valid mask。

重点报告：

- 每相机 center/mid/outer/edge PSNR；
- forward-valid 区域内 PSNR；
- `r<0.7` 守护线；
- `0.7<r<0.9` 的有效子域；
- standard/tele 是否保持基本不变；
- wide/cross/side 的视觉眩晕是否减轻；
- overall masked PSNR 不跨 mask 口径误比。

### 11.3 A/B-2：UT sigma-point validity

PAI/官方配方曾使用更严格的：

```text
ut_require_all_sigma_points_valid = true
```

本项目历史配置为 false。可在 A/B-1 后单变量测试：

- false：现状；
- true：任一 sigma point 投影失败即剔除 Gaussian。

该项可能减少高畸变边缘 splat 拉丝，但不是 forward-valid supervision mask 的替代。

### 11.4 修复 Viser overlay projector

`PinholeForwardProjector` 应：

1. 实现 rational numerator/denominator；
2. 实现 tangential + thin prism；
3. 使用 `0.8 < icD < 1.2`；
4. 对 SDK `camera_rays_to_pixels()` 做逐点 parity test；
5. 覆盖 standard、tele、wide 和 invalid peripheral rays；
6. 重新打开 mixed-camera buglist 中 OpenCVPinhole overlay 外围对齐条目，修复后再关闭。

---

## 12. 推荐判定

当前最合理的工程判断：

1. **不要手工修改 intrinsic coefficients。** 参数传输与 kernel 公式没有发现错位；改系数会破坏真实标定。
2. **优先修 valid-domain supervision contract。** 这是有直接 SDK valid_flag、kernel gate 和 native radial quality 三重证据的缺口。
3. **单独修 overlay projector。** 它是确定的 rational 实现错误，但不是 backdrop blur 根因。
4. **以 PAI/FTheta 作为回归对照。** 新 mask 必须对 FTheta 路径 no-op，PAI 指标与画面不得退化。
5. **先 5 秒机制 A/B，不直接跑 30k。** 若 radial blur 没改善，停止扩展修复并重新检查 Gaussian/UT projection mechanics。

---

## 13. 关键文件

```text
threedgrut/datasets/datasetNcore.py
  - pixels_to_camera_rays cache
  - repair_nonfinite_rays
  - camera valid masks
  - camera parameter extraction

threedgut_tracer/tracer.py
  - Batch intrinsics → plugin parameters

threedgut_tracer/include/3dgut/kernels/cuda/sensors/cameraProjections.cuh
  - OpenCV rational forward projection
  - radial validity gate
  - FTheta polynomial/Newton projection

threedgrut_playground/utils/ftheta_intrinsics.py
  - FTheta pixel→ray helper

threedgrut_playground/utils/ftheta_projector.py
  - FTheta world→pixel overlay

threedgrut_playground/utils/pinhole_projector.py
  - 当前错误的 polynomial radial overlay implementation

threedgrut_playground/engine.py
  - Viser SDK rays + camera parameter Batch path
```

---

## 14. 最终一句话

> PAI/FTheta 没有“中清边糊”，在代码上可以解释：FTheta forward/inverse 共享成对 polynomial 和统一 `max_angle` 有效锥；Inceptio OpenCV rational wide cameras 的 iterative inverse 会为图像外围产生 finite rays，但相同 rays 在 NCore/3DGUT forward projection 中大量超过 `0.8<icD<1.2` 信任域而 invalid，dataset 却继续监督这些 pixels。该正逆有效域不一致是当前最有证据的主根因。
