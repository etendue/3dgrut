# PIN-CAM-1 OpenCV Camera Contract Final Validation

日期：2026-07-17
机器：inceptio RTX 4090
数据：`inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9`

## 1. 结论

PIN-CAM-1 的主因由受控实验定位为 3DGUT CUDA 的固定
`0.8 < icD < 1.2` gate。用标定域 certificate 替代固定 gate 后，九相机
5 秒 / 5k A/B 的全局 masked PSNR 提升 **0.882 dB**，CC masked PSNR
提升 **0.883 dB**，masked LPIPS 降低 **0.0175**；front-wide 的固定径向
最外环 `r>=0.9` PSNR 提升 **2.719 dB**。

30 轮 OpenCV rational inverse 是正确性修复：10 轮在外围未收敛，30 轮把
front-wide 最大重投影残差从 7.321 px 降到 0.0122 px。不过同配方 5k
训练中，10→30 轮的画质差异接近噪声，因此它不是本次外围模糊的画质主因。

valid-only UT 是 mixed-valid sigma point 的安全修复，但 2×2 消融表明它在
calibrated gate 下画质近似中性；不能把九相机收益归因于 UT。

本修复保持原始 RGB、pixel grid、timestamp、pose 与 OpenCV rational 标定
不变。OpenCV→FTheta RGB remap 路线不属于本修复。

## 2. 生产改动

- OpenCV rational inverse 默认 30 轮，保留配置开关复现 10 轮旧行为。
- 由扩展图像边界的完整 rational inverse 计算 `max_valid_r2`，并验证整个
  `[0,max_valid_r2]` 区间无 pole、无 fold。
- CUDA certified 路径检查 `r²`、denominator、radial derivative 与 finite；
  certificate 失败时回退 legacy gate。
- mixed-valid UT 只聚合有效 sigma points 并重新归一化；全有效路径保持旧公式。
- CPU overlay/projector 在提供 `max_valid_r2` 时使用同一 certified domain；
  没有 certificate 时保留 legacy icD gate。
- `mask_forward_invalid_pixels=true` 与 calibrated validity domain 互斥并 fail-fast，
  避免旧 SDK mask 再次裁掉刚恢复的合法外围。

生产实现：`57d4cd7`、`3649f74`、`0994f21`、`47aeb4b`。复现实验 driver：
`9322bab`。

## 3. Front-wide 2×2 消融

共同条件：inverse=30、单 front-wide、5 秒、5k、depth-off、`num_workers=10`。

| Gate | UT | PSNR masked | CC PSNR masked | LPIPS masked |
|---|---|---:|---:|---:|
| legacy | legacy | 23.8528 | 23.0319 | 0.51778 |
| legacy | valid-only | 23.5397 | 22.7329 | 0.52142 |
| calibrated | legacy | **25.3481** | **24.8152** | 0.46286 |
| calibrated | valid-only | 25.3034 | 24.7525 | **0.46272** |

calibrated gate 在 legacy UT 下 masked PSNR **+1.4953 dB**、CC masked PSNR
**+1.7833 dB**、LPIPS **-0.0549**。valid-only UT 在 calibrated gate 下
masked PSNR **-0.0447 dB**，属于近似中性。

## 4. 九相机 5 秒 / 5k A/B

共同条件：同一九相机列表、同一 5 秒窗口、5000 steps、inverse=30、
depth-off、`num_workers=10`、front-tele loss weight=2.0。唯一组合差异：

- Legacy：calibrated domain=false，valid-only UT=false。
- Full fix：calibrated domain=true，valid-only UT=true。

| 指标 | Legacy | Full fix | Δ |
|---|---:|---:|---:|
| mean PSNR | 20.2183 | 20.9621 | **+0.7438** |
| mean PSNR masked | 21.9119 | 22.7940 | **+0.8821** |
| mean CC PSNR masked | 19.1857 | 20.0689 | **+0.8832** |
| mean LPIPS masked | 0.54303 | 0.52553 | **-0.01750** |
| mean SSIM masked | 0.67894 | 0.68372 | **+0.00477** |
| road crop PSNR | 25.5098 | 26.0952 | **+0.5855** |
| class PSNR | 17.6864 | 18.2505 | **+0.5642** |
| train time | 395.34 s | 392.21 s | -3.13 s |

### 4.1 固定径向区域与共同有效域

| Camera | self-mask PSNR Δ | common-domain PSNR Δ | `r0.7-0.9` Δ | `r>=0.9` Δ |
|---|---:|---:|---:|---:|
| front_wide_120 | +2.034 | **+1.755** | +2.075 | **+2.719** |
| cross_left_120 | +0.844 | +0.176 | +1.047 | -0.298 |
| cross_right_120 | +1.108 | +0.265 | +1.534 | +0.848 |
| left_wide_90 | -0.015 | +0.248 | +0.437 | -0.529 |
| right_wide_90 | -0.123 | +0.010 | -0.409 | +0.197 |
| back_rear_wide_90 | +1.792 | +0.173 | **+3.686** | +0.626 |
| rear_left_70 | +1.529 | **+1.162** | +0.439 | +0.971 |
| front_standard_55 | +0.581 | +0.574 | +0.169 | +1.043 |
| front_tele_30 | +0.381 | +0.379 | +0.229 | +0.464 |

共同有效域 PSNR **9/9 相机全部改善**；固定 `r0.7-0.9` 为 8/9 改善，
最外环为 7/9 改善。left/right 的 self-mask 小幅负值主要受评估域变化影响。

持久化证据：

- Full-fix metrics：`/home/inceptio/work/output/pin_cam_9cam_full_fix_5s_5k_eval/`
  `pin_cam_9cam_full_fix_5s_5k/`
  `inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1607_192111/metrics.json`。
- Legacy metrics：`/home/inceptio/work/output/pin_cam_9cam_legacy_5s_5k_eval/`
  `pin_cam_9cam_legacy_5s_5k/`
  `inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1607_190444/metrics.json`。
- 径向报告：`/home/inceptio/work/output/pin_cam_9cam_radial_ab/`
  `radial_analysis_report.json`。

## 5. 数值与运行守护

- 九台 camera 都成功生成标定域 certificate，日志无 `could not certify`。
- rational pole 非有限 ray：left-wide 6 px、right-wide 7 px，均由既有 repair
  与 permanent-invalid mask 处理；离散 pole 不阻塞任务。
- 2×2 四臂与九相机两臂都满足 train exit 0、checkpoint、eval exit 0、
  parseable `metrics.json`。
- 原生产分支 Mac suite 为 **1116 passed, 2 skipped**；实验分支 fresh suite
  为 **1349 passed, 2 skipped**。

结论：PIN-CAM-1 的机制验证、生产实现、九相机守护、真实 CUDA build/render
与证据归档均完成。5 秒窗口用于机制验收；发布级长训不阻塞本修复合并。
