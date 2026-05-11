# WP V1-1 任务报告：NCore v4 校验器 + 首次训练

---

## 一、交付物汇总

| 文件 | 状态 | 说明 |
|---|---|---|
| `threedgrut/tools/__init__.py` | ✅ 已交付 | 包初始化 |
| `threedgrut/tools/ncore_validate.py` | ✅ 已交付 | 核心校验器 + manifest 生成器 |
| `schemas/scene_manifest.schema.json` | ✅ 已交付 | JSON Schema draft-07 |
| `threedgrut/tests/test_ncore_validator.py` | ✅ 已交付 | 12 项单元测试，全部通过 |

---

## 二、ncore_validate.py 功能概述

**CLI 接口：**
```bash
python -m threedgrut.tools.ncore_validate --clip <path_or_dir> --out <manifest.json>
```

**10 步校验流程：**

| 步骤 | 内容 | 阻断训练？ |
|---|---|---|
| Step 1 | 定位 clip JSON（文件或目录） | ERROR |
| Step 2 | JSON 结构校验（必填键、version=v4、时间戳单调性） | ERROR |
| Step 3 | NCore API 可达性（SequenceLoaderV4 初始化） | ERROR |
| Step 4 | 相机校验（数量、分辨率、模型类型、帧数、ego 掩码） | ERROR（0 相机） |
| Step 5 | LiDAR 校验（MCMC 初始化依赖） | ERROR（0 LiDAR） |
| Step 6 | Pose graph 校验（ego 轨迹、world_global 变换） | ERROR |
| Step 7 | 动态 tracks 枚举 | WARNING（可缺失） |
| Step 8 | Map metadata（xodr） | WARNING（可缺失） |
| Step 9 | 错误/警告汇总输出 | — |
| Step 10 | 生成 scene_manifest.json | — |

**校验结果（clip `0a119d27`）：**
```
[OK] scene_manifest.json written
     clip_id  : pai_0a119d27-7022-41f6-aa84-a095c97f85fa
     duration : 20.0s
     cameras  : 7 × 1920×1080 (FTheta)
     lidars   : ['lidar_top_360fov']
     tracks   : 0
```

---

## 三、单元测试结果

```
12 passed in 0.23s
```

覆盖场景：

| 测试用例 | 说明 |
|---|---|
| `test_find_clip_json_accepts_file` | 直接传入 .json 文件 |
| `test_find_clip_json_single_json_in_dir` | 目录中只有一个 JSON |
| `test_find_clip_json_uuid_dir_name_match` | 多 JSON 时按目录名（UUID）匹配 |
| `test_find_clip_json_not_found` | 目录中无 JSON → FileNotFoundError |
| `test_find_clip_json_ambiguous` | 多 JSON 且无法匹配 → ValueError |
| `test_missing_required_keys` | 缺少必填键 → 报错 |
| `test_wrong_version` | version != v4 → 报错 |
| `test_timestamp_monotonicity_equal` | start == stop → 报错 |
| `test_timestamp_monotonicity_reversed` | start > stop → 报错 |
| `test_missing_timestamp_fields` | 缺少 stop 字段 → 报错 |
| `test_cli_bad_json_exits_nonzero` | CLI 传入损坏 JSON → 非 0 退出码 |
| `test_cli_missing_file_exits_nonzero` | CLI 传入不存在路径 → 非 0 退出码 |

---

## 四、训练结果

### 4.1 2 秒子集训练（流程验证）

| 参数 | 值 |
|---|---|
| Clip | `pai_0a119d27-7022-41f6-aa84-a095c97f85fa` |
| 训练帧范围 | `duration_sec=2.0` |
| 相机 | `camera_front_wide_120fov`（单相机验证） |
| 步数 | 30,000 steps |
| 硬件 | NVIDIA A100 (vast.ai) |
| 训练时长 | ~29 分钟 |
| Checkpoint | `ckpt_last.pt`（676 MB） |

**KPI（Step 30000，Test Set）：**

| mean_psnr | mean_ssim | mean_lpips | mean_cc_psnr | mean_cc_ssim | mean_cc_lpips | std_psnr |
|---|---|---|---|---|---|---|
| **34.849 dB** | **0.955** | **0.191** | 34.836 | 0.955 | 0.191 | 1.796 |

---

### 4.2 全量训练

| 参数 | 值 |
|---|---|
| Clip | `pai_0a119d27-7022-41f6-aa84-a095c97f85fa` |
| 训练帧范围 | 全量 ~20 秒 |
| 相机 | 全部 7 个（1920×1080，FTheta） |
| 步数 | 30,000 steps |
| 硬件 | NVIDIA A100 (vast.ai) |

**KPI（Step 30000，Test Set）：**

> 训练完成后补充。

---

## 五、环境信息

| 项目 | 值 |
|---|---|
| 硬件 | NVIDIA A100 (vast.ai) |
| Python | 3.11 |
| PyTorch | 2.x + CUDA |
| NCore | nvidia-ncore >= 19.0.0 |
| 训练配置 | `apps/ncore_3dgut_mcmc` |
| Gaussian 初始化 | LiDAR 点云（MCMC densification） |
| 渲染后端 | 3DGUT（Slang rasterization） |
