# Pinhole→FTheta 转换验证与收编 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 按大g约定：本 plan 不贴 code snippet；每个任务只给 目标 / 文件 / 关键签名 / 测试断言要点+公差 / 验收命令意图。具体实现在执行期 TDD 写。

**Goal:** 把 OpenCV Pinhole→FTheta 转换（PIN-FTHETA-1 fitter）从"未跟踪代码"推进到"有全图机器精度真值验证、7 台相机逐台确认、GPU A/B 实证批准可用于 FTheta Arm B"的状态。

**Architecture:** 真值 = 对完整 OpenCV rational+tangential+thin-prism 前向模型做逐像素 Newton 反演（径向 LUT 初始化 + 2D Newton，残差 ~1e-16）；用它替换 `compute_opencv_reference_rays` 的径向盲区验证；再用 PIN-AB-1 已有的 common-domain 指标框架做 5s GPU A/B 实证门。

**Tech Stack:** 纯 numpy（Mac 可测，Task 1–4）；inceptio RTX 4090 worktree（Task 5）。

## Global Constraints

- Task 1–4 纯 numpy、Mac pytest 可跑，不引入 scipy/cv2 依赖。
- 多项式最小二乘**必须先归一化再拟合**（r/r_max、θ/max_angle）：未归一化时 r⁵≈1.6e15 直接吃掉 float64 精度（本会话实验 C 踩坑实证，ncore 版 t_scale/r_scale 同理）。
- inceptio 训练铁律：depth-off、`num_workers=10`、git worktree 隔离、5s 快测数字不与全量锚对比。
- kanban mermaid 节点标签内括号一律全角（）。
- 每任务完成后同步 `docs/pinhole_camera_kanban.md`（卡片状态 + Done Log + commit hash），文档不同步 = 任务未完成。

## 方案依据（2026-07-15 本会话实测，b6a9 front-wide 真实标定）

| 事实 | 数字 |
|---|---|
| repo fitter 全图角误差（vs 机器精度真值） | mean 0.0112° / P95 0.0259° / P99 0.0522° / max 0.1289° |
| 误差分解：径向 deg-5 拟合残差（主导） | mean 0.0105° / max 0.0902° |
| 误差分解：非径向 floor（tangential+fx≠fy，FTheta 不可表达） | mean 0.0041° / max 0.0402°（p1=4.7e-5, p2=8.8e-6） |
| 前向多项式像素误差（rasterizer 侧） | mean 0.162 px / max 1.131 px（仅最边缘） |
| deg-5 表达力上限（近 minimax） | mean 0.023° / max 0.057°（mean 与 max 是跷跷板） |
| gpt-sol 报告的数字 | 虚高 4–5 倍（疑似其管线整数像素量化伪影，待其交脚本对账） |

**明确不做的事**：
- ❌ 全图二维联合优化器——上限只能吃掉 0.004° 的非径向 floor，主项是 deg-5 一维表达力，投入产出不成立。
- ❌ gpt-sol 的 mean<0.01° 验收门——deg-5 minimax 上限 mean 0.023°，该门对本多项式阶数不可达；验收改为 Task 5 的下游 KPI 实证门。

---

### Task 1: PIN-FTHETA-1 收尾落地（fitter + 测试上分支）

**Files:**
- 已存在（未跟踪）: `threedgrut_playground/utils/ftheta_fitter.py`、`threedgrut/tests/test_ftheta_fitter.py`
- Modify: `docs/pinhole_camera_kanban.md`（新增 PIN-FTHETA-1 卡片）

**Interfaces:**
- Produces: `fit_ftheta_from_opencv_rational(pinhole_dict, n_samples=200, edge_margin_px=3) -> dict`（8-key FTheta dict），后续任务全部依赖。

- [ ] Step 1: 建分支 `feat/pinhole-ftheta-fitter`，跑 focused 测试确认现状全绿（意图：`pytest threedgrut/tests/test_ftheta_fitter.py`，期望 pass 无 fail）。
- [ ] Step 2: 跑 Mac full suite 确认无回归（意图：全量 pytest，期望与 main 基线一致 ~1034 passed/2 skipped 量级）。
- [ ] Step 3: commit fitter+测试；kanban 新增 PIN-FTHETA-1 卡片（状态 ✅，备注"radial-profile fit，全图验证见 PIN-FTHETA-3"）。

### Task 2: PIN-FTHETA-2 OpenCV 完整模型逐像素反演工具（真值发生器）

**Files:**
- Create: `threedgrut_playground/utils/opencv_inverse.py`
- Test: `threedgrut/tests/test_opencv_inverse.py`

**Interfaces:**
- Consumes: pinhole dict（同 fitter 输入约定）。
- Produces: `invert_opencv_full_model(pinhole_dict, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]`（undistorted normalized xy (N,2) + Newton 残差 (N,)）；`opencv_pixels_to_camera_rays(pinhole_dict) -> np.ndarray`（(H,W,3) float64 单位光线，含 tangential/thin-prism 完整模型）。

- [ ] Step 1: 写失败测试。断言要点：① b6a9 front-wide 全图 Newton 残差 max < 1e-12；② 前向（`PinholeForwardProjector.project_points`）→反演往返 normalized 坐标差 < 1e-10；③ 主点像素光线 = (0,0,1)（各分量公差 1e-9）；④ tangential/thin-prism 全零时与径向 LUT 反演角度差 < 1e-8 rad。
- [ ] Step 2: 跑测试确认按预期失败（模块不存在）。
- [ ] Step 3: 实现（径向 LUT 初始化 + 2D Newton 有限差分 Jacobian，8 iter；参考 scratchpad `verify_ftheta_fullimage.py` 已验证的算法，重写为库质量代码）。
- [ ] Step 4: 跑测试全绿。
- [ ] Step 5: commit。

### Task 3: PIN-FTHETA-3 全图角误差基准收编 + 修复自验证盲区

**Files:**
- Modify: `threedgrut_playground/utils/ftheta_fitter.py`
- Modify: `threedgrut/tests/test_ftheta_fitter.py`

**Interfaces:**
- Consumes: Task 2 的 `opencv_pixels_to_camera_rays`。
- Produces: `compute_fullimage_angular_error(pinhole_dict, ftheta_dict, outer_deg=55.0) -> dict`，key 含 `mean_deg / p95_deg / p99_deg / max_deg / outer_mean_deg / outer_p99_deg / nonradial_floor_mean_deg / forward_poly_max_px / forward_poly_mean_px`。

- [ ] Step 1: 写失败测试。断言要点（锚定实测值+余量，抓回归不卡死）：b6a9 front-wide `mean_deg<0.02`、`p95_deg<0.04`、`p99_deg<0.08`、`max_deg<0.15`、`outer_p99_deg<0.10`、`nonradial_floor_mean_deg<0.01`、`forward_poly_max_px<1.5`。
- [ ] Step 2: 跑测试确认失败（函数不存在）。
- [ ] Step 3: 实现 `compute_fullimage_angular_error`（真值光线来自 Task 2；非径向 floor = 全模型真值 vs 径向-only 真值的角差）。
- [ ] Step 4: 处理旧接口 `compute_opencv_reference_rays`：docstring 标注"径向插值参考，对方位角误差盲"，其现有测试（test_ftheta_fitter.py Test 7 三条）改为调用新函数并收紧公差（旧的 median<0.6°/P95<1.7° 松门替换为 Step 1 门）。
- [ ] Step 5: 跑 focused + full suite 全绿；commit；kanban 加卡 PIN-FTHETA-3 ✅（备注实测五元组数字入 Done Log）。

### Task 4: PIN-FTHETA-4 b6a9 全部 rational 相机逐台扫描报告

**Files:**
- Create: `scripts/pin_ftheta_camera_survey.py`（driver：读入各相机 pinhole 参数 json → 每台跑 fitter + `compute_fullimage_angular_error` → markdown 表格输出）
- Create: `scripts/pin_ftheta_b6a9_calibs.json`（7 台 rational 相机 + standard/tele 的标定参数，从 inceptio manifest 一次性提取固化，来源注明 clip/日期）
- Modify: `docs/pinhole_camera_kanban.md`（报告表格回填）

**Interfaces:**
- Consumes: Task 3 的 `compute_fullimage_angular_error`。
- Produces: 每台相机一行的判定表：`p1/p2 量级、非径向 floor mean/max、总角误差 mean/P99/max、前向像素误差 max、判定 🟢/🔴`。

- [ ] Step 1: ssh inceptio 从 b6a9 manifest 提取 9 台相机标定（只读 json，秒级；意图：一条 ssh + python 打印，粘入本地 json 固化）。
- [ ] Step 2: 写失败测试（`threedgrut/tests/test_ftheta_fitter.py` 追加）：survey 的单台评估函数对 front-wide 返回值与 Task 3 直接调用一致（相对公差 1e-6）。
- [ ] Step 3: 实现 survey 脚本；判定规则：`nonradial_floor_mean_deg < 0.01` 且 `forward_poly_max_px < 1.5` → 🟢，否则 🔴（🔴 = 该相机 FTheta 不可等价表达，升级决策）。
- [ ] Step 4: 本地跑 survey，9 台表格落 kanban §5 Evidence + Done Log；commit。
- [ ] Step 5: 决策点：全部 🟢 → 直接进 Task 5；存在 🔴 → 先向大g汇报该相机的处置（排除出 Arm B / 接受误差 / 提高多项式阶不可行说明）。

### Task 5: PIN-AB-2 相机模型替换 5s GPU A/B 实证门（inceptio）

**Files:**
- Create: `scripts/pin_ab2_ftheta_arm_driver.sh`（复用 PIN-AB-1 的 5s/9-cam/R6t/depth-off/nw=10 配方与 common-domain 指标工具）
- 接线改动（如需）: `threedgrut/datasets/datasetNcore.py`（数据侧提供"pinhole 相机以 fitted FTheta dict 出批"的开关，key 与原生 FTheta 一致 `intrinsics_FThetaCameraModelParameters`）

**Interfaces:**
- Consumes: Task 1 fitter 输出的 8-key dict（与原生 FTheta 批数据同构，理论上 rasterizer 零改动）。
- Produces: A/B 报告 json + kanban 结论；门判定结果。

- [ ] Step 1: Mac 侧先写数据接线的失败测试：开关打开时，OpenCVPinhole 相机的 batch 含 `intrinsics_FThetaCameraModelParameters` 且 8-key 完整、`resolution/principal_point` 与源一致；开关默认关、字节级不影响现有路径。
- [ ] Step 2: 实现接线 + focused/full 测试全绿；commit。
- [ ] Step 3: worktree 推 inceptio，双臂各 5k：Arm P = OpenCVPinhole + forward-valid mask（PIN-AB-1 胜者配方）；Arm F = fitted FTheta。同 seed/同 5s 窗/同步数，单变量 = 相机模型表示。启动用 setsid+三 fd 切断+ssh -n 模式。
- [ ] Step 4: 用 PIN-AB-1 的 common-domain 工具比较（两臂有效域交集上的 masked PSNR/SSIM/LPIPS，分 center/periphery）。
- [ ] Step 5: 门判定：common-domain masked PSNR 差 |ΔPSNR| ≤ 0.10 dB 且 periphery 无系统性劣化 → **转换批准用于 FTheta Arm B**，Task 6 取消；否则 Task 6 激活。
- [ ] Step 6: 结果 + commit hash + metrics.json 路径入 kanban Done Log（A800/inceptio 出口任务 ✅ 必须带实测数字）。

### Task 6: PIN-FTHETA-5（条件任务，仅 Task 5 门失败时激活）1D 拟合策略选项

**Files:**
- Modify: `threedgrut_playground/utils/ftheta_fitter.py`
- Modify: `threedgrut/tests/test_ftheta_fitter.py`

**Interfaces:**
- Produces: `fit_ftheta_from_opencv_rational(..., fit_strategy: str = "baseline")`，可选 `"minimax" | "uniform_r"`。

- [ ] Step 1: 写失败测试。断言要点：① 三种 strategy 产物 8-key 完整且多项式单调（复用现有单调测试参数化）；② `baseline` 路径输出与现版本逐系数一致（回归保护，公差 1e-12）；③ `minimax` 的全图 `max_deg < 0.07`（实测上限 0.057 + 余量）；④ 所有 strategy 内部拟合均走归一化域（用极端 r_max 参数构造病态回归测试：不归一化时会 fail 的 case）。
- [ ] Step 2: 实现（权衡数字备查：baseline mean 0.0107°/max 0.116°；minimax mean 0.023°/max 0.057°；uniform-r P99 0.037°/max 0.139°）。
- [ ] Step 3: 全绿、commit、用胜出 strategy 重跑 Task 5 双臂、kanban 收尾。

### Task 7: 文档同步 + gpt-sol 对账

- [ ] Step 1: kanban Board/依赖图加 PIN-FTHETA-2/3/4 与 PIN-AB-2 节点（全角括号），Done Log 逐条落。
- [ ] Step 2: 给 gpt-sol 的对账要求：交出其全图基准脚本；重点核对其真值是否经过整数像素量化（NCore `camera_rays_to_pixels` 返回整数，1px≈0.06°，与其虚高 4–5 倍量级吻合）；以本 plan 的机器精度基准为准绳。
