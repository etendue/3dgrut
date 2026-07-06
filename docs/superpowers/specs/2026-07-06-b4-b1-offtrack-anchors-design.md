# B4 held-out 锚 + B1 双臂（off-track 战役无悔棋前两件）· 任务设计

- 日期：2026-07-06
- 状态：**已批准**（大g 逐节确认：编排方案 A、B4 双 ckpt × 双相机组矩阵、B1 双臂设计）
- 上游：[off-track 战役 spec](2026-07-03-offtrack-campaign-design.md) §3 无悔棋三件套 + §6 作战表 D1-D2；任务编号对应 [v5_plan.md](../../../v5_plan.md) B4 / B1
- 范围：三件套 D1 整体 = WP0 IPC 实物验证（已完成）+ WP1 B4 held-out 锚 + WP2 B1 双臂；**B5 不在本 plan**（但 B4 会顺带在 b6a9 打通 render-FID 链路，B5 只剩 novel 档适配）

---

## 0. 设计期已核实的事实（2026-07-06 探索）

1. **b6a9 NCore manifest 只含 6 台相机**（front_wide_120fov / cross_left_120fov / cross_right_120fov / left_wide_90fov / right_wide_90fov / back_rear_wide_90fov）。12 相机车型的另 6 台未进 NCore clip → 6-cam R3p baseline 在现有数据内无 held-out 相机可用；**B4 必须用 3-cam ckpt × 3 台未参训侧/后相机**。附带结论：C2「扩相机 5→10」实际需重跑 ncore 转换补相机（门 1 后再议，本条仅记录）。
2. **B4 纯 CLI 可行，零代码**：v4 E1.3 已实现 `--dataset-cameras`（`render.py`，eval 数据集构造前替换 `conf.dataset.camera_ids`，见 `threedgrut/render.py` 的 `apply_dataset_cameras_override`）。配套完备：
   - exposure（BilateralGrid，per-camera 参数存 ckpt）在检测到相机集覆盖时**自动禁用**并告警——held-out 相机无已学 exposure，避免索引错位；
   - `cc_psnr/cc_lpips` 为 per-frame 独立仿射色彩拟合（`color_correct_affine`），与相机索引无关 → held-out 可算、且曝光鲁棒；
   - per-camera 指标表对任意相机集合自动聚合进 metrics.json `per_camera` 键；
   - `--novel-fid` 内置 FID/KID（torchmetrics），render 流输出 `mean_render_fid/kid` = 渲染 vs 真图分布距离。
3. **WP0 IPC 实物验证：通过，无需补 smoke**。inceptio `~/work/nurec_e0/` 实物清单：`e07/fixer_done.flag`、`e07/ipc/`（fixer_server / harmonizer_server / model_ipc / e2e_client 全套）、`e07/launch_harmonizer_train.sh`（含 nre-ga 容器互斥 guard + READY 等待 + cmd/mounts 复放逻辑）、`train_out_e07/` 两个 40k run（Qm52…=Fixer 对照臂 α、SqPig…=Harmonizer 臂 β'，各 9 个 USDZ artifacts + hparams.yaml）、`e07/compare/`（lateral 3m/6m 渲染对比产物）、`render_compare.sh`/`render_rest.sh`。
4. **B1 资源在位**：`nvcr.io/nvidia/nre/nre-ga:latest` 与 `harmonizer-cosmos-env:latest` image 都在 inceptio；磁盘 2.8T 空闲；探索时 GPU 空闲。
5. **ckpt 位置**：旧锚 3-cam `~/work/output/inc_b6a9_3cam_multilayer_30k/`（psnr 21.04）；R0c `~/work/output/R0c_3cam_noopreg/`（20.84，aux 修复 + 正则 off）；R1p/R3p 参训 per-cam 数字在各自 `*_eval/` metrics.json 现成。

## 1. 目标

产出**门 1 的三个决策数字中的两个**（held-out gap、B1 臂差），并把 v5 KPI 表两行「未测」实测化：

- off-track 行：b6a9 真 GT 离轴数字（B4）
- NRE gap 行：官方配方双臂在 b6a9 的数字（B1）

## 2. 编排（大g 拍板：方案 A）

| 时段 | 工作包 | GPU 占用 |
|---|---|---|
| 今天白天 | WP1 B4 四个 render-only run（~1h GPU）+ WP2 臂 2 准备（cmd/mounts 适配，纯文件工作） | 轻 |
| 今晚 | WP2 臂 1：nre-ga 官方 baseline docker 挂机（40k，数小时） | 独占 |
| 明晚 | WP2 臂 2：+Harmonizer IPC 蒸馏（臂 1 退出后，guard 自动把关） | 独占 |
| 两臂完成后（D3） | WP2 收尾：`nre render` lateral 3m/6m 帧 + 两臂 FID 对比 | 轻 |

串行避开「nre-ga 训练容器与 IPC server 互斥」约束（e07 踩过 57% 白跑坑，launch 脚本已带 guard）。

## 3. WP1 — B4 held-out 侧相机真 GT off-track 锚

### 3.1 跑数矩阵（2 ckpt × 2 相机组 = 4 run，各 ~15min）

| run | ckpt | `--dataset-cameras` | 回答的问题 |
|---|---|---|---|
| 1 | 旧锚 3-cam（21.04） | 训练 3 台 front_wide/cross_left/cross_right | train-cam FID 基线（旧 eval 未开过 FID，补同口径对照面） |
| 2 | 旧锚 3-cam | held-out 3 台 left_wide/right_wide/back_rear_wide | **战役原问：b6a9 现在离轴差多少** |
| 3 | R0c | 训练 3 台 | 同上对照面（R0c 侧） |
| 4 | R0c | held-out 3 台 | 与 R1p 参训数字构成**单变量扩相机收益证据**（R1p = R0c + 6-cam，其余同配方） |

### 3.2 执行面

- 命令形状：`render.py --checkpoint <ckpt> --path <manifest> --out-dir <out> --dataset-cameras <逗号列表> --novel-fid`（细节参数执行期定）；inceptio 主仓库 `git pull` 同步 main HEAD（`a5083c8`）即可，无代码改动、不需 worktree。
- **顺序**：run 2 先行当 smoke——`--dataset-cameras` 系 E1.3 在 FTheta/PAI 线验证，b6a9 OpenCVPinhole 首用；确认 metrics.json `per_camera` + `mean_render_fid` 字段齐全后再跑其余 3 个。撞墙则转 systematic-debugging + TDD 修码（预期概率低，机制在 dataset 层与相机模型无关）。

### 3.3 读数与口径

- **主报**：`cc_psnr` + `lpips`（exposure 自动禁用 → cc 口径剥离曝光差）+ `mean_render_fid/kid`；raw psnr 附带。
- **对照面**：R1p/R3p 现成 metrics.json 参训 per-cam 数字（R3p：left 18.44 / right 21.24 / back 19.58）。
- **口径注记（入档必带）**：① FID 样本量 ~120 帧/组（<2048 有偏，仅作四组内部 A/B，不跨口径引用）；② held-out 相机 val split 由 datasetNcore 同规则自动切分，与训练相机 eval 同口径；③ 旧锚 ckpt 训练于 aux 修复前，但 eval 不消费 sseg aux，无影响；④ **run 1/3（训练相机组）同样经 `--dataset-cameras` 指定 → exposure 一样被自动禁用**——四组内部口径一致，但与历史 eval 数字（带 exposure）不可直接比，对照历史锚时用 cc 口径。

### 3.4 验收

- 4 组数字（per-cam cc_psnr/lpips + FID）双源交叉（rich log × metrics.json）后写入 v5 §4 Done Log；
- 两个结论单独成文入档：①「held-out − train」gap（KPI 表 off-track 行回填）；② R0c held-out vs R1p 参训差值（思路 B 第一实测证据）。

## 4. WP2 — B1 双臂 NRE 同 clip 对照锚

### 4.1 臂 1（官方 baseline，今晚）

- **命令面**：照 [4cab runbook](../../../inceptio_4cabad44_3dgrut_vs_nre.md) §8——nre-ga 容器 + `--config-name=apps/prod/Hyperion-8.1/car2sim_6cam`、`mode=trainval`、`++trainer.max_steps=40000`、`subsample=2` 三键、ground mesh off、`logger=dummy`；`dataset.path` 换 b6a9 manifest。b6a9 恰为 6 台 pinhole 相机，与配方对口。
- **启动前置检查（三项全过才发射）**：
  1. **depth aux 依赖**：b6a9 aux 系 `depth-backend=none` 生成；查 car2sim_6cam 配方是否消费 depth aux。若消费：**默认补跑 depth aux**（nre-tools depthanythingv2，~1h）——保持官方配方原味（B1 意义 = 官方口径天花板）；仅当补跑遇到实际障碍时才 CLI 关 depth 并在入档时标注该偏离。
  2. **lidar_ids**：查 b6a9 manifest lidar 命名（4cab 用 `lidar_top_360fov`；b6a9 多 lidar 需确认名称与选择）。
  3. GPU 空闲 + WP1 四 run 已收工。
- **启动模式**：`docker run -d --name b1_arm1_baseline`（不带 `--rm`，保留日志与容器现场）；**发射后必须验证进程存活**（7/3 事故教训）；Monitor 只 grep 关键节点（`Traceback|OOM|val/|artifacts|Error`）。
- **产物**：官方 val 指标 + USDZ artifacts + 训练日志。

### 4.2 臂 2（+Harmonizer IPC，明晚，单变量）

- 与臂 1 唯一差异 = `difix.training.enabled=true` + harmonizer_server 挂 :59487。
- 复用 e07 `launch_harmonizer_train.sh` 架构（guard → server 启动 → READY 等待 → cmd/mounts 复放），把 9ae 版 cmd/mounts JSON 适配为 b6a9 版（改 dataset path / out_dir / difix 开关）——**今天白天完成适配**，臂 1 退出后即可发射。
- 前置：臂 1 容器退出（guard 自动拦截）。

### 4.3 收尾（两臂完成后）

- 两臂 USDZ → `nre render` 出 lateral 3m/6m 帧（参照 e07 `render_compare.sh`）→ 与 B4 同工具算 FID → **两臂 off-track FID 差 = 门 1「蒸馏臂增益」输入**。

### 4.4 验收

- 两臂官方口径指标入档，**显式标注口径陷阱**（官方 val 每 3 帧 + 1/4 res + cpsnr，不与我方 21.04 等直接比）；
- lateral 3m/6m 两臂 FID 差入档（门 1 判据）；
- v5 KPI 表 NRE gap 行回填。

## 5. 风险表

| # | 风险 | 缓解 |
|---|---|---|
| 1 | car2sim_6cam 需 depth aux（b6a9 缺） | 前置检查①；补跑 nre-tools depth aux ~1h |
| 2 | `--dataset-cameras` 首用于 pinhole | run 2 先行 smoke；撞墙转 TDD 修码 |
| 3 | 官方配方对 b6a9 分辨率/多 lidar 兼容尖刺 | 4cab 两坑（subsample≤2、ground mesh off）预防带上；新报错走 systematic-debugging |
| 4 | 夜间 run 静默死亡 | 发射后验证存活 + Monitor 关键节点 + 次日晨检 |
| 5 | 数字入档伪造（已踩两次） | rich log × metrics.json 交叉验证（R3 长期纪律） |

## 6. 测试策略

- 本任务包无新代码 → 无新单测；唯一潜在写码点（风险 2 触发）按 TDD 走：先写回归测试 pin 住 pinhole × `--dataset-cameras` 的失败 case，再修。
- 数字验收即测试：每个 run 的 rich log 与 metrics.json 双源一致才算过。

## 7. 文档同步（完成定义的一部分）

- [v5_plan.md](../../../v5_plan.md)：§1.1/1.2 看板（B4 → Done，B1 → In Progress/Done）；§1.3 Phase B 计数；§4 Done Log 新条目（WP0 验证结论 + B4 四组数字 + B1 两臂数字 + lateral FID 差 + commit hash）；§0.2 KPI 表 off-track 行与 NRE gap 行回填。
- 门 1 素材就绪后（本包两个数字 + v4 E2.2 首档三读数）另开门 1 决策会话，本 plan 不含门 1 本身。
- 战役 spec（2026-07-03）为决策记录不回改；本 spec 为其 §3/§6 的任务级细化。

## 8. 明确出界（YAGNI）

- B5 novel 6 档移植（下一个 plan；B4 已顺带打通 b6a9 render-FID 链路）
- B1 之外的任何重训 / A800 联动（E2.2 归 v4 主线）
- C2 相机扩充的 ncore 重转换（门 1 后按数字排）
- viser 目检、行人指标、lane 锚（B2 冻结中）
