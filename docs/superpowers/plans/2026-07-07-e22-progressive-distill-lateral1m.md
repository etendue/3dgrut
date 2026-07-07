# E2.2 渐进蒸馏（第一档 lateral_1m）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **格式说明（大g 约定，覆盖 writing-plans 默认）**：不贴大段代码块；每任务写目标 / 文件 / 关键函数签名（inline）/ 测试断言要点 / 命令意图。代码执行期 TDD 写。
>
> **场地决策（大g 2026-07-07）**：inceptio 被 B1 双臂 docker 占用 → **蒸馏臂落 A800**（Task 0 先重建被清的 env）；渲染+修复必须 inceptio（harmonizer 容器所在），**趁 B1 两臂间隙错峰**。

**Goal:** 打通「渲染→Harmonizer 修复→伪 GT 概率混采蒸馏回 3D」管线并完成第一档 lateral_1m 三读数验证（门 1 第三数字 + 门 2 推进依据）。

**Architecture:** 离线批量包循环——inceptio 生产（render-only 出 375 帧 → e21 IPC 批修复）→ rsync → A800 双卡 λ∈{0.1, 0.3} 并行蒸馏（锚 ckpt 续训 3k 步）→ 三读数（守护线 A800 / 档位 FID+lane 回 inceptio 评）。唯一新代码 = trainer 伪 GT 注入（`distill.*` config，默认字节等价）。

**Tech Stack:** 3dgrut trainer/render、e21_harmonizer_batch_fix（IPC）、eval_frames_dir、novel_view.perturb_c2w、A800 双卡 + inceptio 4090。

**Spec:** [2026-07-06-e22-progressive-distill-design.md](../specs/2026-07-06-e22-progressive-distill-design.md)

## Global Constraints

- **⚠️ log 不进 /tmp**（A800 定时清理实锤 2026-07-07：/tmp 日志 + conda env 被清）——一切训练/渲染 log 重定向到输出目录内（`<out_dir>/<run>/launch.log`）。
- **⚠️ A800 产物即时回传**：每个蒸馏 run 完成后立即把 `ckpt_last.pt` + `metrics.json` + launch.log rsync 回 inceptio `~/work/backup_a800/e22/`（防管理员清理）。
- **A800 env（Task 0 重建后）**：PATH 前缀以 Task 0 实际落点为准（写入 `~/work/yusun/repo/3dgrut/ENV_PATH.txt` 供后续任务 source）；不带 `set -u`。
- **inceptio 铁律**：`export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH`；num_workers=10；渲染/修复前 `docker ps` 查 B1 状态（见 §B1 协调规则）。
- **B1 协调规则**：inceptio GPU 上 B1 训练容器（名含 `b1_arm`）运行时，渲染/修复任务**可以低占用共存**（render-only ~3GB、harmonizer ~6GB，24GB 卡够）但**禁止杀任何 docker 容器/IPC server**；harmonizer_server 若已被 B1 臂 2 拉起则直接复用（server 无状态），否则自起自停。
- **反伪造纪律**：数字入档前 rich log（输出目录内）× metrics.json/eval 输出双源交叉。
- **发射后验证存活**（90s 检查进程 + log 无 Traceback）。
- **ssh 稳定性**：≥5min 任务 inline nohup + `echo PID $!`；嵌套驱动 setsid 四件套；进程查询用 `pgrep -f '[p]attern'`。
- **Mermaid 铁律**：v4_plan.md 看板卡片括号全角（），awk 自查零输出。
- **kill-criterion 登记**：每个蒸馏 run 发射前在 run 说明里登记（run 名 / 3k 步 / 三读数 / 砍单阈值 = 守护线 cc<24.0 即砍 / 砍后动作 = λ 减半重跑）。

## 关键路径速查

| 物件 | 路径 |
|---|---|
| 9ae manifest（inceptio） | `~/work/data/9ae151dc/`（consolidated 变体见同级目录；以 E2.1 用过的为准） |
| 9ae 数据（A800） | `/root/work/yusun/ncore-nurec/data/ncore/clips/` 下（Task 0 确认） |
| 锚 ckpt | 执行期从 [v4_plan.md](../../../v4_plan.md) §5 Done Log E1.1/E1.4/E2.1 条目定位 run 名（inceptio `~/work/output/` 下）；**验证法 = 渲染帧 FID ≈ 锚 124@1m（±15%）** |
| E2.1 批修复脚本 | [`scripts/e21_harmonizer_batch_fix.py`](../../../scripts/e21_harmonizer_batch_fix.py) |
| 离线评估 | [`scripts/eval_frames_dir.py`](../../../scripts/eval_frames_dir.py) |
| harmonizer_server 启动参照 | `inceptio:~/work/nurec_e0/e07/launch_harmonizer_train.sh` 第 2 步（docker run + READY 等待） |
| 三读数锚 | FID@1m 124（baseline 直渲）；lane grad_corr@3m 0.38 口径参照（1m 档 lane 锚执行期首评时记录）；interpolated 守护 cc ≥ 24.7 |
| A800 env 重建脚本 | repo 根 `install_env_uv.sh`（vast 流程验证过） |

---

### Task 0: A800 env 重建 + smoke 验证

**Files:**
- 产出：A800 `/root/work/yusun/repo/3dgrut/.venv/`（uv 流）+ `ENV_PATH.txt`（记录 PATH 前缀供后续任务用）

**Interfaces:**
- Produces: 可用的 A800 python 环境（import threedgrut + JIT 编译通过）+ 9ae 数据路径确认。

- [ ] **Step 1**: 确认 A800 联网（pip index 可达）与磁盘；`install_env_uv.sh` 在 A800 repo 执行（rsync 最新 main 代码先行——沿用 CLAUDE.md 四目录 rsync + grep 验证）。若 uv 流在 A800 撞墙（系统 lib 缺），fallback = conda 重建 `3dgrut2`（miniforge3 现存，`conda create -n 3dgrut2 python=3.11` + repo requirements）。
- [ ] **Step 2**: 验证——import threedgrut、slangc 可见、9ae 数据路径确认（`clips/` 下 9ae 目录 + manifest json）。
- [ ] **Step 3**: 300 步 smoke（9ae multilayer 配方 + `dataset.train.duration_sec=5` 快测窗 + sky_backend=mlp + depth 显式关）确认 JIT 编译全过、训练循环健康；log 写输出目录非 /tmp。
- [ ] **Step 4**: PATH 前缀写 `ENV_PATH.txt`；smoke 输出目录删除。

### Task 1: trainer 伪 GT 注入（TDD，唯一核心代码任务）

**Files:**
- Create: `threedgrut/datasets/distill_frames.py`
- Modify: `threedgrut/trainer.py`（init 建 source + train step 概率替换 + 光度 λ 缩放）
- Modify: config 默认组（跟随 `trainer.pose_adjustment` 等 opt-in 组的既有文件位置模式，加 `distill` 组：`enabled: false / frames_dir: null / p: 0.3 / lam: 0.1 / mode: lateral_1m / region_weight_mask: null` 预留键）
- Modify: `threedgrut/render.py`（`--novel-only` 从硬编码 3m+6m 参数化为逗号档位列表，默认值保持 `lateral_3m,lateral_6m` 向后兼容）
- Test: `threedgrut/tests/test_distill_frames.py`

**Interfaces:**
- Consumes: `novel_view.perturb_c2w(mode, c2w)`（确定性位姿变换）；E2.1 帧包格式（`{mode}/{camera_id}/{idx:06d}.png` + `frames_map.json` 的 `ts:{camera_id}:{timestamp_us}` key）。
- Produces: `DistillFrameSource` 类——`__init__(frames_dir, mode, dataset)`（建 ts 索引、校验帧包）与 `sample() -> Batch`（随机帧 → 反查原 dataset 帧 pose（含 T_to_world_end）→ perturb_c2w → Batch(novel pose, 修复帧 as rgb_gt, mask=None, is_distill 标记)）；trainer 端 `conf.distill.*` 消费该类。

- [ ] **Step 1**: 调查步——train.py 是否已有「从 ckpt warm-start 训练」机制（grep resume / init_checkpoint / load_checkpoint 于 train.py+trainer.py）。有 → 记录用法；无 → 本 task 范围内加最小 warm-start（加载 ckpt 参数、重置 optimizer 与 step 计数，CLI 键 `++init_checkpoint=<path>`），同样 TDD。
- [ ] **Step 2**: 写四项失败测试（spec §2.3）——① `enabled=false` 不建 source、resolved config 与 main 无行为差异；② p=1 + 合成 2 帧 mock 帧包：每步 batch 为伪 GT、光度项 ×λ 断言（λ=2 时光度 loss 恰为 λ=1 的 2 倍）、正则项数值不变；③ pose 重建一致性：同 ts 同 mode 下 `DistillFrameSource` 输出 pose 与渲染侧 `perturb_c2w` 逐元素一致（公差 1e-6）；④ 缺帧 / 空包 / mode 不匹配显式 raise。
- [ ] **Step 3**: 跑测试确认全部 FAIL（未实现）。
- [ ] **Step 4**: 最小实现（distill_frames.py ~80 行 + trainer 接线 ~40 行 + config 组 + render.py 档位参数化 ~20 行）。
- [ ] **Step 5**: 四项测试全绿 + 既有测试全绿（字节等价证明）。
- [ ] **Step 6**: commit `feat(E2.2): trainer 伪 GT 概率混采注入 + novel-only 档位参数化`。

### Task 2: 锚 ckpt 定位 + inceptio 渲染 lateral_1m 帧包

**Files:**
- 产出：`inceptio:~/work/e22/raw_1m/lateral_1m/{camera_id}/*.png` + `frames_map.json`；`~/work/e22/ANCHOR.txt`（锚 ckpt 路径记录）

**Interfaces:**
- Consumes: Task 1 的 `--novel-only lateral_1m`。
- Produces: 375 帧原始渲染包 + 锚 ckpt 确认路径（Task 3/4/5 引用）。

- [ ] **Step 1**: 从 v4_plan.md §5 Done Log（E1.1/E1.4/E2.1 条目）定位锚 ckpt run 名 → inceptio `~/work/output/` 下找到 `ckpt_last.pt`，路径写 ANCHOR.txt。
- [ ] **Step 2**: inceptio 代码同步（`git pull` 到 main HEAD 含 Task 1 commit）+ B1 状态检查（协调规则：容器在跑则低占用共存）。
- [ ] **Step 3**: 发射渲染——render.py `--checkpoint <锚> --path <9ae manifest> --render-only --novel-only lateral_1m --novel-fid`，log 到输出目录；90s 存活验证；~30min 完成。
- [ ] **Step 4**: 验收——375 帧齐（5 相机 × 75）+ frames_map.json 合法 + **锚验证：本次渲染的 FID@1m ≈ 124（±15%）**（金标准：锚 ckpt 找对且链路无漂移）。

### Task 3: inceptio harmonizer 批修复

**Files:**
- 产出：`inceptio:~/work/e22/fixed_1m/lateral_1m/…` 修复帧包（与 raw 同构）

**Interfaces:**
- Consumes: Task 2 的 raw 帧包；harmonizer_server（复用或自起，参照 e07 launch 脚本第 2 步）。
- Produces: 修复帧包（Task 4 的蒸馏输入）。

- [ ] **Step 1**: server 检查——`docker ps` 有 harmonizer_server 则复用；无则按 e07 脚本模式自起（READY 等待）；B1 容器在跑时禁止 rm 任何容器。
- [ ] **Step 2**: `e21_harmonizer_batch_fix.py` 对 raw_1m 跑批修复（nontemporal V=1，~1h）；log 到输出目录。
- [ ] **Step 3**: 验收——修复帧数与 raw 一致 + frames_map 同步 + **抽 3 帧目检**（修复前后并排：伪影减少、无异物、车道线未被抹平的初判）；若自起的 server 则用完停掉。

### Task 4: A800 双臂蒸馏发射（λ sweep）

**Files:**
- 产出：A800 `/root/work/yusun/ncore-nurec/output/e22_1m_lam01/` 与 `e22_1m_lam03/`（各含 ckpt_last.pt + metrics.json + launch.log）

**Interfaces:**
- Consumes: Task 0 env、Task 1 代码（rsync + grep 验证）、Task 2 锚 ckpt、Task 3 修复帧包。
- Produces: 两臂蒸馏后 ckpt（Task 5 输入）。

- [ ] **Step 1**: 传输——修复帧包（~0.5GB）+ 锚 ckpt（~1GB）rsync 至 A800；Task 1 代码 rsync + grep 关键串（`DistillFrameSource`）验证。
- [ ] **Step 2**: kill-criterion 登记（两 run：3k 步 / 三读数 / 砍单 = 训练 loss 爆炸或 2k 步时 TB psnr 崩 >2dB / 砍后 = λ 减半）。
- [ ] **Step 3**: 双卡发射——卡0 `++distill.lam=0.1` / 卡1 `++distill.lam=0.3`，共同：`++distill.enabled=true ++distill.frames_dir=<包> ++distill.p=0.3 ++distill.mode=lateral_1m ++init_checkpoint=<锚>`（键名以 Task 1 实现为准）+ 续训 3k 步 + multilayer 配方 + sky_backend=mlp + depth 显式关 + num_workers=24；log 到各自输出目录；90s 存活验证 + 确认 log 出现伪 GT 采样迹象（Task 1 实现时加一行 sanity 日志）。
- [ ] **Step 4**: ~1.5h 后确认两臂退出 exit 0 + ckpt/metrics 落盘 + **立即 rsync 产物回 inceptio `~/work/backup_a800/e22/`**。

### Task 5: 三读数 eval（两臂 × 三指标）

**Files:**
- 产出：守护线 metrics（A800 标准 eval）+ 档位帧包 eval 结果（inceptio eval_frames_dir 输出 json）+ `~/work/e22/readout_1m.md`（三读数汇总表底稿）

**Interfaces:**
- Consumes: Task 4 两臂 ckpt + Task 2 锚数字。
- Produces: 三读数表（Task 6 入档依据）。

- [ ] **Step 1**: 守护线——A800 对两臂 ckpt 各跑 render.py 标准 eval（interpolated），读 `mean_cc_psnr`；判据 cc ≥ 24.7。
- [ ] **Step 2**: 档位读数——A800 对两臂 ckpt 各渲 lateral_1m 帧包（同 Task 2 命令形状）→ 帧包 rsync 回 inceptio → `eval_frames_dir.py` 评 FID/KID + lane（plane-warp）+ NTA。
- [ ] **Step 3**: 三读数汇总表（两臂 × {FID@1m, lane grad_corr/band_psnr@1m, interpolated cc}）+ 对照列（锚直渲 FID 124 / Task 2 实测锚 lane@1m / cc 24.7 线）写 readout_1m.md；全部数字 rich log × eval 输出双源交叉。
- [ ] **Step 4**: 判定——① FID 落在锚与 E2.1 修复上限之间且改善 → 机制生效；② lane 不塌；③ 守护线不破。三项全过 → 选优臂（FID 与 lane 综合，lane 优先）；部分不过 → 按 spec 风险表旋钮（λ 降 / region 加权评估 / MCMC 温和化）定复跑方案。

### Task 6: 入档 + 文档同步 + 门 1 素材

**Files:**
- Modify: `v4_plan.md`（§1.2 E2.2 状态 ⬜→🟡（第一档 ✅ 注记）、§5 Done Log 新条目、§1.3 gap 表 E2.2 列）
- Modify: `v2_architecture.md`（§6 文件清单加 distill_frames.py、§7 不变量加 `distill.enabled=false` 字节等价行）

- [ ] **Step 1**: Done Log 条目——日期 + commit + 两臂三读数全表 + 选优结论 + 帧包/ckpt 路径 + 口径注记（A800 log 保护纪律执行情况）。
- [ ] **Step 2**: mermaid awk 自查零输出；commit `feat(E2.2): lateral_1m 首档蒸馏三读数入档 + docs 同步`；push。
- [ ] **Step 3**: 门 1 素材登记——E2.2 首档结果 = 门 1 第三数字（自研修复链 off-track 改善证据）；与 B4/B1 数字凑齐后提醒大g 开门 1 决策会话。

### Task 7: 档位推进 runbook（说明性收尾，非执行）

- [ ] **Step 1**: 在 readout_1m.md 末尾附「2m 档复跑指引」——同 Task 2-6 流程，仅换 `--novel-only lateral_2m` / `++distill.mode=lateral_2m` / 起点 ckpt = 1m 选优臂；3m 档读数后触发门 2（战役 spec §7）。**2m 及之后档位的执行等门 1 后大g 指令，不自动推进。**

---

## Self-Review 记录

- **Spec 覆盖**：spec §2.1 循环 → Task 2/3/4/5；§2.2 注入设计 → Task 1；§2.3 四测试 → Task 1 Step 2；§3 编排表 → Task 0-6 + 场地更新（A800 蒸馏、大g 7/7 拍板）；§4 三读数 → Task 5；§5 风险表 → 全局约束（log/回传/kill-criterion）+ Task 5 Step 4 旋钮；§7 文档同步 → Task 6；§8 出界 → Task 7 明示不自动推进。新增覆盖：A800 env 重建（事故响应）= Task 0。
- **Placeholder 扫描**：「键名以 Task 1 实现为准」系 TDD 期定义的显式衔接（Interfaces 已给类名/键名骨架）；锚 ckpt「执行期定位」配了可验证的金标准（FID≈124）。无 TBD。
- **一致性**：`DistillFrameSource` / `distill.*` 键名 / 帧包路径（raw_1m/fixed_1m）/ ANCHOR.txt 贯穿 Task 1-5 一致；`init_checkpoint` 键在 Task 1 Step 1 与 Task 4 Step 3 呼应。
