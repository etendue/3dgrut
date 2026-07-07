# B4 held-out 锚 + B1 双臂 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **格式说明（大g 约定，覆盖 writing-plans 默认）**：本 plan 不贴大段代码块；每任务只写目标 / 文件 / 关键命令意图 / 验收要点。执行期写码（若触发）走 TDD。
>
> **任务形状说明**：本 plan 是**实验执行任务包**（远程 GPU 执行 + 数字入档），非代码 feature。多数任务无新代码，"测试循环"由「执行 → rich log × metrics.json 双源交叉 → 入档」替代。唯一潜在写码点 = 风险 2（`--dataset-cameras` pinhole 撞墙）触发时，先写回归测试再修。

**Goal:** 产出门 1 三个决策数字中的两个——B4 held-out gap（b6a9 真 GT 离轴差距）+ B1 双臂差（官方 baseline vs +Harmonizer 蒸馏），并回填 v5 KPI 表 off-track / NRE gap 两行。

**Architecture:** B4 = 复用 E1.3 `--dataset-cameras` 的 4 个 render-only run（2 ckpt × 2 相机组，纯 CLI 零代码）；B1 = nre-ga 官方 docker 双臂串行（臂 1 baseline 今晚 → 臂 2 +Harmonizer IPC 明晚，复用 e07 IPC 架构），收尾 lateral 3m/6m 两臂 FID 对比。

**Tech Stack:** 3dgrut render.py（main HEAD `a5083c8`+）、nre-ga / nre-tools-ga / harmonizer-cosmos-env docker、inceptio 4090。

**Spec:** [docs/superpowers/specs/2026-07-06-b4-b1-offtrack-anchors-design.md](../specs/2026-07-06-b4-b1-offtrack-anchors-design.md)

## Global Constraints

- **inceptio env 铁律**：我方 python 命令一律 `export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH` 前置；docker 命令不需要。
- **ssh 稳定性**：≥5min 任务不走前台 ssh——单条命令用 inline nohup + `echo PID $!`；嵌套驱动脚本用 setsid 四件套（`ssh -n` + `setsid` + `< /dev/null` + `> log 2>&1`）；ssh 调用之间留间隔防 sshd 限流；查进程用 `ps`/`pgrep`（`[p]ython` 方括号防自匹配），不用 nvidia-smi。
- **发射后必须验证进程/容器存活**（7/3 事故教训：驱动脚本秒死未察觉，浪费一夜）。
- **反伪造纪律（R3）**：一切数字入档前 rich log × metrics.json 交叉验证，两源一致才可写入文档。
- **Monitor 纪律**：只 grep 关键节点（`Traceback|OOM|val/|artifacts|Error|完成 flag`），不 grep 逐帧 PSNR。
- **Mermaid 铁律**：v5_plan.md 看板卡片内括号一律全角（），改完跑 CLAUDE.md 的 awk 自查（应零输出）。
- **文档同步 = 完成定义**：任务完成即更新 v5_plan.md（看板 + Done Log + KPI 表），与代码/产物同 commit 或紧随 commit。
- **口径纪律**：B1 官方 val 口径（每 3 帧 + 1/4 res + cpsnr）与我方口径数字不直接比，入档显式标注；B4 四 run 内部同口径（exposure 均被自动禁用），对照历史锚用 cc 口径。

## 关键路径速查（执行者零上下文备查）

| 物件 | 路径（inceptio） |
|---|---|
| b6a9 manifest | `~/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json` |
| 旧锚 3-cam ckpt | `~/work/output/inc_b6a9_3cam_multilayer_30k/<run_dir>/ckpt_last.pt`（Task 1 确认 run_dir） |
| R0c ckpt | `~/work/output/R0c_3cam_noopreg/<run_dir>/ckpt_last.pt`（Task 1 确认） |
| R1p / R3p 参训 metrics | `~/work/output/R1p_6cam_maskoff_eval/` / `~/work/output/R3p_6cam_interp_eval/` 下 metrics.json |
| e07 IPC 资产 | `~/work/nurec_e0/e07/`（launch_harmonizer_train.sh、ipc/、fixer_run_cmd.json、fixer_run_mounts.json） |
| B1 输出根 | `~/work/nurec_b1/`（本 plan 新建：arm1/ arm2/ lateral/） |
| 4cab runbook 命令 | Mac 仓库 [inceptio_4cabad44_3dgrut_vs_nre.md](../../../inceptio_4cabad44_3dgrut_vs_nre.md) §8 |
| 训练相机组（3 台） | `camera_front_wide_120fov,camera_cross_left_120fov,camera_cross_right_120fov` |
| held-out 相机组（3 台） | `camera_left_wide_90fov,camera_right_wide_90fov,camera_back_rear_wide_90fov` |

---

### Task 1: WP1 准备——inceptio 代码同步 + 路径清单落盘

**Files:**
- 产出：`inceptio:~/work/nurec_b1/paths.env`（本 plan 全程引用的路径变量清单）

**Interfaces:**
- Produces: 确认后的 5 个绝对路径（两 ckpt 完整 run_dir、manifest、R1p/R3p metrics.json），供 Task 2/3/4 直接引用。

- [ ] **Step 1**: inceptio 主仓库同步——`ssh inceptio 'cd ~/repo/3dgrut2 && git fetch origin && git checkout main && git pull'`；验收：`git log --oneline -1` = `a5083c8`（或更新的 main HEAD）。
- [ ] **Step 2**: `find` 确认旧锚与 R0c 的 `ckpt_last.pt` 完整路径、manifest 存在、R1p/R3p `metrics.json` 存在；把 5 个路径写入 `~/work/nurec_b1/paths.env`（shell 变量格式）。
- [ ] **Step 3**: 验收——`ssh inceptio 'source ~/work/nurec_b1/paths.env && ls -l $CKPT_OLD $CKPT_R0C $MANIFEST $R1P_METRICS $R3P_METRICS'` 五项全存在非空。

### Task 2: B4 run 2 smoke——旧锚 × held-out 组（首用 pinhole 验证）

**Files:**
- 产出：`inceptio:~/work/output/b4_old_heldout/`（render 输出 + metrics.json）
- 日志：`inceptio:/tmp/b4_run2.log`

**Interfaces:**
- Consumes: Task 1 的 paths.env。
- Produces: 验证过的命令模板（其余 3 run 复用，仅换 ckpt/相机组/out 名）；run 2 的 metrics.json。

- [ ] **Step 1**: 发射 run 2——单条 ssh inline nohup，命令意图：`python render.py --checkpoint $CKPT_OLD --path $MANIFEST --out-dir ~/work/output/b4_old_heldout --dataset-cameras "<held-out 组 3 台>" --novel-fid`，env PATH 前置，`> /tmp/b4_run2.log 2>&1`，echo PID。
- [ ] **Step 2**: 发射后 30s 存活验证：`pgrep -f '[r]ender.py'` 有进程 + log 无 Traceback；随后 Bash run_in_background 轮询至完成（预期 ~15min）。
- [ ] **Step 3**: **字段验收（smoke 判据）**——metrics.json 必须同时含：`per_camera` 键（恰好 3 个 held-out 相机名，各含 `mean_cc_psnr`/`mean_lpips`/`n_frames`）+ `mean_render_fid`/`mean_render_kid`；log 中出现 exposure 自动禁用告警（确认 cc 口径生效）。
- [ ] **Step 4（条件分支）**: 若字段缺失或 Traceback → 停止批量，转 superpowers:systematic-debugging；若需改码，先写回归测试 pin 住 pinhole × `--dataset-cameras` 失败 case（Mac pytest），修复合入后重跑本 task。

### Task 3: B4 其余 3 run 批量执行（run 1 / 3 / 4）

**Files:**
- Create: `inceptio:~/work/nurec_b1/b4_batch.sh`（串行驱动脚本：run 1 → 3 → 4，每 run 输出目录 `b4_old_train` / `b4_r0c_train` / `b4_r0c_heldout`）
- 日志：`/tmp/b4_batch.log`

**Interfaces:**
- Consumes: Task 2 验证过的命令模板。
- Produces: 三个 metrics.json + 各 run rich log（Task 4 的输入）。

- [ ] **Step 1**: 写 b4_batch.sh——三条 render 命令串行（同模板，换 ckpt×相机组×out 名），每条前 echo 分隔标记 `=== RUN <n> ===`，任一失败即 exit 1。
- [ ] **Step 2**: setsid 四件套发射 + 30s 存活验证（pgrep render.py）。
- [ ] **Step 3**: run_in_background 轮询至脚本退出（预期 ~45min）；验收：log 含三个 `=== RUN` 标记且 exit 0，三个输出目录各有非空 metrics.json。

### Task 4: B4 数字提取 + 双源交叉 + 两个结论

**Files:**
- 产出：`inceptio:~/work/nurec_b1/b4_summary.md`（四组数字表 + 结论草稿，Task 5 入档的底稿）

**Interfaces:**
- Consumes: 4 个 metrics.json + 4 份 rich log + R1p/R3p 参训 metrics.json。
- Produces: 两个结论数字——① held-out−train gap（旧锚侧 + R0c 侧，cc_psnr/lpips/FID 三口径）；② R0c held-out vs R1p 参训同 3 台差值（单变量扩相机收益）。

- [ ] **Step 1**: 从 4 个 metrics.json 抽 per-cam `mean_cc_psnr`/`mean_lpips`/`n_frames` + 全局 `mean_render_fid/kid`，与各自 log 尾部指标表交叉验证（两源一致才采纳；不一致按反伪造纪律停下排查）。
- [ ] **Step 2**: 从 R1p/R3p metrics.json 抽 left_wide/right_wide/back_rear_wide 三台参训 per-cam 数字作对照列。
- [ ] **Step 3**: 算两个结论写入 b4_summary.md：gap 表（run2−run1、run4−run3）+ 扩相机收益表（R1p 参训 − R0c held-out，注明 R1p=R0c+6cam 单变量链）；附口径注记四条（spec §3.3）。

### Task 5: B4 入档 commit

**Files:**
- Modify: `v5_plan.md`（§1.1 看板 B4 卡移 Done、§1.2 B4 状态 ✅、§1.3 Phase B 计数 1/5、§0.2 KPI 表 off-track 行回填、§4 Done Log 新条目含 WP0 IPC 验证结论）

**Interfaces:**
- Consumes: b4_summary.md。

- [ ] **Step 1**: 更新 v5_plan.md 五处；Done Log 条目含：日期、4 组数字、两个结论、口径注记、输出目录路径。
- [ ] **Step 2**: mermaid 全角括号 awk 自查（CLAUDE.md 命令，应零输出）。
- [ ] **Step 3**: commit——`feat(B4): held-out 侧相机真 GT off-track 锚 + docs(plan) 看板/Done Log 同步`。

### Task 6: B1 臂 1 前置检查（三项）

**Files:**
- 产出：检查结论追加到 `~/work/nurec_b1/paths.env` 注释区（lidar_ids 值、depth 依赖结论）

**Interfaces:**
- Produces: `LIDAR_IDS` 变量 + depth aux 决策（走 Task 7 与否）→ Task 9 的发射参数。

- [ ] **Step 1**: **depth aux 依赖**——在 nre-ga 容器内 dump `car2sim_6cam` 配方（docker run --rm 覆盖 entrypoint 查 config 文件，或 `--help`/dry-run），grep depth 相关键判断是否消费 depth aux；结论三选一：不消费（直接过）/ 消费且可 CLI 干净关（记录关闭键，入档标注偏离）/ 消费且需补（触发 Task 7，默认路线）。
- [ ] **Step 2**: **lidar_ids**——python 读 b6a9 manifest 抽 lidar 传感器名列表，选定训练用 lidar（对照 4cab 的 `lidar_top_360fov` 语义 = 顶置 360 主雷达）。
- [ ] **Step 3**: GPU 空闲（`pgrep -f '[p]ython|[t]rain'` 无训练进程）+ 磁盘 ≥100G + WP1 Task 3 已收工确认。

### Task 7:（条件任务）补跑 b6a9 depth aux

**Gate:** 仅当 Task 6 Step 1 结论 =「消费且需补」。

**Files:**
- 产出：depth aux itar 落 clip 目录（与既有 aux 同级）

- [ ] **Step 1**: 照 CLAUDE.md「nre-tools 容器生成 aux」runbook，仅开 `--depth-backend=depthanythingv2`、其余 backend =none、6 相机、`--no-lidar-seg-camvis`；注意 itar write-once 铁律（一次跑完不中断）。预期 ~1h。
- [ ] **Step 2**: 验收：clip 目录出现 `*.aux.depth*.itar` 且非空；原有 sseg/lidar-sseg aux 未被动。

### Task 8: B1 臂 2 launch 资产准备（白天完成）

**Files:**
- Create: `inceptio:~/work/nurec_b1/arm2/b6a9_run_cmd.json`、`b6a9_run_mounts.json`、`arm2_launch.sh`

**Interfaces:**
- Consumes: e07 的 `launch_harmonizer_train.sh`（架构模板）+ `fixer_run_cmd.json`/`fixer_run_mounts.json`（格式样例）+ Task 6 的 LIDAR_IDS。
- Produces: 臂 2 一键发射脚本（Task 11 直接执行）。

- [ ] **Step 1**: 生成 b6a9 版 cmd JSON——基底 = 臂 1 完整命令（Task 9 同源），追加 difix 开关（e07 样例中的确切键名，预期 `difix.training.enabled=true`，以 e07 cmd JSON 实际键为准）；mounts JSON 改挂 b6a9 clip 目录与 `~/work/nurec_b1/arm2` 输出。
- [ ] **Step 2**: 复制改造 arm2_launch.sh——保留 guard（nre-ga 训练容器运行时拒绝执行）/ harmonizer_server 启动 / READY 等待三段，cmd/mounts 指向 Step 1 产物。
- [ ] **Step 3**: 验收：`bash -n arm2_launch.sh` 语法过；两 JSON `python -m json.tool` 解析过；guard 一段与 e07 原版逐行 diff 仅路径差异。

### Task 9: B1 臂 1 发射（gate：今晚 + Task 3/6 完成 + Task 7 若触发已完成）

**Files:**
- 产出：docker 容器 `b1_arm1_baseline`（`-d` 不 `--rm`）+ `~/work/nurec_b1/arm1/` 训练输出

**Interfaces:**
- Consumes: Task 6 的 LIDAR_IDS + depth 决策。
- Produces: 臂 1 训练日志流 + 最终 USDZ artifacts（Task 10/13 输入）。

- [ ] **Step 1**: 组装发射命令——照 4cab runbook §8「NRE 训练」命令，改四处：`dataset.path`=b6a9 manifest、`dataset.lidar_ids=[<Task 6 选定>]`、`out_dir`=/workdir/output 挂 `~/work/nurec_b1/arm1`、`docker run -d --name b1_arm1_baseline`（替换 `--rm`）；保留 `subsample=2` 三键 + ground mesh off + `++trainer.max_steps=40000` + `logger=dummy`。命令全文存 `~/work/nurec_b1/arm1_cmd.sh`（臂 2 基底 + 复现档案）。
- [ ] **Step 2**: 发射 + 60s 存活验证：`docker ps` 容器 Up + `docker logs` 无 Traceback 且出现训练启动标记（step/loss 行）。
- [ ] **Step 3**: 挂 Monitor（关键节点 grep）+ 记录发射时间与预期完成窗（4cab 40k ≈ 75min，b6a9 6 cam 全帧预计数小时）。

### Task 10: B1 臂 1 晨检收数（gate：次日晨 / 容器退出）

**Interfaces:**
- Produces: 臂 1 官方 val 指标 + USDZ 路径清单（b4_summary.md 同款底稿 `b1_summary.md` 开档）。

- [ ] **Step 1**: `docker ps -a` 确认容器 Exited (0)；`docker logs --tail` 抽官方 val 指标终值；输出目录确认 USDZ artifacts + val metrics 文件存在。
- [ ] **Step 2**: 双源交叉（log vs 官方 metrics 文件）后把臂 1 数字写入 `~/work/nurec_b1/b1_summary.md`，显式标注官方口径三要素。
- [ ] **Step 3**: 异常分支：非 0 退出 → systematic-debugging（保留容器现场，先 docker logs 定位；4cab 已知坑先查 subsample/ground-mesh/分辨率整除）。

### Task 11: B1 臂 2 发射（gate：臂 1 容器退出）

- [ ] **Step 1**: 执行 `arm2_launch.sh`（guard 自查通过 → harmonizer_server READY → 训练容器起）。
- [ ] **Step 2**: 双容器存活验证：harmonizer_server Up + READY 日志、训练容器 Up 且 log 出现 difix/IPC 活动迹象（e07 经验：修复请求日志）；挂 Monitor。

### Task 12: B1 臂 2 晨检收数（gate：臂 2 完成）

- [ ] **Step 1**: 同 Task 10 流程收臂 2 官方指标 + USDZ；追加写入 b1_summary.md。
- [ ] **Step 2**: 单变量确认：diff 两臂实际生效 config（容器内 resolved config 或启动命令 diff），唯一差异 = difix 开关，写入 b1_summary.md 作 A/B 有效性证据。

### Task 13: lateral 3m/6m 渲染 + 两臂 FID 对比

**Files:**
- Create: `inceptio:~/work/nurec_b1/lateral/`（两臂 × 两档渲染帧）+ FID 结果追加 b1_summary.md

**Interfaces:**
- Consumes: 两臂 last USDZ + e07 `render_compare.sh`（lateral 渲染流程参照）。
- Produces: 两臂 off-track FID 差（门 1「蒸馏臂增益」输入）。

- [ ] **Step 1**: 照 e07 render_compare.sh 流程对两臂 USDZ 各出 lateral 3m / 6m 帧（`nre render`，同帧集同档位）。
- [ ] **Step 2**: 与 B4 同工具口径算 FID（渲染帧集 vs 训练相机真图集；两臂同参照集）；样本量与口径注记同 B4 纪律。
- [ ] **Step 3**: 两臂 × 两档 FID 表 + 臂差结论写入 b1_summary.md。

### Task 14: B1 入档 + 门 1 素材汇总 commit

**Files:**
- Modify: `v5_plan.md`（§1.1/1.2 看板 B1 → Done、§1.3 计数、§0.2 KPI 表 NRE gap 行回填、§4 Done Log 新条目）

- [ ] **Step 1**: 更新 v5_plan.md；Done Log 条目含两臂官方数字（口径标注）、lateral FID 差、arm1_cmd.sh/launch 资产路径、B4+B1 合并的门 1 素材小结（差 v4 E2.2 首档三读数后即可开门 1 会话）。
- [ ] **Step 2**: mermaid awk 自查零输出。
- [ ] **Step 3**: commit——`feat(B1): NRE 同 clip 双臂对照锚 + docs(plan) 同步`；push 分支。

---

## Self-Review 记录

- **Spec 覆盖**：spec §0（事实）→ 速查表；§2 编排 → 任务 gate 标注；§3 B4 四 run → Task 2-5；§4.1 臂 1 三前置 → Task 6/7/9；§4.2 臂 2 → Task 8/11/12；§4.3 收尾 → Task 13；§4.4/§7 入档 → Task 5/14；§5 风险 → Task 2 Step 4（风险2）、Task 6/7（风险1）、Task 9-10（风险3/4）、Task 4 Step 1（风险5）；§6 测试策略 → Task 2 Step 4 TDD 分支。无缺口。
- **Placeholder 扫描**：`<run_dir>` 两处为 Task 1 的显式产出（非 TBD）；difix 确切键名标注了「以 e07 cmd JSON 实际键为准」的解析来源（非悬空）。
- **一致性**：相机组名、输出目录名（b4_old_heldout 等四个）、b1_summary.md 贯穿引用一致。
