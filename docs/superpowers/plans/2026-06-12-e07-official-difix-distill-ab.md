# E0.7 官方 difix-distill 对照 run — 执行 Plan（α/β 两段）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans（inline 顺序执行）。本 plan 是远端运维/实验链路——任务间强串行依赖、全部状态在 inceptio 上，**不适合** subagent-driven 并行。Steps 用 checkbox（`- [ ]`）跟踪。

**Goal:** 在同 clip 9ae151dc 上用官方 nre 容器把 difix train-time 蒸馏打开（单 key `difix.training.enabled=true`）重训 40k，与 E0.3 锚（difix OFF，test/psnr 30.30）单变量对照，得到「官方修复器蒸馏增益上限锚」，直接校准 E2.2（渐进外推蒸馏移植）的预期收益。

**Architecture:** 纯容器运维 + 评测对比 + 文档回填，**不改 3dgrut2 任何训练代码**。α/β 两段（2026-06-12 大g拍板）：

- **α 段**（硬 gate 仅「权重可得」）：权重 spike → 300 步 smoke → 40k 全量 → C1 interpolated 对比 + C2-FID（gt/lat3m/lat6m）+ C3 目视。完成后 E0.7 状态停 **🔵**。
- **β 段**（gate＝E1.1/E1.2 merged）：lane grad_corr/band_lpips@3m/6m + NTA-IoU 三档，对已落盘的两个 USDZ **纯渲染回填，不重训**。回填后才标 ✅（守 CLAUDE.md「metrics 没齐不许 ✅」）。

**Tech Stack:** nre-ga 容器（26.4.146-c63f08a4，pin IMAGE_ID）、inceptio RTX 4090 24GB、NGC CLI（inceptio 已 login）、HF 候选权重（`nvidia/Fixer` / `nvidia/Difix3D`）、E0.2 已验证的渲染三档 + FID 流程。

**执行机器约定:** 全部经 Mac `ssh inceptio` 执行（[CLAUDE.md](../../../CLAUDE.md)：长任务 inline nohup + `echo PID \$!`；Monitor 只 grep 关键节点）。本任务工作目录统一 `~/work/nurec_e0/e07/`（log / csv / 探针脚本 / 决策记录全放这，**不入 git 仓库**——沿 E0.2「临时脚本不入项目代码」先例）。

---

## 0. 背景锚（已核实事实，执行时不需重查）

| 项 | 值 | 来源 |
|---|---|---|
| E0.3 对照锚（difix OFF） | test/psnr **30.30** / cpsnr road 38.27 · car 34.59 · person 32.65 · sky 38.81 / chamfer 0.295（官方口径：val 每 3 帧 + 1/4 分辨率 + cpsnr） | [v4_plan.md](../../../v4_plan.md) §5 Done Log |
| E0.3 run 产物 | inceptio `~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/`（`artifacts/last.usdz` 1.1GB + metrics.yaml + parsed.yaml 8438 行） | 同上 |
| E0.3 配方 | Hyperion-8.1 `car2sim_6cam` + `references/configs/pai.yaml` overlay（`--config-name=external_overrides`）；数据 `~/work/data/9ae151dc_consolidated/`（13 itar，自产 lane aux 必须移出） | 同上 |
| E0.3 工程数 | 40k 步 2h07m（7.45 it/s）/ 峰值 16GB / 2.62M gaussians | 同上 |
| difix 蒸馏钩子参数（**已固化在 E0.3 parsed.yaml，唯 enabled=false**） | `difix.name=cosmos-difix`；start_step=20000；p_scheduler p_init=0.5, milestones=[25000,28000], gamma=0.5；use_color_transfer=true；novel_view_poses.translation=[0,±3.0,0]；shuffle_novel_views=false；分辨率 576×1024 | [E0.5 spec §10](../specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md) |
| 官方权重 | `difix.model_url=nurec-fixer/cosmos_3dgut.pt`，NGC URL `https://api.ngc.nvidia.com/v2/org/nvidia/team/nre/models/nurec-fixer/versions/cosmos_3dgut/files/cosmos_3dgut.pt`；**容器内无此文件、运行时下载；HF 无副本；普通 key 疑似权限不足** | E0.5 调查 + R-v4.10 |
| 容器内 difix 配置 | `/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/configs/difix/cosmos_difix.yaml`；另有 legacy 变体 `difix=sd_difix`（Stable-Diffusion 架构） | E0.5 调查 + nre SKILL.md |
| cache 约定 | `~/.cache/nre`（nre SKILL.md Teardown 节背书；**确切子路径 Task 0 实证**） | nre SKILL.md |
| ckpt 默认 | `checkpoint.every_n_train_steps: 40000`（挂在 35k＝全白烧）→ 全量 run 覆盖为 5000（纯 I/O key，不破坏 A/B）；resume 按不可用设计（v3 R7 前科） | E0.5 spec §9 |
| Hydra 覆盖 | `difix.training.enabled` 等 key 已存在于 resolved config → 直接 `key=value` 覆盖，无需 `+/++` | CLAUDE.md Hydra 节 |

**开源候选权重（hack，R-v4.10）**：

1. HF [`nvidia/Fixer`](https://huggingface.co/nvidia/Fixer) — Cosmos-Predict2-0.6B，与 cosmos_3dgut 同架构族；本仓库 [`threedgrut/correction/difix.py`](../../../threedgrut/correction/difix.py) 一代集成同源（`pretrained_fixer.pkl`，注意另有 tokenizer 两件套 `model_fast_tokenizer.pt`/`tokenizer_fast.pth`，而 cosmos_3dgut.pt 可能是 all-in-one 单文件）。
2. HF [`nvidia/Difix3D`](https://huggingface.co/nvidia/Difix3D) — SD 架构，nre SKILL.md 认定为 `sd_difix` 的 HF 对应物。
3. HF `nvidia/Harmonizer`（inceptio `~/repo/harmonizer` 已下载）— 时间条件模型，IO 契约最远，**默认不试**（除非探针显示意外兼容）。

**权重来源等级（口径标注机制，贯穿全文档与增益表列头）**：

> **A** ＝ 官方 `cosmos_3dgut.pt`（NGC）— 口径完全等价，可直接当 E2.2 上限锚
> **B** ＝ HF `nvidia/Fixer`（同 Cosmos-Predict2-0.6B 架构族、post-train 数据不同）— 增益方向可信、幅度打折号
> **C** ＝ `difix=sd_difix` + HF `nvidia/Difix3D`（一代 SD 架构）— 只能当「蒸馏机制有效性」下界锚

**预期管理（R-v4.5 同款教训，写在最前防误判）**：**interpolated 不涨 ≠ 失败**——蒸馏优化的是 novel 分布（DiFix3D+ 的几何增益 +1.03 也是外推视角上量的）；增益主战场＝C2 的 3m/6m。**Δ@3m（增强分布内，官方钩子恰好 ±3m）vs Δ@6m（增强分布外）的比值＝E2.2「渐进推到 6m」还能指望多少的直接读数**——这是本实验对 E2.2 校准的最大信息量。

**单变量 A/B 纪律（本 plan 的科学性核心）**：E0.7 run 相对 E0.3 **恰好三处差**——① `difix.training.enabled=true`（被测变量）② `checkpoint.every_n_train_steps=5000`（纯 I/O 保险）③ cache mount（权重供给）。`sqa_difix_distill.yaml` 整套配方**不采用**（可能携带其他配套改动破坏单变量），仅作 Task 0 的文档性对照确认。

---

## Task 0 — 容器与配方调查（只读，零 GPU，~1h）

**Files:** 产出（均在 inceptio）：`~/work/nurec_e0/e07/IMAGE_ID`（镜像 pin）、`~/work/nurec_e0/e07/E03_command.sh`（恢复的 E0.3 精确命令）、`~/work/nurec_e0/e07/task0_notes.md`（key-map + loader 四事实 + sqa 差异清单）

- [ ] **Step 1: pin 镜像 + 验 E0.3 资产仍在（风险 N1/N7）**

```bash
ssh inceptio 'mkdir -p ~/work/nurec_e0/e07 && docker inspect nvcr.io/nvidia/nre/nre-ga:latest --format "{{.Id}} {{.Created}}" | tee ~/work/nurec_e0/e07/IMAGE_ID'
ssh inceptio 'ls ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/artifacts/ && grep -rn -m5 "psnr" ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/ --include="*.yaml" | head'
```

Expected: Created≈2026-05-28（=26.4.146-c63f08a4）；`last.usdz` 在；test/psnr 30.30 找得到。
**后续所有 docker run 一律用 `$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID)`，不用 `:latest`。**
FAIL 分支：镜像 Created 漂移 → 查 E0.3 run 日志头/parsed.yaml 内 version 字段找回当时 ID，找不回则记录漂移入档；E0.3 产物没了 → E0.7 升级前置「重跑 E0.3」（+2h）后再继续。

- [ ] **Step 2: 恢复 E0.3 精确启动命令 → `E03_command.sh`**

```bash
ssh inceptio 'ls ~/work/nurec_e0/*.sh 2>/dev/null; find ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U -maxdepth 2 \( -name "*overrides*" -o -name "*.log" -o -name "parsed.yaml" \) 2>/dev/null'
ssh inceptio 'grep -n "docker run" ~/.bash_history 2>/dev/null | grep -iE "nre|train" | tail -20'
```

把恢复出的完整 docker run（含 `--gpus` / shm / 数据与输出 mount / `--config-name=external_overrides` / pai overlay 路径 / logger 设置）写入 `~/work/nurec_e0/e07/E03_command.sh`。拿不到完整原件 → 按 parsed.yaml + nre SKILL.md 重构，文件头注明 `# RECONSTRUCTED`（入档时声明）。

- [ ] **Step 3: key-map——从 parsed.yaml 钉死后续覆盖 key 的精确路径**

```bash
ssh inceptio 'P=$(find ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U -name "parsed.yaml" | head -1); echo $P; \
  grep -nE "max_steps|n_steps|total_steps|n_iterations" $P | head; \
  grep -nE "every_n_train_steps" $P; \
  grep -n -A30 "^difix:" $P | head -50; \
  grep -nE "model_url" $P'
```

Expected 四个确认值写进 `task0_notes.md`：① 训练步数 key（值 40000 者）；② ckpt 间隔 key 确切路径；③ `difix.training.enabled` 与钩子参数（start_step / p_scheduler.* / use_color_transfer / novel_view_poses）确切层级；④ `difix.model_url` 值。**Task 2/3 的覆盖串以此为准**（本 plan 后文写的 `difix.training.start_step` 等层级来自 E0.5 报告，如有出入以 key-map 为准）。

- [ ] **Step 4: 定位 + cat `sqa_difix_distill` 原始 yaml（文档性确认，E0.7 不采用）**

```bash
ssh inceptio 'IMG=$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID); docker run --rm --entrypoint bash $IMG -c "find /app -iname \"*sqa*\" 2>/dev/null | head -20"'
ssh inceptio 'IMG=$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID); docker run --rm --entrypoint bash $IMG -c "cat <上一步找到的 sqa_difix_distill.yaml 路径>"'
```

产出 ≤30 行「官方开蒸馏时配套动了哪些 key」清单 → 记 task0_notes.md（供 E2.2 设计参考）。同时回答 Task 2 smoke FAIL 时的「enable 机制」问题：官方是单 key 开还是 defaults 组切换（如 `difix=cosmos_difix_distill`）/ callbacks 注册。

- [ ] **Step 5: difix loader 机制四事实（决定 Task 1 全部机制）**

```bash
ssh inceptio 'IMG=$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID); docker run --rm --entrypoint bash $IMG -c "ls /app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/configs/difix/ && cat /app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/configs/difix/*.yaml"'
ssh inceptio 'IMG=$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID); docker run --rm --entrypoint bash $IMG -c "grep -rln \"model_url\" /app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles/_main/ --include=*.py | head"'
ssh inceptio 'IMG=$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID); docker run --rm --entrypoint bash $IMG -c "grep -n -B3 -A12 -iE \"model_url|cache|download|api_key|ngc\" <上一步定位的 loader.py>"'
```

必须拿到并写进 task0_notes.md：
(a) **cache 确切路径 + 期望文件名**（Task 1 预放点与 docker `-v` mount 目标）；
(b) 下载是否读 `NGC_API_KEY` env；
(c) **difix 模型构造时机**（trainer init 即建 vs start_step 懒加载——决定失败暴露在第 0 分钟还是第 75 分钟）；
(d) `sd_difix` 的 model_url 指向（NGC / HF / 公网——决定 C 级路径成本）。

- [ ] **Step 6: resume / ckpt 事实（查归查，设计上不依赖）**

```bash
ssh inceptio 'P=$(find ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U -name "parsed.yaml" | head -1); grep -inE "resume|ckpt_path" $P | head'
```

---

## Task 1 — 权重获取（官方一次 → hack 决策树；timebox 3h，全不通 → ⏸ 收尾）

**Files:** 产出：cache 预放权重文件（路径按 Task 0 Step 5(a)）+ `~/work/nurec_e0/e07/weight_decision.md`（sha256 / 来源 / 等级 / 探针输出摘录）

- [ ] **Step 1: 先试官方 NGC 下载（大g已在 inceptio `ngc login`，预期权限不足但试一次仅几分钟）**

```bash
ssh inceptio 'which ngc && ngc config current 2>&1 | head -5'
ssh inceptio 'cd ~/work/nurec_e0/e07 && ngc registry model download-version "nvidia/nre/nurec-fixer:cosmos_3dgut" 2>&1 | tail -20'
```

成功 → `sha256sum` 记录 → **等级 A** → 直接到 Step 4 预放。
失败 → 把错误原样记 weight_decision.md（**403 vs 404 有诊断价值**：403＝key 权限不足；404＝版本名/路径变了，回 Task 0 Step 5 重查 model_url）→ Step 2。

- [ ] **Step 2: 盘点本地存量权重（HF 下载又慢又要代理，先查存量）**

```bash
ssh inceptio 'find ~ -maxdepth 5 \( -name "pretrained_fixer.pkl" -o -name "*cosmos_3dgut*" -o -iname "*difix3d*" \) 2>/dev/null; ls ~/repo/harmonizer/ 2>/dev/null | head'
```

- [ ] **Step 3: hack 决策树（顺序可被 Task 0 Step 5(d) 实证调整）**

**B 级首选**：HF `nvidia/Fixer`。本地有 `pretrained_fixer.pkl` 直接用；否则下载（先确认 mihomo 已起，启动命令见 [CLAUDE.md](../../../CLAUDE.md)「访问外网」节）：

```bash
ssh inceptio 'export http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 HF_HUB_DISABLE_XET=1 \
  && huggingface-cli download nvidia/Fixer --local-dir ~/work/nurec_e0/e07/hf_fixer 2>&1 | tail -5'
```

**C 级备选**：`difix=sd_difix` + HF `nvidia/Difix3D` 同法下载到 `~/work/nurec_e0/e07/hf_difix3d`。
Harmonizer pkl 默认不试。
（`HF_HUB_DISABLE_XET=1` + 代理＝Done Log 环境铁律，xet/直连均会卡死。）

- [ ] **Step 4: state_dict 探针（训练前，最短诊断路径——不要用训练启动当诊断器）**

预放（cache 路径/文件名按 Task 0 Step 5(a) 实证值）：

```bash
ssh inceptio 'mkdir -p ~/.cache/nre/<实证子路径> && cp <候选权重文件> ~/.cache/nre/<实证子路径>/<期望文件名> && sha256sum ~/.cache/nre/<实证子路径>/<期望文件名>'
```

容器内探针（先 scp 脚本再跑，避免 ssh 内 heredoc 引号地狱——CLAUDE.md vast 节同款经验）：

```bash
cat > /tmp/probe_sd.py <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
ks = list(sd.keys()) if hasattr(sd, "keys") else []
print(type(sd), len(ks))
print("first15:", ks[:15])
print("last5:", ks[-5:])
PY
scp -q /tmp/probe_sd.py inceptio:~/work/nurec_e0/e07/tools/probe_sd.py
ssh inceptio 'IMG=$(cut -d" " -f1 ~/work/nurec_e0/e07/IMAGE_ID); docker run --rm -v ~/.cache/nre:/root/.cache/nre -v ~/work/nurec_e0/e07/tools:/tools --entrypoint python $IMG /tools/probe_sd.py /root/.cache/nre/<实证子路径>/<期望文件名>'
```

对照 Task 0 Step 5 grep 出的 loader 构造代码，**判定矩阵**：
- 纯前缀错位 → 写一次性 remap 脚本（`~/work/nurec_e0/e07/tools/remap_sd.py`，ops 产物不入仓库）产出转换后权重；
- 模块/shape 缺失 → 弃该候选，换决策树下一个；
- 全部候选不通 → **E0.7 ⏸**（见「失败出口」）。

- [ ] **Step 5: 决策记录**

`weight_decision.md` 写明：最终选用权重文件 sha256 + 来源 URL + **等级（A/B/C）** + 各候选探针输出摘录。
key 安全（风险 N6）：若用了 NGC key，只存 `~/.ngc_key`（chmod 600）；任何文档只记「来源：大g / 日期」，**不记 key 值**，命令行不回显。

---

## Task 2 — 短 smoke（300 步强迫 difix 立刻 fire，~15 min GPU）

**Files:** `~/work/nurec_e0/e07/launch_smoke.sh` + `smoke.log` + `vram_smoke.csv`；输出目录 `~/work/nurec_e0/train_out_e07_smoke/`

- [ ] **Step 1: 写 launcher（= E03_command.sh + 覆盖；key 层级以 Task 0 Step 3 key-map 为准）**

在 `E03_command.sh` 的 docker run 基础上追加/修改：

```bash
# launch_smoke.sh 相对 E03_command.sh 的差异（key 确切层级以 key-map 为准）：
#   -v ~/.cache/nre:/root/.cache/nre \          ← cache mount（子路径按 Task 0 实证）
#   <训练步数key>=300 \
#   difix.training.enabled=true \
#   difix.training.start_step=50 \
#   difix.training.p_scheduler.p_init=1.0 \     ← 强迫 step 50 起每步都 fire
#   <out_dir 指到 ~/work/nurec_e0/train_out_e07_smoke>
```

- [ ] **Step 2: 启动 + VRAM watcher（proven inline nohup 模式）**

```bash
ssh inceptio "nohup bash ~/work/nurec_e0/e07/launch_smoke.sh > ~/work/nurec_e0/e07/smoke.log 2>&1 & echo PID \$!"
ssh inceptio "nohup bash -c 'while true; do echo \$(date +%T),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) >> ~/work/nurec_e0/e07/vram_smoke.csv; sleep 2; done' >/dev/null 2>&1 & echo WPID \$!"
```

- [ ] **Step 3: PASS 判据（四条缺一不可，防伪完成）**

```bash
ssh inceptio 'grep -icE "difix" ~/work/nurec_e0/e07/smoke.log; grep -iE "difix" ~/work/nurec_e0/e07/smoke.log | head -20'
ssh inceptio 'grep -iE "traceback|error|missing key|unexpected key|size mismatch|download" ~/work/nurec_e0/e07/smoke.log | head'
ssh inceptio 'sort -t, -k2 -n ~/work/nurec_e0/e07/vram_smoke.csv | tail -3'
```

① 日志出现 difix 加载/触发证据行（**确切措辞回填进本 plan 此处，全量 run 复用同一 grep**）；
② 无 Traceback / missing key / unexpected key / size mismatch / download 错误；
③ VRAM 在 step 50 后台阶式跳升（0.6B 模型驻留证据）；
④ 测得 difix-on 段 it/s（推全量 ETA）。
**exit 0 但无 ① ＝ FAIL** → 最可能 enable 机制不止一个开关（「key 存在 ≠ 模块实例化」，E0.5 §0 警告）→ 回 Task 0 Step 4 的 sqa yaml 看官方怎么开（defaults 组切换 / callbacks 注册），修正 launcher 后重跑 smoke。

- [ ] **Step 4: 收尾 + 边界声明**

杀 watcher（`kill <WPID>`）。**边界声明**：smoke 在 ~step 50 触发时只有 init 高斯 + 低阶 SH，**不验证 24GB 显存头寸**（真实峰值在 20k 后：16GB 基线 + 2.62M 高斯 + SH3 + difix 并发驻留同时发生）——头寸验证移到 Task 3 检查窗。

---

## Task 3 — 全量 40k（~3.5h ≈ 2h07m 基线 + difix 调用 5k×0.5+3k×0.25+12k×0.125≈4750 次 ≈ +1~1.5h）

**Files:** `~/work/nurec_e0/e07/launch_full.sh` + `full.log` + `vram_full.csv`；输出 `~/work/nurec_e0/train_out_e07/<RUNID>/`

- [ ] **Step 1: launch 前清单（CLAUDE.md A800 把关清单同款纪律）**

```bash
ssh inceptio 'nvidia-smi | tail -15; docker ps'                                  # GPU 空闲、无残留 viewer/serve-grpc
ssh inceptio 'ls ~/work/data/9ae151dc_consolidated/ | sort'                      # 13 itar 与 E0.3 一致，lane aux 不在
ssh inceptio 'cat ~/work/nurec_e0/e07/IMAGE_ID'                                  # 与 E0.3 同镜像
```

smoke PASS 四条证据已贴 task0_notes.md。

- [ ] **Step 2: launch_full.sh ＝ E03_command.sh + 恰好三处差，启动**

```bash
# launch_full.sh 相对 E03_command.sh 的差异（恰好三处 + 环境变量）：
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \   ← 环境变量，非配方
#   -v ~/.cache/nre:/root/.cache/nre \                   ← ② cache mount
#   difix.training.enabled=true \                        ← ① 被测变量
#   <ckpt间隔key>=5000 \                                 ← ③ 纯 I/O 保险
#   <out_dir 指到 ~/work/nurec_e0/train_out_e07>
ssh inceptio "nohup bash ~/work/nurec_e0/e07/launch_full.sh > ~/work/nurec_e0/e07/full.log 2>&1 & echo PID \$!"
ssh inceptio "nohup bash -c 'while true; do echo \$(date +%T),\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) >> ~/work/nurec_e0/e07/vram_full.csv; sleep 10; done' >/dev/null 2>&1 & echo WPID \$!"
```

Monitor 只 grep `Traceback|OOM|out of memory|Test Metrics|🎊`——**不要 grep difix**（~4.7k 行触发 Monitor rate-limit，CLAUDE.md 同类教训）。

- [ ] **Step 3: T+75min（≈step 20k）人工检查窗**

检查：difix 首次 fire 证据（用 smoke 回填的 grep 串）+ VRAM 新峰值 + it/s 落差 + train loss/psnr 未崩。
**abort 判据**（立即杀省 1.5h）：OOM；或 loss/psnr 相对 E0.3 同步数明显恶化（hack 权重产出垃圾监督的早期信号，风险 N4）。
**OOM 合法缓解序（风险 N2）**：expandable_segments（已带）→ 官方自带 `collect_garbage_mem_usage` 调低 → 同配方 48GB 卡重跑 ON 侧（`vast-train` skill；硬件不是配方变量）。**禁用**「降 difix 频率 / 改 16-mixed」——前者改变被测处理（测的就不再是官方增益上限），后者破坏与 E0.3 fp32 的可比性。

- [ ] **Step 4: 完成验证 + 单变量审计（本 Task 核心证据）**

```bash
ssh inceptio 'PA=$(find ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U -name parsed.yaml | head -1); \
  PB=$(find ~/work/nurec_e0/train_out_e07 -name parsed.yaml | head -1); \
  diff $PA $PB | grep -vE "run_id|timestamp|out_dir|date" | head -40'
ssh inceptio 'grep -c -iE "difix" ~/work/nurec_e0/e07/full.log'
ssh inceptio 'find ~/work/nurec_e0/train_out_e07 \( -name "*.usdz" -o -name "metrics.yaml" \)'
```

判据：① diff 仅剩 `difix.training.enabled` + ckpt 间隔两处差（**diff 输出直接贴 Done Log，单变量 A/B 铁证**）；② difix fire 总数与 ~4.7k 量级吻合（数量级偏离＝p_scheduler 没按预期走，查 log 定位）；③ metrics.yaml + last.usdz 在。

---

## Task 4 — 评测对比（C1/C2-FID/C3＝α；C2-lane/NTA＝β 占位）

**Files:** 渲染输出 `~/work/nurec_e0/renders/e07_compare/{off,on}_{gt,lat3m,lat6m}/`；真帧 `~/work/nurec_e0/real_frames/9ae151dc/`；目视存档 `~/work/nurec_e0/renders/e07_compare/visual/`

- [ ] **C1（α）: 双 metrics.yaml 直读填增益表**

```bash
ssh inceptio 'MA=$(find ~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U -name metrics.yaml | head -1); \
  MB=$(find ~/work/nurec_e0/train_out_e07 -name metrics.yaml | head -1); \
  grep -E "psnr|chamfer" $MA | head -30; echo ===; grep -E "psnr|chamfer" $MB | head -30'
```

官方口径天然同协议（每 3 帧 + 1/4 res + cpsnr），零新工具。填表：test/psnr、cpsnr road/car/person/sky、chamfer。

- [ ] **C2-FID（α）: 双 USDZ 渲三档 + FID/KID vs 自有 clip 真帧**

1. 找回 E0.2 渲染命令形态（`~/work/nurec_e0/renders/0fd06bc3/` 的生成方式 / shell history / E0 工作分支 commit `8b2fcbe` 附近的脚本），对**两个 USDZ**（OFF＝E0.3 `last.usdz`、ON＝E0.7 `last.usdz`）各渲 gt / lat3m / lat6m（rig offset 法，同轨迹同帧集同相机）。**自训 USDZ 必须 `--no-enable-nrend` 走 torch 路径**（nrend 对自训 USDZ 间歇性 `NRenderer.render failed`，Done Log 已记）。
2. 自有 clip 真实参考帧（E0.2 的 real_frames 是官方场景的，本 clip 需新备）：优先查训练 run val 目录是否带 GT dump；否则写一次性解码脚本 `~/work/nurec_e0/e07/tools/dump_gt_frames.py`（容器内 python 用官方 ncore 库读 itar，前向相机全帧）→ `~/work/nurec_e0/real_frames/9ae151dc/`。
3. FID/KID：优先复用 E0.2 已验证流程；需重建则在 harmonizer-cosmos-env 容器内用 cleanfid（**帧数 <500 用 KID**，E1.4 既定；缺包则带代理 pip 装 clean-fid）：

```python
from cleanfid import fid
real = "/d/real_frames/9ae151dc"
for tag in ["off_gt", "off_lat3m", "off_lat6m", "on_gt", "on_lat3m", "on_lat6m"]:
    print(tag, fid.compute_kid(real, f"/d/renders/e07_compare/{tag}"))
```

4. **两侧渲染都必须 raw（不过 Harmonizer）**——E0.7 测的是 train-time 蒸馏本身，后处理会混杂变量。

- [ ] **C2-lane / NTA-IoU（β）`[gate: E1.1/E1.2 merged]`**

占位：E1.1（lane grad_corr / band_lpips @3m/6m）+ E1.2（NTA-IoU @gt/3m/6m）工具 merge 后，对已落盘的两个 USDZ 渲帧喂项目侧 eval（E0.4 同法：nre render 出帧 → 项目指标代码，保口径一致），填增益表 β 行。**不重训。**

- [ ] **C3 目视（α）: viewer 对比 + 大g终评**

viewer（host patch `~/work/nurec_e0/patches/av_patched.py` 挂载 + `--no-enable-nrend`）分别开两个 USDZ，Camera Translation Offset 0 / 3 / 6m 各截图入 `renders/e07_compare/visual/`。重点：车道线连续性、路侧结构涂抹、悬浮伪影。请大g目视终评（沿 E0.3 先例）；FID 须与目视同向佐证（R-v4.5）。

- [ ] **（可选，默认 skip）20k ckpt 等价性抽查**

difix 20k 才启动 → 两 run 的 20k ckpt 理论等价（同 seed 同配方），存疑时各跑一次快速 val 互证单变量成立。

---

## Task 5 — 文档回填（α 后一次、β 后一次）

- [ ] **α 回填**：[v4_plan.md](../../../v4_plan.md) §1.2 E0.7 行（实测数 + 权重等级，状态 → 🔵）、§1.3 gap 表「官方蒸馏增益」行、§5 Done Log（增益表全文 + parsed.yaml diff 摘录 + 权重 sha/来源/等级 + fire 次数 + 时长/峰值显存）、§3 R-v4.10 更新（哪条权重路径通了/没通）；[E0.5 spec](../specs/2026-06-11-e05-nurec-vs-multilayer-recipe-diff.md) §10 加一行 cross-ref 指向本 plan 执行结果。
- [ ] mermaid 全角括号自查（`awk '/```mermaid/{i=1;next} /```/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v4_plan.md` 零输出）+ commit（`docs(plan):` 约定 + Co-Authored-By）。
- [ ] **β 回填**（E1.1/E1.2 后）：增益表 lane/NTA 行 + E0.7 → ✅ + Done Log 补条。

---

## 风险表

| ID | 风险 | 缓解 |
|---|---|---|
| N1 | `:latest` 镜像漂移成隐形变量 | Task 0 pin IMAGE_ID 全程用 ID；parsed.yaml diff 兜底捕获 |
| N2 | OOM 缓解手段污染处理变量 | **禁**「降 difix 频率 / 16-mixed」；合法序＝expandable_segments → collect_garbage → 48GB 卡重跑 ON 侧 |
| N3 | 容器内联网不可达（mihomo 在 host 127.0.0.1 回环，容器内不可达） | 一律 host 预下载 + cache mount，容器零联网需求 |
| N4 | hack 权重「加载成功但产出垃圾监督」 | Task 3 的 20k 检查窗 abort 判据；interpolated 守护；「该权重不可用」本身是有效结论入档 |
| N5 | key/环境窗口时效 | α/β 拆分（已拍板）：权重到手立刻跑训练，eval 永远可补 |
| N6 | NGC key 泄漏进日志/文档 | key 只存 `~/.ngc_key`（600）；文档只记「来源：大g/日期」不记值 |
| N7 | E0.3 对照锚产物被清 | Task 0 Step 1 先验；缺失则前置重跑 E0.3（+2h） |
| — | 显存逼近 24GB（R-v4.2 关联） | E0.3 峰值 16GB + 0.6B difix 驻留 + novel 渲染缓冲；见 N2 合法序 |

---

## 增益表模板（Task 4/5 填；元数据强制项缺一不可入档）

**元数据**：OFF run id（PVG7YYV72YKPLumogi7F7U）/ ON run id / IMAGE_ID（26.4.146-c63f08a4）/ 数据目录（9ae151dc_consolidated，13 itar）/ 蒸馏参数摘要（start 20000 · p 0.5→[25k,28k]×0.5 · color_transfer · ±3m · 576×1024）/ parsed.yaml diff 结论 / **权重等级写进 ON 列头**（表被复制引用时 caveat 不丢）。

| 区块 | 指标 | 档位 | OFF（E0.3） | ON（E0.7，权重级=?） | Δ | 工具 | 时点 |
|---|---|---|---|---|---|---|---|
| interpolated（守护线） | test/psnr | val 官方口径 | 30.30 | | | metrics.yaml | α |
| | cpsnr road / car / person / sky | 同上 | 38.27 / 34.59 / 32.65 / 38.81 | | | metrics.yaml | α |
| | chamfer | 同上 | 0.295 | | | metrics.yaml | α |
| 外推-感知 | FID 或 KID（<500 帧）vs 自有 clip 真帧 | gt / lat3m / lat6m | | | | E0.2 流程/cleanfid | α |
| 外推-lane | grad_corr / band_lpips | lat3m / lat6m | 待 E1.1 | 待 E1.1 | | E1.1 | β |
| 外推-actor | NTA-IoU | gt / lat3m / lat6m | 待 E1.2 | 待 E1.2 | | E1.2 | β |
| 运行工程 | 时长 / 峰值显存 / it/s / n_gaussians / difix fire 次数 | — | 2h07m / 16GB / 7.45 / 2.62M / 0 | | | log + watcher | α |

附注记（随表入档）：
1. **Δ@3m vs Δ@6m 比值**＝E2.2 渐进蒸馏「推到 6m 还有多少肉」的直接读数（±3m 恰是官方增强分布，3m＝同分布增益、6m＝泛化增益）。
2. interpolated 是守护线非收益轴（不涨 ≠ 失败）；FID 行须配 C3 目视同向佐证（R-v4.5 双协议）。

---

## 失败出口

Task 1 全部候选不通（官方 + B + C 都失败）→ **E0.7 ⏸**：`weight_decision.md` 的逐候选失败证据入 v4_plan §5 Done Log（「cosmos-difix 权重不可得」本身是有效结论），R-v4.10 更新为「已尝试、不可得」，E2.2 预期收益校准改用 E2.1 spike 实测。**不阻塞 v4 主线。**
