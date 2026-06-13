# E0.7 移交方案：用 DiffusionHarmonizer 替换 Fixer 跑官方蒸馏训练（A/B）

> **移交文档**：本方案自包含，面向无上下文的执行 session。所有路径为 inceptio（`ssh inceptio`）绝对路径。
> **前置工作已全部完成并实测**（2026-06-12，另一 session 完成）：harmonizer_server.py 已写好且 smoke 通过、对照组命令已存档、一键启动脚本已就绪。**执行方只需三步：等对照组结束 → 跑一个脚本 → 盯日志。**

## 1. 目标与原理

**目标**：在 NVIDIA nre 官方训练的 difix 蒸馏环节，用 DiffusionHarmonizer（修复器三代）替换 nv-tlabs/Fixer（二代），与正在跑的 Fixer 蒸馏 run 构成单变量 A/B，量化修复器代际增益。

**架构原理**（大g 已搭好的无 key 路径）：官方 difix-distill 本需 NGC 下载 `cosmos_3dgut.pt`；现行方案改为训练容器经 **socket IPC（127.0.0.1:59487）** 调一个独立修复服务。训练侧（pycena，`difix.training.enabled=true`）把降质渲染帧发给 server，server 修复后返回，训练把修复帧当蒸馏目标。**换修复器 = 换 server 进程，训练侧零改动。**

**为什么用非时间（nontemporal）变体**：train-time 蒸馏每个 step 是随机 novel view，无帧序可言；Harmonizer 时间模式需要回读自己前 4 帧输出，不适用。官方推理脚本的 `--nontemporal` 路径 = 同一模型、5D 输入 V=1、无历史参考——server 即按此实现。

## 2. 已就绪资产清单（全部实测在位）

| 资产 | 路径（inceptio） | 状态 |
|---|---|---|
| Harmonizer IPC server | `~/work/nurec_e0/e07/ipc/harmonizer_server.py` | ✅ smoke 通过 |
| 对照组（Fixer run）完整命令 | `~/work/nurec_e0/e07/fixer_run_cmd.json`（12 args） | ✅ docker inspect 抓取 |
| 对照组容器挂载 | `~/work/nurec_e0/e07/fixer_run_mounts.json` | ✅ 同上 |
| **一键启动脚本** | `~/work/nurec_e0/e07/launch_harmonizer_train.sh`（bash -n 过） | ✅ |
| Harmonizer 权重 | `~/repo/harmonizer/models/diffusion_harmonizer.pkl`（5GB） | ✅ E0.2 已下 |
| Cosmos 底模缓存 | `~/repo/harmonizer/src/checkpoints/nvidia/Cosmos-Predict2-0.6B-Text2Image/` | ✅ |
| 容器镜像 | `harmonizer-cosmos-env:latest`（33.1GB）/ `nvcr.io/nvidia/nre/nre-ga:latest` | ✅ |
| smoke 目视对照 | `~/work/nurec_e0/e07/smoke_{input,harmonized}.png` | ✅ 见 §5 |

## 3. server 关键实现细节（已编码，备查）

- **协议与 fixer_server.py 完全一致**：length-prefixed（`>Q` 8 字节长度头）+ `torch.save` 字典 `{input: (h·w, 3) tensor, img_size: [h, w]}` → 返回同形状修复 tensor。前后处理复刻 nre DifixModel：resize 576×1024、归一化 [-1,1]；color_transfer 留在训练侧 client（nre 有 kornia）。
- **模型构造**（来自官方 `inference_pix2pix_turbo_harmonizer.py`）：`pix2pix_turbo_harmonizer.Pix2Pix_Turbo(pretrained_path=diffusion_harmonizer.pkl, timestep=250, train_full_unet=True, freeze_vae=False, vae_skip_connection=False, use_sched=True, dtype=bf16)`；forward 吃 5D `(B, C, V, H, W)`，nontemporal 即 V=1。
- **两个坑已修/已知**：① Cosmos 底模按**相对路径** `checkpoints/nvidia/...` 解析 → server 启动时 `os.chdir("/work/src")`（已写入）；② 端口经 env `HARMONIZER_PORT` 可配——smoke 用 59488 与 Fixer server 并存，**正式跑必须 59487**（训练侧写死）。

## 4. 执行步骤（执行 session 做的事）

**Step 0 — 确认对照组（Fixer run）已结束**：

```bash
ssh inceptio 'docker ps --format "{{.ID}} {{.Image}} {{.Status}}" | grep -v harmonizer'
# pycena 训练容器（image 1c3a838440fc = nre-ga）不在列表 = 已结束。
# 同时确认产物：ls -t ~/work/nurec_e0/... 对照组 out_dir 下新 run-id 目录 + usd-out/last.usdz
```

⚠️ 对照组没跑完**不要**启动——单卡 24GB，两个蒸馏训练并行会 OOM；且脚本第一步会杀 fixer_server，对照组训练会断。

**Step 1 — 一键启动实验组**：

```bash
ssh inceptio 'bash ~/work/nurec_e0/e07/launch_harmonizer_train.sh'
```

脚本内部顺序：停 `fixer_server` 容器（释放 :59487）→ 起 `harmonizer_server` 容器（同 env、:59487）→ 轮询 READY（模型加载 ~3 min）→ 用 `fixer_run_cmd.json` + `fixer_run_mounts.json` **逐字复刻**对照组的 docker run（容器名 `e07_harmonizer_train`）。

**Step 2 — 监控**：

```bash
# 训练主日志
ssh inceptio 'docker logs -f e07_harmonizer_train 2>&1 | grep -E "iter|loss|difix|Error|Traceback"'
# server 是否在被调用（蒸馏钩子 start_step=20000 之后才有流量）
ssh inceptio 'docker logs --tail 20 harmonizer_server'
```

判活要点：① 训练正常推进至 step 20000 前与对照组节奏一致；② **step 20000 后** harmonizer_server 日志开始出现连接处理（没有 ERR 行）；③ 预期总时长 ≥ 对照组（Harmonizer 单帧 ~0.5-1s @576×1024，比 Fixer 慢，钩子 p≈0.5 频率下整体拖慢可感知——slowdown 数字本身就是 A/B 产出的一部分，请记录）。

**Step 3 — 收尾**：训练完成（`docker ps` 中 e07_harmonizer_train 退出、out_dir 出现新 run-id + `usd-out/last.usdz` + `val/metrics.yaml`）后：

```bash
ssh inceptio 'docker rm -f harmonizer_server'   # 释放显存
```

## 5. 验收与对照口径

**第一层（开箱即得，官方口径）**：两个 run 的 `val/metrics.yaml` 配方/口径完全相同（每 3 帧 + 1/4 分辨率 + cpsnr），**直接可比**。三方对照表：

| run | difix 蒸馏 | test/psnr | cpsnr car / road | 备注 |
|---|---|---|---|---|
| E0.3（2026-06-11） | 关 | 30.30 | 34.59 / 38.27 | 已有锚 |
| e07-Fixer（对照组） | Fixer 二代 | 待填 | 待填 | 大g run |
| e07-Harmonizer（实验组） | Harmonizer 三代 | 待填 | 待填 | 本方案 |

⚠️ 注意：官方 val 是 interpolated 口径——蒸馏的主战场是**外推**，interpolated 数字可能差异很小甚至持平，这不代表 A/B 失败（DiFix3D+ ablation：蒸馏提几何 +1.03，大头在外推档）。

**第二层（外推口径，可选移回原 session）**：两个 USDZ 各用 `nre render` 沿原轨迹 + lateral 3m/6m 出帧，喂 3dgrut2 仓库 `scripts/eval_frames_dir.py`（lane warp 指标 / NTA-IoU / FID-KID 同口径）——此工具链在 3dgrut2 主线 session 已就绪（E0.4），可把两个 USDZ 路径交回那边统一评。

**目视基准**：`~/work/nurec_e0/e07/smoke_{input,harmonized}.png` 是 Harmonizer 对本 clip 最重伪影帧（baseline lateral_6m，满屏银色悬浮涂抹）的零微调修复效果——街道/车辆/建筑全部恢复连贯。蒸馏目标的质量上限可参照此对。

## 6. 风险与排错

| 症状 | 原因 | 处置 |
|---|---|---|
| harmonizer_server 启动报 `FileNotFoundError: checkpoints/nvidia/...` | cwd 不在 /work/src（chdir 行被改动） | 确认 server 第 11 行 `os.chdir("/work/src")` 在 import 模型类之前 |
| 训练 step 20000 后报 socket connection refused | server 没起或端口不对 | `docker logs harmonizer_server` 看 READY 行是否 `:59487`（不是 59488） |
| OOM | 对照组没退干净 / 其他 GPU 任务并存 | `nvidia-smi` 清场：训练 ~7GB + server ~8GB 峰值，24GB 内其余任务勿超 8GB |
| server 日志大量 ERR | 输入张量形状/类型不符 | 协议字段是 `input`（(h·w,3)）+ `img_size`（[h,w]）；与 fixer_server 一字不差，若 nre 侧 client 被改动需同步 |
| 训练时长暴涨不可接受 | Harmonizer 单帧推理比 Fixer 慢 | 属预期；可记录 it/s 对比后酌情把 difix p_init 调低对照（但那会破坏单变量，建议先跑完原配） |

## 7. 边界声明（防双头执行）

- 原 session（3dgrut2 主线）**已停止** E0.7 的一切执行与监控（watcher 已撤），只保留：E0.4/E0.6/E1.5 主线 + 接收两个 USDZ 做第二层外推评测（如需要）。
- 本方案执行完成后，请把以下三样回传给大g（或 3dgrut2 主线）：两个 run 的 `val/metrics.yaml` 数字、两个 `last.usdz` 路径、训练 it/s 对比——用于 v4_plan.md §1.3 gap 表与 §5 Done Log 回填。
