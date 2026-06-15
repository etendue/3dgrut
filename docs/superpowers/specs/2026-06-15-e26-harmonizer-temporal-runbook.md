# E2.6 Harmonizer Temporal 后处理 Runbook

> **状态**：✅ 已实测（2026-06-15, inceptio RTX 4090 24GB）
> **关联**：v4_plan E2.6（PR #27），`harmonizer_temporal_server.py` / `harmonizer_client.py` / `harmonizer_protocol.py`
> **目的**：把 DiFix3D+ 论文的 **DiffusionHarmonizer 时间模式（temporal mode）** 接入 `viser_gui_4d` 交互渲染，对连续 Play 序列做去闪烁后处理。这是唯一能发挥 Harmonizer 时序一致性优势的场景（E2.1 离线是单帧，E2.2 训练蒸馏是随机 novel view 无帧序）。

---

## 1. 架构速览

```
┌─────────────────────────────┐         HMN1 (TCP :59490)        ┌──────────────────────────────┐
│  viser_gui_4d (3dgrut2 env) │ ──── 1 + K 帧 uint8 ──────────▶ │  harmonizer_temporal_server  │
│  HarmonizerTemporalClient   │                                   │  (harmonizer-cosmos-env 容器)│
│  - K-帧自引用 history deque  │ ◀──── 单帧 DFX1 reply ────────── │  Pix2Pix_Turbo 5D forward    │
│  - seek/scrub → clear deque  │                                   │  V=1+K → 取 V=0 输出          │
└─────────────────────────────┘                                   └──────────────────────────────┘
```

**关键设计**（详见 `harmonizer_protocol.py` 头注释）：
- **History 在 client**：server 无状态，每连接读 `(1 + K_in)` 帧做一次 forward。Mac 可用 echo stand-in 测协议，不需 GPU。
- **冷启动 warmup 规则**（`harmonizer_client.py` `fix()`）：history 未满 K 时只发 curr（**V=1**），满了才发 V=1+K。原因：Harmonizer 的 temporal CausalConv3d kernel=3，只接受 V=1 或 V≥3，中间值（V=2,3,...,K-1）会让 forward 崩。对齐官方 `inference_pix2pix_turbo_harmonizer.py` 的 `have_history = len >= min_history`。
- **Reset 语义**：seek/scrub/拖动时间轴（`_on_time_change` 的 `source != "play"`）→ client `history.clear()` → 下一帧冷启动。

---

## 2. 前置资产（inceptio 上已就位）

| 资产 | 路径（inceptio） | 说明 |
|---|---|---|
| Harmonizer 权重 | `~/repo/harmonizer/models/diffusion_harmonizer.pkl` (5GB) | Pix2Pix_Turbo ckpt |
| Harmonizer 源码 | `~/repo/harmonizer/src/` | 含 `pix2pix_turbo_harmonizer.py` + Cosmos 依赖 |
| HF 缓存 | `~/.cache/huggingface/` | Cosmos 底模（首次自动拉，~10GB） |
| cosmos 镜像 | `harmonizer-cosmos-env:latest` (33GB) | 含 cosmos_predict2 / transformer_engine / flash-attn |
| E2.6 代码 | `~/repo/3dgrut2-wt/e26/` | git worktree，分支 `e26-harmonizer-temporal` |
| E0.7 nontemporal server | `~/work/nurec_e0/e07/ipc/harmonizer_server.py` | V=1 版，:59489 |

**首次部署 worktree**（Mac → inceptio）：
```bash
# Mac: push 分支
git push inceptio e26-harmonizer-temporal:e26-harmonizer-temporal
# inceptio: 建 worktree + 补 submodule
ssh inceptio 'cd ~/repo/3dgrut2 && \
  git worktree add ~/repo/3dgrut2-wt/e26 e26-harmonizer-temporal && \
  for p in $(git config --file .gitmodules --get-regexp path | cut -d" " -f2); do \
    rsync -a ~/repo/3dgrut2/$p/ ~/repo/3dgrut2-wt/e26/$p/; \
  done'
```

---

## 3. 启动 temporal Harmonizer server（:59490）

**容器内跑**（cosmos 镜像，需要 GPU + Harmonizer 依赖栈）：

```bash
ssh inceptio 'docker rm -f harmonizer_temporal_server 2>/dev/null
docker run -d --name harmonizer_temporal_server --gpus all --net=host \
  -e HARMONIZER_PORT=59490 \
  -e HARMONIZER_CKPT=/work/models/diffusion_harmonizer.pkl \
  -e HARMONIZER_SRC=/work/harm_src \
  -e PYTHONPATH=/work/repo \
  -w /work/repo \
  -v ~/repo/3dgrut2-wt/e26:/work/repo:ro \
  -v ~/repo/harmonizer/src:/work/harm_src:ro \
  -v ~/repo/harmonizer/models:/work/models:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python harmonizer-cosmos-env:latest \
  threedgrut_playground/harmonizer_temporal_server.py'
```

**关键点**：
- `PYTHONPATH=/work/repo` + `-w /work/repo`：`harmonizer_temporal_server.py` 用 `from threedgrut_playground.utils...` 绝对导入，需 worktree 根在 sys.path。
- `-v ~/repo/3dgrut2-wt/e26:/work/repo:ro`：挂 E2.6 worktree（含 temporal server + protocol + client）。
- `--net=host`：viser（host 3dgrut2 env）经 `127.0.0.1:59490` 连容器，无需端口映射。
- 端口 **59490** 与 E0.7（59487）/ E2.1（59489）区分，防混。

**确认 READY**（模型加载 ~3min，warmup ~1.3s）：
```bash
ssh inceptio 'docker logs harmonizer_temporal_server 2>&1 | tail -5'
# 看到 "[harm-temporal] READY" 即可用
```

**查运行时 V 维度**（验证 temporal 在工作）：
```bash
ssh inceptio 'docker logs --tail 20 harmonizer_temporal_server 2>&1 | grep "V="'
# warmup 期 V=1，history 满后 V=1+K（如 K=4 则 V=5）
```

---

## 4. 启动 viser_gui_4d（接 temporal server）

**host 3dgrut2 env 跑**（不需容器，但需 GPU 渲染高斯）：

```bash
ssh inceptio 'export PATH=~/miniforge3/envs/3dgrut2/bin:$PATH && \
  export CUDA_VISIBLE_DEVICES=0 && \
  export PYTHONPATH=~/repo/3dgrut2-wt/e26 && \
  cd ~/repo/3dgrut2-wt/e26 && \
  nohup python threedgrut_playground/viser_gui_4d.py \
    --gs_object <ckpt路径.pt> \
    --dataset_path <pai_*.json路径> \
    --harmonizer_temporal_server 127.0.0.1:59490 \
    --harmonizer_temporal_K 4 \
    --port 8080 > /tmp/viser_e26.log 2>&1 & echo "PID $!"'
```

**实测 ckpt / dataset**（2026-06-15 验证用）：
```bash
--gs_object ~/work/output/p1_2_runB_fix_30k/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7-0506_170051/ckpt_last.pt
--dataset_path ~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json
```

**CLI 参数说明**：
| 参数 | 说明 |
|---|---|
| `--harmonizer_temporal_server host:port` | temporal server 地址；与 `--difix_server` **互斥**（main() reject） |
| `--harmonizer_temporal_K N` | 历史深度，默认 4（论文默认）。K=2 可降延迟（V=3，~600ms vs V=5 ~1000ms） |
| `--port 8080` | viser viewer 端口 |

**访问**：浏览器开 `http://inceptio:8080`（或 inceptio IP）。

**GUI 操作**：
1. Render Controls 面板出现 **"Harmonizer (temporal, de-flicker)"** checkbox + "Harmonizer RTT" 文本框。
2. **勾选** → 启用后处理；RTT 框显示往返毫秒。
3. **Play** → 连续序列走 temporal（前 K 帧 V=1 warmup，之后 V=1+K）。
4. **拖动时间轴 seek** → 自动 reset 历史（下一帧冷启动 V=1）。

---

## 5. 对照：启动 nontemporal Harmonizer server（:59489，E2.1 路径）

用于 A/B 对比 temporal vs 单帧效果。**协议不同**（length-prefixed torch.save，非 HMN1），是独立进程：

```bash
ssh inceptio 'docker rm -f harmonizer_nontemporal_server 2>/dev/null
docker run -d --name harmonizer_nontemporal_server --gpus all --net=host \
  -e HARMONIZER_PORT=59489 \
  -v ~/work/nurec_e0/e07/ipc:/shared:ro \
  -v ~/repo/harmonizer/src:/work/src:ro \
  -v ~/repo/harmonizer/models:/work/models:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python harmonizer-cosmos-env:latest /shared/harmonizer_server.py'
```

> ⚠️ viser_gui_4d **不能直接接** nontemporal server（:59489）——`--difix_server` 接的是 DiFix/Fixer 协议，与 Harmonizer length-prefixed 协议不兼容。要做 temporal vs nontemporal **同模型**对比，用 §6 的离线 demo 脚本（两 server 都接）。

---

## 6. 离线三列对照 demo（raw / nontemporal / temporal）

`scripts/e26_temporal_demo.py` 从连续帧序列出三列对照，不需 viser 交互：

```bash
# 先渲连续帧（interp = 时序连续的 eval 序列）
ssh inceptio 'cd ~/repo/3dgrut2-wt/e26 && \
  PYTHONPATH=. python -m threedgrut.render ... --render-only'  # 产出 <out>/ours_N/ours/interp/

# 跑三列对照（两 server 都要起）
ssh inceptio 'export PATH=~/miniforge3/envs/3dgrut2/bin:$PATH && \
  cd ~/repo/3dgrut2-wt/e26 && \
  python scripts/e26_temporal_demo.py \
    --raw-dir <render输出根>/ours_N/ours \
    --out-dir <render输出根>/e26_demo \
    --mode interp \
    --nontemporal-port 59489 \
    --temporal-port 59490 \
    --K 4'
# 产出: <out>/e26_demo/interp/{raw, nontemporal_fixed, temporal_fixed}/
```

---

## 7. 停止 & 清理

```bash
# 停 server（释放 GPU）
ssh inceptio 'docker rm -f harmonizer_temporal_server harmonizer_nontemporal_server'
# 停 viser
ssh inceptio 'pkill -f viser_gui_4d.py'
# 删 worktree（任务完成后）
ssh inceptio 'cd ~/repo/3dgrut2 && git worktree remove ~/repo/3dgrut2-wt/e26'
```

---

## 8. 已知坑 & 排错

| 症状 | 原因 | 解决 |
|---|---|---|
| server 启动报 `FileNotFoundError: checkpoints/nvidia/...` | Cosmos 底模按相对路径解析，cwd 不对 | 确认 `-w /work/repo` + `HARMONIZER_SRC=/work/harm_src`（server 内部 `os.chdir(HARMONIZER_SRC)`） |
| server 启动报 `ModuleNotFoundError: threedgrut_playground` | worktree 根不在 sys.path | 确认 `-e PYTHONPATH=/work/repo` + `-w /work/repo` |
| server forward 报 `Kernel size (3,1,1) can't be greater than actual input size` | V=2..K-1 禁区（Conv3d kernel=3） | 不该出现——client 的 warmup 规则已挡。若出现说明 client 版本旧（< cce14ba），`git pull` worktree |
| viser 日志 `socket closed mid-frame: got 0/16 bytes` | server 崩了或协议不匹配 | `docker logs harmonizer_temporal_server` 看 server Traceback；通常是上一条的 V 禁区 bug |
| viser 连不上 server（RTT 显示 "unavailable"） | server 没起 / 端口错 / 容器退了 | `docker ps` 确认容器 Up；`docker logs` 看 READY 行端口是否 59490 |
| V=5 延迟 ~1000ms，Play 卡顿 | 5 帧 0.6B forward 重 | 降 `--harmonizer_temporal_K 2`（V=3，~600ms）；或接受 ~1fps 用于质检而非实时交互 |

---

## 9. 端口约定（防混）

| 端口 | 用途 | 协议 | 引入 |
|---|---|---|---|
| 59487 | E0.7 蒸馏 server（nontemporal，训练侧写死） | length-prefixed torch.save | E0.7 handoff |
| 59489 | E2.1 离线 batch_fix nontemporal server | length-prefixed torch.save | E2.1 |
| **59490** | **E2.6 temporal server**（本 runbook） | **HMN1**（20B 头 + (1+K)×uint8） | E2.6 |
| 8765 | DiFix/Fixer server（单帧，`difix_server.py`） | DFX1 | T8/difix |
