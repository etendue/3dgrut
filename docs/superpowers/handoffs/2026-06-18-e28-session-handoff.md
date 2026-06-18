# E2.8 系统性 dynamic rigid 全替流水线 — 会话交接文档

> **日期**：2026-06-18 · **分支**：`claude/sweet-engelbart-56e3da`（worktree：`.claude/worktrees/sweet-engelbart-56e3da`）
> **GPU 机**：inceptio（RTX 4090 24GB）· worktree：`~/repo/3dgrut2-wt/e28`
> spec：[`../specs/2026-06-17-e28-systematic-rigid-replacement-pipeline-design.md`](../specs/2026-06-17-e28-systematic-rigid-replacement-pipeline-design.md) · plan：[`../plans/2026-06-17-e28-systematic-rigid-replacement-pipeline.md`](../plans/2026-06-17-e28-systematic-rigid-replacement-pipeline.md)

## TL;DR

E2.8 把 NRE USDZ 重建场景拆成「静态底（bg+road）+ per-track dynamic rigid」，把**全部 active/附近 vehicle track 换成干净 AH 资产**（bus/truck 配不到 AH 的从 baseline ckpt 抽真 recon），丢弃非 vehicle，加跨源 MLP 天空，可接 DiffusionHarmonizer 去闪烁。**端到端跑通**，产物 `packed_ckpt.pt` 在 viser 可编辑/可仿真。**35 个 e28 测试全绿**，17 commit 干净在分支上。

**产物**：`inceptio:~/work/output/e28_run/packed_ckpt.pt` = 21 辆干净车（20 AH automobile + 1 跨源真 bus，各被 cuboid 框住、朝向 0.99）+ MLP 天空。

## 6 阶段流水线（已实现）

```
USDZ(checkpoint.ckpt flavor) ──①拆──► ckpt{bg,road,dyn} + viz_4d(ego/track poses)
  + vehicle_catalog(全 70 vehicle track + active帧 + 到ego距离)
        │
        ②配 select_vehicle_tracks_to_place(active≥20帧 & ≤40m) → 21 辆
        │  split_vehicle_tracks_by_ah_match(size ratio≤1.5 走AH，超→recon)
        ▼
  ③替换/插入 replace_all_vehicle_tracks(AH 对齐) + inject_recon_tracks(bus跨源)
        │  + keep_only_track_slots(drop 非vehicle) + add_sky_from_recon(MLP天空)
        ▼
  ⑤QA qa_sanity(coverage/opacity/skip) → packed_ckpt.pt
        ▼
  ⑥viser_gui_4d 消费（+可选 harmonizer 去闪烁）
```

## 新增/修改文件

| 文件 | 作用 |
|---|---|
| **`threedgrut/layers/asset_bank.py`** | bank 查询：class 过滤 + L2 最近 + fallback ladder |
| **`threedgrut/layers/e28_replace.py`** | 全替编排核心：assign / `_align_asset` / `replace_all_vehicle_tracks`（守护非目标字节不变）/ `qa_sanity` / `select_vehicle_tracks_to_place`(insert 过滤) / `split_vehicle_tracks_by_ah_match`(size-gate) / `place_tracks_in_dyn_node` / `extract_recon_node_tensors` / `inject_recon_tracks`(跨源) / `keep_only_track_slots`(drop 非vehicle) |
| **`threedgrut_playground/utils/nre_usdz_viz4d.py`** | USDZ→可渲染 ckpt：移植 fervent-knuth `parse_rig_trajectories`/`build_viz4d_dict`/`build_ftheta_dict` + `convert_usdz_to_ckpt_with_tracks` + `apply_nre_to_world_translate` + `build_vehicle_catalog` + `add_sky_from_recon` |
| **`scripts/e28_systematic_replace_pipeline.py`** | 一条龙 driver（CLI 见下） |
| `threedgrut_playground/viser_gui_4d.py` | +22 行：active-cuboid 只显 vehicle 类（`_is_vehicle_cuboid`，过滤 person/animal） |
| `threedgrut/tests/test_e28_{asset_bank,replace,qa_sanity,usdz_viz4d}.py` | 35 测试 |
| **复用不改（Task 0 从 `claude/practical-mcnulty-94bb49` vendor）**：`threedgrut/layers/{e25_inject,warmstart_metadata,warmstart_ply}.py`、`scripts/e25_inject_ah_replace.py` |

## 怎么跑（inceptio worktree）

**前置**：`ssh inceptio` 后 worktree 在 `~/repo/3dgrut2-wt/e28`（git worktree，已补 submodule）。改了 Mac 代码后 `rsync -az <file> inceptio:/home/inceptio/repo/3dgrut2-wt/e28/<file>`（注意用绝对路径，别用 `~`，会被本地展开）。

```bash
# pytest（纯 CPU 逻辑，inceptio conda env 跑）
ssh inceptio "export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:\$PATH && cd ~/repo/3dgrut2-wt/e28 && python -m pytest threedgrut/tests/test_e28_*.py -q"

# 重生成 packed_ckpt.pt（GPU，build_native_ckpt 建 MoG）
ssh inceptio "export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:\$PATH && export CUDA_VISIBLE_DEVICES=0 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd ~/repo/3dgrut2-wt/e28 && PYTHONPATH=~/repo/3dgrut2-wt/e28 python scripts/e28_systematic_replace_pipeline.py \
  --usdz /home/inceptio/work/nurec_e0/train_out_e07/Qm52hePw64ydcMF3cz3vdA/artifacts/last.usdz \
  --asset_bank /home/inceptio/work/nurec_e0/assets/bundle \
  --out_dir /home/inceptio/work/output/e28_run --out_name packed_ckpt.pt \
  --recon_ckpt /home/inceptio/work/output/v3_base_scratch30k_lam01/ckpt_30000.pt"
```

**driver CLI**：`--insert`(默认开，给 active/附近无gaussian的vehicle也放AH) · `--insert_max_dist_m 40` · `--insert_min_active_frames 20` · `--recon_ckpt <baseline>`(size配不好的大车跨源抽真recon) · `--max_size_ratio 1.5` · `--keep_nonvehicle`(默认丢非vehicle) · `--no_sky`(默认有recon_ckpt时加MLP天空) · `--max_pts`(子采样控显存)

**viser 两种模式（24GB 装不下 viser+harmonizer+sky 三者，二选一）**：
```bash
# sky 模式（流畅探索，FPS~10）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python threedgrut_playground/viser_gui_4d.py \
  --gs_object ~/work/output/e28_run/packed_ckpt.pt \
  --dataset_path ~/work/data/9ae151dc/pai_9ae151dc-...json --port 8090

# harmonizer 模式（去闪烁，需先起 docker server，viser 里手动关 sky_envmap 腾显存）
docker run -d --name harmonizer_temporal_server --gpus all --net=host \
  -e HARMONIZER_PORT=59490 -e HARMONIZER_CKPT=/work/models/diffusion_harmonizer.pkl \
  -e HARMONIZER_SRC=/work/harm_src -e PYTHONPATH=/work/repo -w /work/repo \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v ~/repo/3dgrut2-wt/e28:/work/repo:ro -v ~/repo/harmonizer/src:/work/harm_src:ro \
  -v ~/repo/harmonizer/models:/work/models:ro -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python harmonizer-cosmos-env:latest threedgrut_playground/harmonizer_temporal_server.py
# 等 ~30s-3min 模型加载（轮询 :59490），再起 viser 加 --harmonizer_temporal_server 127.0.0.1:59490 --harmonizer_temporal_K 4
```
浏览器经 IP **10.8.28.130:8090**（inceptio 双网卡，默认走这条；10.8.31.113 被防火墙挡）。

## 关键学习 / 踩坑（务必读）

1. **slot-basis 铁律**：dyn `track_ids` 的 slot → tid 走 `sorted(viz_4d.tracks.keys())`（layered_model.py:378），**不是** `sorted(set(track_order))`。viz_4d 带全部 179 tid，track_order 只是 cuboid 子集 → 两 basis 不一致就让每辆车套**错 tid 的 pose**（=最初的「cuboid≠asset 90°」真凶，commit `adc2a5c`）。
2. **坐标约定**（E2.7 golden + 实测）：bg/road gaussians 在 NRE local frame **要 +translate**（`-world_to_nre.translation`≈+38m）；**ego(rig c2w)/track poses/dynamic_rigids 都不变换**（ego 实测与 baseline NCore 相机旋转差 0.0°、world_to_nre 旋转块=单位阵）。「ego 需要 transform_nre」是误记，translate 只动 bg/road。
3. **跨源 recon 无 90°**：同 clip(9ae151dc)，baseline vs USDZ track pose 旋转差仅 0.4° → baseline bus object-local + USDZ track pose 渲染朝向 0.99，**注入不需 frame 矫正**。AH pickup 摆 bus 上的「90°」是 cuboid H>W 导致 `compute_axis_alignment` 把 W↔H 轴对调的 roll，跟 recon 无关。
4. **NRE vs NCore 重建差异**：USDZ(NRE) dynamic_rigids 只重建了 27 个 cuboid（8 automobile + 19 行人），**没 bus/truck**；baseline(NCore) ckpt_30000 重建了 13 车含 bus(21006 gaussians)/heavy_truck → bus/truck 走跨源。
5. **GPU 显存**：24GB 4090 装不下 viser(~8G) + cosmos-harmonizer-server(常驻~15G) + sky(0.5G)。sky-freeze、harmonizer-Play 报错**都是 OOM**，不是代码 bug。
6. **harmonizer K 必须 = 4**：cosmos temporal 要 V=1+K=**5** 帧；K=3 给 4 帧 shape 不符 → server `socket closed`。RTT ~1s/帧，FPS~6。
7. **opacity ~0.10 正常**：AH 资产 + NRE recon dynamic gaussian 固有低 opacity（E2.5 baseline 更低 0.04 都目测 OK）；qa floor 校准为 0.02（只挡 near-zero 退化）。
8. **inceptio ssh 偶发抖动**（exit 255 / connection closed）；**别快速重试**（疑似触发限流）。长任务用 inline nohup + `echo PID`。

## 已验证（数值 + viser 目测）

- 21 辆车朝向 \|长轴·velocity\|=0.93~1.00；每簇 centroid 偏移 0.05-0.09m（在各自 cuboid 内）；无 90°/重复/孤立。
- bus t405 = baseline 真 recon 21006 gaussians、12.5m、朝向 0.99（非 pickup）。
- sky_envmap 层加载渲染、harmonizer K=4 RTT 1017ms 0 OOM 0 socket 错误。

## 后续工作（留当前 worktree 继续）

| # | 任务 | 怎么做 | 备注 |
|---|---|---|---|
| **Task 7** | **bus AH 收割** → 干净 AH bus 替掉偏淡 recon | `asset-harvester` skill Workflow N：clone+setup(~30GB) → `run_ncore_parser.sh --component-store 9ae151dc.json` → `run.sh` → `orient_gaussians_for_nurec` → `generate_external_assets_metadata.py` → 入 `~/work/nurec_e0/assets/bundle/metadata.yaml` → 重跑 driver | **需你先去 HF 接受 `nvidia/asset-harvester` model card**（gated）；需 ~16GB GPU（先腾）；多小时 |
| **Task 6** | **offline 定量 QA**（NTA-IoU/FID） | driver 加 `--with-quant` 段：render-only 渲 packed_ckpt 帧 → `e21_harmonizer_batch_fix.py` batch → `vehicle_detector.py`(E1.2 NTA) + E1.4 `--novel-fid` → `qa_report.json` | 需 GPU；Monitor 只 grep `⭐\|FID\|NTA\|Traceback\|OOM` |
| **Task 8** | **文档回填** | `v4_plan.md` §1.2/§1.1看板/§5 Done Log 标 E2.8 ✅ + 实测数；`v2_architecture.md` §6 文件清单 + §7 不变量（slot-basis、坐标约定、跨源） | 本会话已起头（commit `7e310e0`）；补完整 |
| — | heavy_truck t165 | 现 8m 走 AH pickup(ratio 1.38<1.5)；`--max_size_ratio 1.3` 可让它也走 baseline recon | baseline 有 t165 recon |

## inceptio 关键路径

- USDZ：`~/work/nurec_e0/train_out_e07/Qm52hePw64ydcMF3cz3vdA/artifacts/last.usdz`（seq=pai_9ae151dc）
- AH bank：`~/work/nurec_e0/assets/bundle/`（3 consumer_vehicles + 3 VRU_pedestrians，**无 bus/truck**）
- baseline recon ckpt：`~/work/output/v3_base_scratch30k_lam01/ckpt_30000.pt`（含 bus/truck recon + MLP sky_envmap_state）
- dataset manifest：`~/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json`
- 产物：`~/work/output/e28_run/{packed_ckpt.pt, replace_report.json, qa_sanity.json}`
- harmonizer：image `harmonizer-cosmos-env:latest` + `~/repo/harmonizer/{src,models}` + HF `nvidia-Fixer`
