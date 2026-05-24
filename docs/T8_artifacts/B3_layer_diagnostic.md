# B3 Layer Diagnostic — Baseline `bug4_v2_full_30k` ckpt

**Date**: 2026-05-24
**Branch**: `worktree-distributed-beaver`
**ckpt**: `~/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt` (954 MB, 30k iter, FTheta, 70 tracks, 19.93 s)
**Script**: [`scripts/diagnose_bg_in_cuboid.py`](../../scripts/diagnose_bg_in_cuboid.py)
**Raw JSON**: [`B3_baseline.json`](./B3_baseline.json)
**Mac tests**: 13/13 PASS

---

## 假设

T8/B3 — 浏览器 viser GUI 勾掉 `dynamic_rigids` checkbox 后车辆 Gaussian 不消失，根因是 **训练时车辆 Gaussian 被错误分到 `background` 层，且没有任何机制把它们推回 `dynamic_rigids`**。

## 方法

- 加载 v2 LayeredGaussians ckpt（CPU only，无需 CUDA）
- 对每个 particle 层取 `model.layers[name].positions`（world frame）
- 对每个 active track，按 `tracks_poses[tid] [F,4,4]` 取 5 个均匀采样的 active 帧
- 计算 `T_w2o = inv(pose)` → `local = T_w2o @ pad_h(positions)` → 判定 `|local| ≤ size/2` per axis → OR 跨 (track, frame)
- 输出每层「在 任何 active cuboid 内」的粒子数 + 百分比

## 数据

| 层 | 总粒子数 | 在 cuboid 内 | 占比 |
|---|---:|---:|---:|
| **background** | 1,000,000 | **101,740** | **10.17 %** |
| road | 200,000 | 11,224 | 5.61 % |
| dynamic_rigids | 200,000 | 200,000 | 100 % (按定义，已在 object-local frame) |

**Top-10 tracks（按 bg_inside 排序）**：

| track_id | size (m, w/h/d) | active_frames | bg_inside | road_inside |
|---|---|---:|---:|---:|
| 24  | 5.82×2.45×2.50  | 598 | **73,911** | 173 |
| 405 | 12.45×3.08×3.52 | 263 | 9,347 | 3,257 |
| 7   | 4.41×2.03×1.69  | 571 | 4,893 | 3,893 |
| 165 | 7.99×2.76×3.27  | 599 | 2,757 | 1 |
| 16  | 4.19×1.97×1.65  | 522 | 2,471 | 0 |
| 244 | 4.70×1.95×1.48  | 581 | 2,276 | 2,013 |
| 18  | 4.14×1.94×1.67  | 548 | 1,569 | 0 |
| 549 | 4.14×1.93×1.59  |   8 | 1,466 | 0 |
| 47  | 5.40×2.28×2.36  | 587 | 1,154 | 0 |
| 43  | 4.11×1.90×1.55  | 492 | 662   | 0 |

- 全部 70 tracks 加起来 raw `sum(bg_inside)` = 105,831，去重后 unique = 101,740 → ~4k 个 bg 粒子被多个 track 同时认领（cuboid 邻接 / 相邻车辆）
- 单 track 24（一辆 5.82 m 中型车）一帧就能吞掉 ~74k 个 background 粒子 — 占整个 bg 层的 **7.4 %**

## 结论

**假设钉死**：bg 层中 **10.17 %** 粒子物理上落在 active cuboid 内，这些是「应该归 dynamic_rigids 但错误进了 background」的粒子。它们正是用户报告「勾掉 dynamic_rigids 但车没消失 / 勾掉 background 才让车消失」的视觉根因。

附带发现：
- road 层也有 5.61 % 粒子在 cuboid 内（不那么严重，但同源问题）
- `dynamic_rigids_outside_own_cuboid_pct = None`：ckpt 未保留 `track_ids` buffer（`init_layer_from_points` 注册过，但 `init_from_checkpoint` 路径没还原；属 T8.B3 follow-up，不影响主结论）

## Phase C 验收门

修复 PR 5k smoke 后再跑此脚本，期望：
- `background_inside_pct` **≥ 5 倍下降**（10.17 % → ≤ 2 %）
- `road_inside_pct` 同向下降
- `dynamic_rigids` 层总数从 200k 增到 ≥ 700k（dyn 接管 bg 走人后的视觉空缺）

参考 plan 文件：[`/Users/etendue/.claude/plans/t8-viser-gui-4d-distributed-beaver.md`](../../../../.claude/plans/t8-viser-gui-4d-distributed-beaver.md) §Phase C
