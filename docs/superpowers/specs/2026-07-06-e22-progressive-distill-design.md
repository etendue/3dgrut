# E2.2 渐进外推蒸馏（off-track 战役思路 A 主线）· 任务设计

- 日期：2026-07-06
- 状态：**已批准**（大g 逐节确认：概率混采注入、全图小 λ 起步、执行编排与三读数验收）
- 上游：[v4_plan.md](../../../v4_plan.md) E2.2 任务卡（§1.2 L136 + §4 实现草案 L251）+ [off-track 战役 spec](2026-07-03-offtrack-campaign-design.md) §2.1（执行场地 / 三读数 / fallback / 算力）§5（实验纪律）§7（门 2）
- 范围：**蒸馏管线基建（唯一新代码 = trainer 伪 GT 注入）+ 第一档 lateral_1m 全流程验证**；后续档（2m→3m→6m）按同 runbook 复跑，不另起 spec。**b6a9 迁移（B5 联动）不在本 spec**——门 2 过后才启动。

---

## 0. 设计期已核实的事实（2026-07-06 探索）

1. **渲染 / 修复 / 评估三段零改动纯复用**：
   - 档位：`threedgrut/utils/novel_view.py` 的 `NOVEL_VIEW_MODES` 已含 lateral_1m/2m/3m/6m 四档（1m/2m 系 legacy 锚集、3m/6m 系 E1.1 扩档）；`perturb_c2w()` 为确定性位姿变换（伪 GT pose 重建的依据）。
   - 出帧：`render.py --render-only`（关监督 4.62s/帧）+ frames_map.json（`ts:{camera_id}:{timestamp_us}` key ↔ relpath，E2.1 交付的帧-时刻对齐机制）；`--novel-only` 现硬编码 3m+6m，需小改为可传档位列表（~20 行）。
   - 修复：[`scripts/e21_harmonizer_batch_fix.py`](../../../scripts/e21_harmonizer_batch_fix.py)（IPC socket → harmonizer_server，nontemporal V=1，含 Reinhard color transfer），输入输出同构帧包。
   - 评估：[`scripts/eval_frames_dir.py`](../../../scripts/eval_frames_dir.py)（novel 档 FID/KID + plane-warp lane 指标 + NTA-IoU；lane 白名单 = camera_front_wide_120fov）。
2. **trainer 数据流挂点**：`Batch`（`threedgrut/datasets/protocols.py`）字段齐备（rgb_gt / T_to_world / T_to_world_end / mask / image_infos / timestamp_us）；loss 主体在 `trainer.get_losses()`；layered_l1 有区域 mask 先例、SSIM 全图（D7 设计）。
3. **对照锚三方全现成（同一 baseline ckpt，E1/E2.1 时代）**：baseline 直渲 FID render 75 → 1m 124 → 3m 168 → 6m 193；E2.1 修复后 3m/6m FID 113/139（−33%/−28%）但 lane grad_corr −0.085/−0.095；lane 锚 @3m 0.38 / @6m 0.30；interpolated 守护线 cc ≥ 24.7。**蒸馏起点必须 = 该锚 ckpt**（run 名执行期从 v4 §5 Done Log 定位），否则三方对照失效。
4. **E0.7 α/β' 校准**：官方在线蒸馏钩子形态 = ±3m novel poses + p_scheduler + color transfer（从头训）；B 级权重 40k 蒸馏 interpolated 代价 −0.4~0.5dB → **λ 须小起步**；Harmonizer(29.91) > Fixer(29.77) → 修复器用 Harmonizer。
5. 数据/容器：PAI 9ae 数据 inceptio（`~/work/data/9ae151dc*`）与 A800（`/root/work/yusun/ncore-nurec/data/ncore/clips/`）双侧在位；harmonizer-cosmos-env image 仅 inceptio。

## 1. 目标

把「渲染→Harmonizer 修复→蒸馏回 3D」的渐进档位链在 PAI 9ae 打通并完成第一档（lateral_1m）三读数验证，产出门 1 的第三个决策数字（官方 vs 自研 off-track FID 的自研侧改善证据）与门 2 的推进依据。

## 2. 架构（大g 拍板的两个核心决策）

### 2.1 离线批量包循环（每档一轮，手动分档推进）

```
当前 ckpt → render-only 出档位帧包（375 帧 = 5 相机 × 75 时刻）
         → e21 批修复（IPC nontemporal）→ 修复帧包（同构 + frames_map.json）
         → 蒸馏 run（从当前 ckpt 续训 2-4k 步，概率混采）→ 新 ckpt
         → 三读数 eval → 过 → 下一档；不过 → 调旋钮重跑本档
```

- 离线包模式依据：A800 消费模式（无 harmonizer 容器也能跑蒸馏臂）+ 帧包可审计 + 训练循环无 IPC 依赖。
- 手动分档依据：决策门纪律（门 2 在 3m 档后）；档内 λ 为常数，**不实现自动档位调度器**（YAGNI）。
- 修复用 nontemporal：档位帧是抽样帧非连续序列，与 E2.1 同模式。

### 2.2 trainer 伪 GT 注入 = 概率混采（唯一新代码，~100-150 行 + 单测）

- **新 config 组 `distill.*`**（默认 `enabled=false` **字节等价**）：`frames_dir`（修复帧包）、`p`（采样概率，缺省 0.3）、`lam`（光度权重，缺省 0.1）、`mode`（档位名）。
- **伪 GT batch 构造**：训练步以概率 p 触发——帧包随机取帧 → frames_map 的 ts key 反查原 dataset 帧 → 原 pose（含 T_to_world_end）过 `perturb_c2w(mode)` 得 novel pose → batch =（novel pose，修复帧 as rgb_gt，mask=None）。**修复帧是独立训练样本，绝不与真图像素混合**（不同位姿不可混——设计期已纠正探索 agent 的误案）。
- **loss 路径**：伪 GT batch 走全图 L1 + SSIM 整体 ×λ（不进 layered_l1——novel 帧无 sseg/road mask；sky 不豁免，harmonizer 不动大结构）；正则项、MCMC、exposure 逻辑全部不动。
- **区域加权**：第一档不启用（全图小 λ 起步）；`distill.region_weight_mask` 键位预留作 lane 塌时第二旋钮，触发才实现。
- **增量微调**：`--checkpoint` 加载锚 ckpt 续训 2-4k 步；MCMC 第一档默认参数观察，剧变旋钮 = `max_relocation_fraction` 等现成键。

### 2.3 测试要点（Mac 单测，TDD）

1. `distill.enabled=false` 与现行为字节等价（resolved config diff + loss 数值回归）；
2. p=1 强制路径：每步伪 GT batch、光度项 ×λ 断言、正则项数值不变；
3. pose 重建一致性：同 ts 同 mode 下注入侧与渲染侧 `perturb_c2w` 输出逐元素一致（公差 1e-6）；
4. frames_map 缺帧 / 空包 / mode 不匹配的显式报错路径。

## 3. 第一档（lateral_1m）执行编排

| 步 | 内容 | 场地 | 时长 |
|---|---|---|---|
| 0 | trainer 注入 TDD 实现 + 单测绿 + commit | Mac | 半天 |
| 1 | 定位锚 ckpt（v4 §5 Done Log E1.1/E1.4 run 名）+ 渲染 lateral_1m 帧包 | inceptio | ~30min |
| 2 | harmonizer_server 起 + e21 批修复 375 帧 | inceptio | ~1h |
| 3 | 帧包 rsync A800（分钟级）；双卡 λ∈{0.1, 0.3} 两臂并行蒸馏（锚 ckpt 续 3k 步） | A800 | ~1.5h |
| 4 | 两臂 ckpt 各跑三读数（标准 eval 守护线 + lateral_1m 档 FID/lane） | inceptio / A800 | ~1h |
| 5 | 选优臂 → 数字入档 → 推进 2m 或调旋钮 | Mac | — |

- 后续档同流程复跑，起点 = 上一档选优 ckpt；**3m 档读数后过门 2**（战役 spec §7：FID+lane 双改善 → 推 6m + b6a9 迁移准备；lane 塌调不回 → E2.4 带 kill-criterion）。
- 与其他战线的算力咬合：A800 由 I1（7/6 夜）跑完腾出；inceptio 白天渲染/修复与 B4/B1 错峰（B1 夜间 docker）。

## 4. 每档三读数验收（战役 spec §2.1，缺一不可）

1. **FID/KID@档位改善**——双锚夹逼：蒸馏后直渲 FID 落在 baseline 直渲（1m 档 124）与 E2.1 修复上限之间即机制生效；
2. **lane grad_corr / band_psnr@档位不塌**——E2.1 修复帧本身 lane −0.09 前科，蒸馏回 3D 后 lane 走向是第一档最重要观察；
3. **interpolated 守护线** cc ≥ 24.7——E0.7 证据蒸馏代价 −0.4dB 量级，λ 小起步即为守此线。

## 5. 风险表

| # | 风险 | 缓解 |
|---|---|---|
| 1 | MCMC 增量微调剧变已收敛结构 | 第一档默认参数观察；旋钮 = 降 `max_relocation_fraction` 等现成键，单变量重跑 |
| 2 | lane 被扩散抹平（E2.1 −0.09 前科） | 读数②盯死；第二旋钮 = region 加权（接口已留）；仍不行 → E2.4（kill-criterion 先行） |
| 3 | interpolated 守护线跌破 24.7 | λ 降档重跑；双臂 sweep 快速定位安全 λ |
| 4 | 伪 GT pose 重建与渲染侧不一致（帧-位姿错位 = 蒸馏毒药） | 单测③钉死一致性；执行期首包抽 3 帧目检套准 |
| 5 | A800/inceptio 代码不同步 | 帧包自包含 + 蒸馏臂发射前 grep 验证（CLAUDE.md 铁律） |
| 6 | run 无 kill-criterion 浪费 GPU | 战役纪律 §5：发射前登记 run 名/proxy 步数/读数指标/砍单阈值/砍后动作 |

## 6. 测试策略

- 新代码（trainer 注入 + `--novel-only` 档位参数化）全 TDD：先失败测试后实现（§2.3 四项）；
- 既有测试全绿 = 字节等价证明（E1/eval 链路回归）；
- 数字入档 rich log × metrics.json / eval_frames_dir 输出双源交叉（反伪造纪律）。

## 7. 文档同步（完成定义）

- 每档数字入 [v4_plan.md](../../../v4_plan.md) §5 Done Log + §1.2 E2.2 状态推进 + §1.3 gap 表 E2.2 列；
- 第一档完成 = 门 1 第三个数字就绪（与 B4/B1 数字合并开门 1 决策对话）；门 2 判定另开决策对话；
- [v2_architecture.md](../../../v2_architecture.md)：trainer 注入模块登记 §6 文件清单 + §7 不变量（`distill.enabled=false` 字节等价）。

## 8. 明确出界（YAGNI）

- 自动档位调度器 / λ scheduler（手动分档，档内常数）
- 区域加权实现（接口预留，lane 塌才做）
- temporal 修复模式（抽样帧无时序）
- E2.3 actor 弱面蒸馏、E2.4 域内微调（备选，触发才投）
- b6a9 迁移与 B5 联动（门 2 后）
