# inceptio 5cam 训练实验（去掉 front_tele）

状态：✅ **已执行**（结果在 2026-06-25 调查中随 6cam/2cam A/B 一并回填；B3 归档 2026-07-06）｜ 机器：inceptio ｜ 创建：2026-06-25

## ✅ 执行结果（2026-06-25 调查回填，B3 归档 2026-07-06）

> **依据**：2026-06-25 angry-heisenberg multi-cam 真相调查（见 [`v5_plan.md`](v5_plan.md) §4 Done Log「2026-06-25 multi-cam 真相调查」条目）。本 5cam 实验在该调查中随 6cam / 2cam A/B 一并执行完成。

| 配方 | mean PSNR @7k | 备注 |
|---|---|---|
| 6cam baseline（含 front_tele） | **24.02**（refix 23.24） | 唯一低点 front_tele 18.04 |
| **5cam（去 front_tele，本任务）** | **~24.9** | 去掉长焦低分项后 mean 抬升 ~0.9 |
| 2cam（front_wide + front_tele） | **26.69** | 相机越少 mean 越高（per-pixel mean 稀释） |

**结论**：
- 去掉 front_tele 后 5cam mean **~24.9** > 6cam 24.02——印证 front_tele 18.04 是 mean 的主要拖累项，假设方向正确。
- 但**真根因不是「front_tele 不该一起训」，而是 per-pixel mean loss 缺 per-camera 权重**：telew 实验给 front_tele 加权后 tele 18.04→**26.24**、mean 23.93（6cam 全量保留），说明长焦相机可以纳入、只要加权。**正解 = v5 C1（per-camera loss weight 重实现），而非永久删相机**——5cam 只是移除拖累项的权宜验证，非最终配方。
- 2cam 26.69 > 5cam ~24.9 > 6cam 24.02 的单调关系坐实：mean 随相机数下降主要是 per-pixel 平均稀释，**非「多相机训练崩溃」**（原「6cam 崩 20.20」结论系注入伪造已撤回，见 [`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md) §5 勘误）。
- ⚠️ inceptio **depth-off** 口径，不可与 A800 lidar-on 跨机比。

## 背景 / 动机
- **inceptio 6cam rig ≠ pai 6cam rig**——相机布局和内参不同，不能照搬 pai 配方。
- 6cam baseline（4cabad44 clip, 7k smoke）实测：mean **24.02 dB**，5/6 相机健康（24–27），唯一低点 **front_tele 18.04**。
- front_tele 是长焦看远处：① PSNR 被近处像素稀释；② radial distortion 系数与其余相机量级差异大（标定/畸变疑点）。判断它不适合与其余 5 个一起训。
- **假设**：去掉 front_tele、只训 5cam，让其余 5 个相机监督更干净、整体更稳。

## 目标
5cam vs 6cam A/B：去掉 front_tele 后，5 个公共相机的 per-cam PSNR 是否提升 / 更稳，mean 是否改善。

## 执行（inceptio）（↓ 原计划步骤，实际已按此执行，结果见上方 §执行结果）
1. 环境：`ssh inceptio` → conda `3dgrut2` → 建 git worktree 隔离（**补 submodule**），按项目 CLAUDE.md worktree 工作流。
2. **首步实地确认 3 件事**（别凭记忆）：
   - 6 个 camera 的确切 id/名字 → 排除 `front_tele` → 5cam 列表；
   - clip 路径（4cabad44 的 `pai_*.json` manifest）；
   - 6cam baseline ckpt 路径（对比基线；前序在 `inc4cab_6cam_real/…144727/ours_7000/`，到机器核实）。
3. 训练命令（multilayer + inceptio 铁律）：
   ```
   python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
     n_iterations=7000 \
     path=<clip pai json> \
     'dataset.camera_ids=[<5 个 cam，不含 front_tele>]' \
     trainer.sky_backend=mlp \
     use_lidar_depth=false use_depth_prior=false load_lidar_depth_map=false \
     num_workers=10 \
     out_dir=<out> experiment_name=inc4cab_5cam_7k
   ```
   - 与 6cam baseline **对齐 iter（7k）** 才好 A/B；
   - inceptio 铁律：**depth-off + num_workers=10**（内存），`sky_backend=mlp` 必须。

## 验收（防伪造——此 clip 踩过注入污染伪造数字的坑）
- 训完读 `metrics.json` 真实 per-cam PSNR，**只认 rich log + metrics.json 交叉验证**。
- 对比口径：**5cam 的 5 个相机 vs 6cam baseline 里同样那 5 个相机（24–27）**：
  - 明显提升 / mean 改善 → front_tele 确实拖累，5cam 是更好的配方；
  - 基本没变 → front_tele 影响有限，如实回报。两种结果都算有效结论。
- 注意：inceptio 为 **depth-off** baseline，**不可与 A800 lidar-on 数字跨机比**。
