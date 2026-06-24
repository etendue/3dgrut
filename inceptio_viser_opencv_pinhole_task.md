# 任务：让 viser_gui_4d 能交互看 OpenCVPinhole rational distortion 训出来的 ckpt

> 创建：2026-06-24，之前 session 的接手任务
> Worktree path：`/Users/etendue/repo/3dgrut2/.claude/worktrees/awesome-haslett-0cc964`
> Branch：`claude/awesome-haslett-0cc964`
> 用户称呼："大g"

> ## ✅ 已解决（2026-06-24，新 session）— **不重训（否定了路径 A）**
>
> **真根因不是前任以为的 distortion / ckpt 类型 dispatch，而是相机光线约定 + 渲染 dispatch**：
> kaolin `Camera.from_args(view_matrix=c2w)` 的 raygen 把矩阵当 w2c 外参算，对 NCore
> Z-up/+Z-forward 相机生成的**世界系光线偏 ~90°**（中心 ray 指世界 +Y 而非 +X 沿路）；
> 且 viewer 默认走 `_render_playground_hybrid`（吃这把错光线），而非接 NCore intrinsics 的
> `_trace_scene_mog` batch 路径——因为 playground 自动载 1 个 GLASS primitive 使
> `has_visible_objects()==True`。eval（render.py）走 Batch 路径所以清晰。
>
> **修复（`threedgrut_playground/engine.py` 3 处，§详见 `inceptio_4cabad44_3dgrut_vs_nre.md` §7）**：
> ① `render_pass` dispatch：NCore 相机强制走 `_trace_scene_mog` batch 路径；
> ② `_trace_scene_mog` guard 放开，v1 MoG 有 NCore intrinsics 时也走 Batch；
> ③ 补 `import numpy as np`（修前任 commit 7417636 潜伏的 `NameError`）。
>
> **验证**：headless `render_pass` non-black 0.80→1.00；真 viser（Chrome 截图）frame0
> front_wide + Follow Ego 另一帧均清晰高速场景（远山/护栏/车道线/绿色指示牌可辨）。
> 截图存 `/tmp/inc4cab_vis/fix_render_pass_idx12.png`、`v3_batch_idx12.png`。
>
> 下方 §6 路径 A（重训 multilayer）**未采用**——已证明真因与 ckpt 类型无关，重训修不到点子上。

## 0. 起源 + 当前状态

之前 session 围绕 inceptio 4cabad44 finalmask clip 做 NRE vs 3dgrut 对照训练。NRE 跑出 28.99 dB；3dgrut **single-cam (front_wide_120fov only)** 跑出 **28.44 dB，eval val PNG 视觉清晰可辨**（远山+护栏+车道线+指示牌全在）。但**用 viser_gui_4d 看同一 ckpt 时画面糊到完全找不到场景**（大片黑 + ego frustum 矩形 overlay + 中央模糊 splat band）。

为什么 eval PNG OK 但 viser 不 OK——之前 session 已经定位到一个层面的差异，但**我（之前 session 的 Claude）的修复路径走错了分支，新 session 需要继续**。

完整诊断 + 数据 + per-class metrics + 视觉对比报告：[`inceptio_4cabad44_3dgrut_vs_nre.md`](inceptio_4cabad44_3dgrut_vs_nre.md)（已存 commit `c52b34a`）。

## 1. 目标

**让 viser_gui_4d 加载 `ckpt_last.pt`（OpenCVPinhole rational distortion 训出来）能交互看到清晰场景**，跟 eval val PNG 视觉一致（或接近）。

成功验证标准：浏览器 http://localhost:8090 上拖 Frame 滑块，能看到 inceptio 高速干线场景沿时间推进（远山/护栏/车道线/绿色指示牌可辨）。

## 2. 关键数据 / Path

| 资源 | 位置 |
|---|---|
| Single-cam ckpt（验证目标） | inceptio:`/home/inceptio/ncore_data/inc4cab_3dgrut_singlecam_out/inc4cab_singlecam_ds05/inceptio_4cabad44-6d56-4c2e-999f-8db32983849c-2406_130110/ckpt_last.pt` |
| Multi-cam ckpts（对照） | `inc4cab_3dgrut_out/`（multilayer 20.20 dB） + `inc4cab_3dgrut_single_out/`（single-layer 20.99 dB） |
| Dataset manifest | inceptio:`/home/inceptio/ncore_data/inc_4cabad44_v2_20s_finalmask/inceptio_4cabad44-6d56-4c2e-999f-8db32983849c/inceptio_4cabad44-6d56-4c2e-999f-8db32983849c.json` |
| Single-cam eval val mp4（视觉 ground truth） | inceptio:`<ckpt run dir>/ours_30000/renders/00000..00024.png` + 已 scp 一份 `/tmp/inc4cab_vis/sc_pred.mp4` |
| 之前几张 viser canvas dump | `/tmp/inc4cab_vis/viser_sc_canvas.png`（initial=cross_left 错向）、`viser_sc_fixed.png`（initial=front_wide）、`viser_sc_with_opcv_fix.png`（我的 fix 后，几乎一样）、`viser_before_after.png` |
| 主仓库（inceptio editable install root） | inceptio:`~/repo/3dgrut2/` ← **重要：见坑 #1** |
| Worktree | inceptio:`~/repo/3dgrut2-wt/viewer/` ← branch checkout 的地方 |

## 3. 之前 session 已经完成的 commits（branch claude/awesome-haslett-0cc964）

| Commit | 改动 |
|---|---|
| `8d86961` | viewer fallback `_load_metadata` `NCoreDataset(...)` 加 `n_val_image_subsample=1`（避开 1918%4 assert） |
| `1052015` | 同 fix 加到 `_load_multi_cam_poses`（multi-cam dropdown 不再报 failed） |
| `c52b34a` | NRE vs 3dgrut 对照报告 |
| `7417636` | **OpenCVPinhole ray re-derivation fix（dead code，见 §6）** |

## 4. 现状 / 已知事实（必读）

1. **eval val PNG 28.44 dB 清晰**——证明 ckpt 学到了正确的场景 gaussians
2. **viser 画面糊**——但不是"模糊"是"完全看不到场景元素，只有 ego frustum + 中央模糊 splat band"
3. **3dgrut multi-cam × OpenCVPinhole 训练 fail**（multilayer 20.20 / single-layer 20.99 都垮）；single-cam (front_wide only) OK 28.44；这是 NVIDIA 3dgrut 在 inceptio 这种**强 rational distortion (k4-k6 显著非零) + 6 cam 跨视角差异**组合下的 fail mode，**不影响**本任务（本任务只针对 single-cam ckpt 的 viewer 显示）
4. **inceptio 6 cam 全部是 OpenCVPinhole rational** — k1=0.48-1.40, k4=0.53-1.76, front_tele 极端 k3=-138/k6=-153。是真 rational mode，**不是** simple polynomial
5. **T_world_to_world_global = identity** — inceptio finalmask world frame === world_global frame，**没有 frame mismatch**
6. **trajectory transform 完全自洽**：ego/rig 从 [0,0,0] 到 [42, -5, 0.2]，camera = ego + T_sensor_rig(c2w-rotated, dtype float32) — 物理位置完全合理

## 5. 之前 session 走错的路径（这次别再撞）

我（之前 session 的 Claude）做了一整套"OpenCVPinhole ray re-derivation"修复（commit `7417636`），改了 5 处代码：
- `viser_gui_4d._load_multi_cam_poses`：抽 OpenCVPinhole intrinsics dict + 预计算 NCore SDK rational distortion 反演 rays
- `viser_gui_4d.__init__`：加 `self.opencv_pinhole_intrinsics / rays / render_wh`
- `viser_gui_4d.fast_render`：透传给 `engine.render_pass`
- `engine.render_pass`：加 `opencv_pinhole_intrinsics / rays` 参数
- `engine._trace_scene_mog`：加 OpenCVPinhole 分支（FTheta path 的平行版本）

**Log 显示 fix 生效**：`[viz_4d] OpenCVPinhole rational ray path active for cam 'camera_front_wide_120fov', W×H=(1918, 1078)`。

**但 viser 视觉画面 fix 前后几乎一样**（before/after 对比图 `viser_before_after.png` 已生成）。

### 真原因（最后才发现）

`engine._trace_scene_mog` 我加的所有 OpenCVPinhole fix **都在 `if isinstance(self.scene_mog, LayeredGaussians)` 分支内**。

但是 single-cam ckpt 用 `ncore_3dgut_mcmc.yaml`（v1 baseline config）训出来是 **v1 MixtureOfGaussians (no layers)**。T8.13-DIAG log 明确显示：
```
scene_mog type: MixtureOfGaussians (v1 MixtureOfGaussians, no layers)
viser checkboxes registered: []
```

v1 MoG ckpt 走 `_trace_scene_mog` 函数尾部的 fallback：
```python
return self.scene_mog.trace(rays_o=rays_ori, rays_d=rays_dir)
```

这条 path **完全不用 batch.intrinsics_***，直接用 kaolin 的 ideal pinhole rays raster。我所有 fix dead code，不触发。

之前 session **耗时 2 小时在错的分支上**，多次跟用户保证"修了 multi_cam dropdown"/"加了 initial_cam_id"/"加了 ray re-derivation"，画面都没变。

## 6. 真正能修的两条路径（任选其一）

### 路径 A：让 ckpt 变成 LayeredGaussians 类型 — **重训 + 用 multilayer config**

**思路**：让 single-cam ckpt 走 LayeredGaussians 分支，**我之前的 fix 自动生效**。

**重训命令**（在 inceptio 上）：

```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd ~/repo/3dgrut2 \
  && nohup python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=30000 \
    num_workers=10 \
    path=/home/inceptio/ncore_data/inc_4cabad44_v2_20s_finalmask/inceptio_4cabad44-6d56-4c2e-999f-8db32983849c/inceptio_4cabad44-6d56-4c2e-999f-8db32983849c.json \
    "dataset.camera_ids=[camera_front_wide_120fov]" \
    "dataset.lidar_ids=[lidar_top_360fov]" \
    "layers.enabled=[background,road,sky_envmap]" \
    "dataset.load_lidar_depth_map=false" \
    "trainer.use_lidar_depth=false" \
    "trainer.use_depth_prior=false" \
    trainer.sky_backend=mlp \
    out_dir=/home/inceptio/ncore_data/inc4cab_3dgrut_singlecam_ml_out \
    experiment_name=inc4cab_singlecam_multilayer \
    > /tmp/inc4cab_3dgrut_singlecam_ml.log 2>&1 & echo PID $!'
```

预计 ~30-40 min。新 ckpt 类型 = `LayeredGaussians`，加载 viser 时**自动走我的 OpenCVPinhole fix path**。

**潜在 issue / 坑**：multilayer config 在 6cam OpenCVPinhole 上之前 fail 到 20.20 dB（之前 session 数据）。**单 cam 用 multilayer 配方没试过**。可能 fail 因为：
- multilayer `bg_road_penalty` + `layered_loss` + sky envmap MLP 都是为 cuboids 数据设计的
- 单 cam 没 cross-cam 干扰，但 multilayer 那套 region weighting 可能仍 sub-optimal
- 验证：跑完看 mean PSNR vs single-cam baseline 28.44，应该接近或不差太多
- 如果新 ckpt PSNR 极烂（< 25），重训用别的方法（**先验证 multilayer single-cam 训得出**，再 viewer 测试）

### 路径 B：改 v1 MoG `scene_mog.trace()` 让它也接 OpenCVPinhole intrinsics — **不重训**

**思路**：修 NVIDIA 上游 v1 MoG `MixtureOfGaussians.trace()` 让它也 distortion-aware。

**入口**：`threedgrut/model/model.py`（应该）的 `MixtureOfGaussians.trace()` 方法。`trace()` 接受 `(rays_o, rays_d)`，但**底层调 CUDA tracer**，可能能传 distortion intrinsics。

**关键查证**：
1. `MixtureOfGaussians.trace()` 实际签名（看是 vararg 还是固定）
2. 底层是不是同一个 `_3dgut_plugin` CUDA kernel（应该是；FTheta path 在 LayeredGaussians 都用 tracer.py:render）
3. 看 `tracer.py:render` 实际签名，trace() 是不是直接给 ray-based call 不走 batch 路径

**风险**：v1 MoG path 走的可能完全不是 batch-based tracer.render 那条。可能是简化的 `splat_to_image(rays_d)` 直接 rasterize。如果是这样**没地方塞 distortion intrinsics**，修不动。

**判断方法**：grep `def trace` 在 `threedgrut/model/` 看实现。如果它内部不构造 Batch + 不调 tracer.py:render，**那不能从这条路修，必须走路径 A**。

### 路径 C（fallback）：放弃 viser，文件 + 报告收工

Eval val PNG / mp4 已经清晰可辨（`/tmp/inc4cab_vis/sc_pred.mp4`）。如果 A 和 B 都太麻烦，可以**承认 viser 对 inceptio OpenCVPinhole 数据的 limitation 是 design issue**，写进对比报告 known limitations，移交。

## 7. 验证流程（Chrome MCP 自动 截图，无需用户在浏览器拖）

新 session 可以直接用 Chrome MCP 验证（之前 session 已经走通这条路）：

```python
# 1. 起 viser (在 inceptio 上)
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && cd ~/repo/3dgrut2-wt/viewer \
  && rm -f /tmp/viser_inc4cab_sc.log \
  && nohup timeout 7200 python threedgrut_playground/viser_gui_4d.py \
       --gs_object <NEW_CKPT_PATH> \
       --dataset_path <DATASET_JSON> \
       --initial_cam_id camera_front_wide_120fov \
       --port 8090 \
       > /tmp/viser_inc4cab_sc.log 2>&1 & echo PID $!'

# 2. sleep 50 确认启动 + listening
# 3. 起 Mac tunnel: ssh -N -L 8090:localhost:8090 inceptio &
# 4. ToolSearch select chrome MCP tools + navigate http://localhost:8090
# 5. javascript_tool dump main canvas to PNG (download trick - 之前 session 用过)
# 6. Read PNG 看视觉
```

## 8. 已知坑（**全部踩过**，新 session 别再撞）

### 坑 #1: editable install 路径优先于 worktree
**症状**：worktree 改了代码 git push 完了，但 viser 启动**仍报旧版本错误**（unexpected keyword argument 等）。
**根因**：inceptio conda env `3dgrut2` 有 `pip install -e ~/repo/3dgrut2/` editable install hook (`__editable__.threedgrut-0.0.2.finder.__path_hook__` 在 sys.path)。Python import `threedgrut_playground.engine` 时**走主仓库 `~/repo/3dgrut2/` 不走 worktree `~/repo/3dgrut2-wt/viewer/`**。
**解法**：改完 worktree 代码后**额外 cp 到主仓库**：
```bash
ssh inceptio 'cp ~/repo/3dgrut2-wt/viewer/threedgrut_playground/{engine,viser_gui_4d}.py ~/repo/3dgrut2/threedgrut_playground/'
ssh inceptio 'find ~/repo/3dgrut2/threedgrut_playground -name __pycache__ -exec rm -rf {} +'
```
backup 已在 `/tmp/engine.py.bak` 和 `/tmp/viser_gui_4d.py.bak`，结束时记得 restore（git status 主仓库会脏）。

### 坑 #2: ckpt type dispatch — LayeredGaussians vs v1 MoG
**症状**：viser_gui_4d 的 GUI panel（Camera dropdown / Follow Camera / Gaussian Layers checkboxes / Time slider 等）**只在 LayeredGaussians ckpt 时注册**。v1 MoG ckpt 注册 0 个 GUI element（只看到 viser builtin "Save Canvas/Reset View/Dev Settings"）。
**诊断**：viser log 找 `T8.13-DIAG` 段，看 `scene_mog type`：`MixtureOfGaussians (v1, no layers)` = 没 GUI；`LayeredGaussians` = 完整 GUI。
**关联**：T8.13 ray re-derivation fix（FTheta + 我加的 OpenCVPinhole）也**只对 LayeredGaussians 生效**，对 v1 MoG dead code。

### 坑 #3: `--initial_cam_id` 必须传
**症状**：没传时 viser default camera 落在 multi-cam list alphabetical 第一个 = `camera_cross_left_120fov`，朝 +y 看，但 ego trajectory 是 +x -y 方向 → 看不到 gaussians。
**解**：viewer CLI 加 `--initial_cam_id camera_front_wide_120fov`。L1944 已有这个 flag，help text 直接写"prevents 'unrecognizable artifacts'"。

### 坑 #4: 1918×1078 不能被 4 整除
**症状**：NCoreDataset assert `Validation subsample factor 4 invalid for camera ... with resolution 1918x1078`。
**fix 已 commit**（`8d86961` + `1052015`）：viewer fallback 调 NCoreDataset 加 `n_val_image_subsample=1`。如果新 session 要在别处构造 NCoreDataset，记得也带这个参数。

### 坑 #5: render.py:1447 `json` UnboundLocal
**症状**：训练完成、ckpt + per-frame PSNR 都正常，但 `metrics.json` 0 bytes，崩在 `json.dump(metrics_json, f, indent=2)` 上。
**临时 workaround**：从 `/tmp/inc4cab_3dgrut*.log` 用 grep + awk 抽 per-frame PSNR 算 mean（之前 session 这么做的）：
```bash
grep -oE "Frame [0-9]+, PSNR: [0-9.]+" /tmp/inc4cab_3dgrut.log | awk -F"PSNR: " '{print $2}' | \
  awk 'BEGIN{mn=999} {n++;s+=$1;if($1<mn)mn=$1;if($1>mx)mx=$1} END{printf "n=%d mean=%.2f min=%.2f max=%.2f\n",n,s/n,mn,mx}'
```
不阻塞 ckpt 视觉验证。

### 坑 #6: ssh inceptio 偶发 exit 255 抖动
**触发**：连发多次 ssh + 含 `pkill` 杀大进程时容易触发（CLAUDE.md 多次记录）。
**应对**：等 5-10s 再 ssh；启动 nohup 用 inline 形式 (`nohup ... & echo PID $!`)，不用 `setsid bash`；ssh 命令之间留间隔。

### 坑 #7: viser_gui_4d 的 "OpenCVPinhole rational ray path active" 是骗人的
**症状**：viser log 显示 `[viz_4d] OpenCVPinhole rational ray path active for cam ...`（commit 7417636 引入），让人以为 fix 生效了。
**真相**：那个 log 是在 `__init__` 时打的，只表示**self.opencv_pinhole_intrinsics 字段被填了**。但 `_trace_scene_mog` 实际渲染时如果 ckpt 是 v1 MoG，**根本不进入用 self.opencv_pinhole_intrinsics 那条分支**。
**验证 fix 是否真生效**：viser canvas 截图 fix 前后对比。如果画面一致 = fix 没生效（v1 MoG fallthrough）。

### 坑 #8: viser canvas 中央的"绿色矩形 + 模糊白色 splat"是 ego frustum overlay 不是 gaussian raster
不要被它骗以为 "gaussian 渲染出来了只是糊"。可以在 viser GUI panel 取消勾 `Ego frustum`（如果有这个控件）看 raster 实际内容。

### 坑 #9: `T_world_to_world_global` 是 identity，不是 frame 问题
**之前思考方向**：有人怀疑 ckpt gaussians 在 world_global frame 但 ego trajectory 用 world frame，导致坐标系不一致 → gaussians 在错位置 → viser 看不到。
**实测验证（已做）**：`pg.get_edge("world", "world_global").T_source_target` 返回 identity matrix。**inceptio finalmask 上 world === world_global，没有 frame mismatch**。

### 坑 #10: distortion mismatch **不是** "完全看不到东西" 的根因
**之前 session 错误推理**：distortion ideal vs rational 不一致 → ray gen 错位 → angular region 错位 → 完全看不到 gaussians。
**用户反驳（正确）**："只是 distortion 不一致只会让画面模糊或扭曲，不会让所有物体消失。我站在苍穹下视野内能看到的还是该看到，只是糊。"
**真原因**：见坑 #2（v1 MoG 不走那条 fix path），跟 distortion 数学**关系小**。

## 9. 推荐执行顺序

1. **先读** `inceptio_4cabad44_3dgrut_vs_nre.md`（已 commit）了解整体上下文
2. **走路径 A 先试**（重训 multilayer single-cam，~30min）。等训练时**同时**做路径 B 的代码探查（看 v1 MoG.trace() 可不可改）。
3. 路径 A 训练完 → 起 viser + Chrome 截图验证。**期望看到清晰场景**（远山/护栏/车道线）。
4. 如果路径 A 也糊（说明问题不在 ckpt 类型 dispatch，是更深层的）→ 走路径 B 或 C。
5. 验证成功后：cleanup viewer + tunnel + 写 commit + push + 更新 task report

## 10. 工具 / 环境备忘

- inceptio conda env：`3dgrut2`，激活 `source ~/miniforge3/etc/profile.d/conda.sh && conda activate 3dgrut2`
- 训练用 GPU：RTX 4090 24GB，CUDA_VISIBLE_DEVICES=0
- 内存铁律：`num_workers=10`（不是 24，会 OOM）
- depth-off 配方（inceptio 内存撑不住 depth）：`dataset.load_lidar_depth_map=false trainer.use_lidar_depth=false trainer.use_depth_prior=false`
- inceptio IP：10.8.31.113（也可走 10.8.28.130；都需 SSH tunnel 看 viser）
- 浏览器 Mac tunnel：`nohup ssh -N -L 8090:localhost:8090 inceptio > /tmp/viser_tunnel.log 2>&1 &`
- Chrome MCP 加载：`ToolSearch query "select:mcp__Claude_in_Chrome__navigate,mcp__Claude_in_Chrome__javascript_tool,mcp__Claude_in_Chrome__tabs_context_mcp" max_results 3`

## 11. 验证 fix 是否真生效的 minimal 测试

新 session 验证 OpenCVPinhole fix（不论走 A 还是 B）是否真起作用，最 minimal 的方式：

1. 起 viser 加载新 ckpt + `--initial_cam_id camera_front_wide_120fov`
2. Chrome navigate `http://localhost:8090`
3. JS dump main canvas as PNG（下载到 ~/Downloads/viser_*.png，cp 到 /tmp/inc4cab_vis/）
4. Read PNG 看：
   - **失败迹象**：大片黑 + 中央 ego frustum 绿色矩形 + 中央模糊 splat → fix 没生效
   - **成功迹象**：能看到 inceptio 高速场景（远山 + 护栏 + 车道线 + 绿色指示牌）→ fix work
5. 如果 success：拖时间滑块（如果 GUI 有的话）几个 frame 看 motion 是否平滑
6. 跟 eval val PNG 视觉对比（`/tmp/inc4cab_vis/sc_pred.mp4` 或同目录 `sc_cam0_t10/t15/t20.png`）确认 viser 跟 eval 一致

## 12. 跟用户的沟通风格

- 用户称呼 "大g"（CLAUDE.md 顶部规则）
- 中文回复
- 直接 + 不要 over-claim：之前 session 多次说"修了 X"但画面没变，最后承认"修没生效"，用户对这种 over-claim 不满
- 不要陷在错的 reasoning 里：用户反驳 distortion 论时直接给反例，应该当场承认而不是套理论；这次任务 #2 / #5 / #7 都犯了这种错
- 用户会用 chrome 截图自己看：fix 完了让他看，不要自己保证

## 13. 我之前几个推理错误的复盘

| 错误推理 | 用户怎么反驳 | 真原因 |
|---|---|---|
| "viser 画面糊是 distortion mismatch" | "近视也只是糊，不会完全找不到物体" | v1 MoG ckpt dispatch（坑 #2）：fix 不触发 |
| "outlier gaussians 干扰 viser raster" | "相机视野内的 gaussians 才被 raster，outliers 不在视野不影响" | 同上 |
| "重训用 multilayer 去 dynamic_rigids = 验证 multi-cam fail 的隔离实验" | (用户其实需要更进一步) | 验证完后没继续诊断为啥 multilayer 也烂 |
| "PinholeForwardProjector 不能用因为它是 polynomial 不是 rational + CPU 不是 GPU" | (这次推理对) | 这是少数对的诊断 |
| "我加的 fix 让 OpenCVPinhole path active" | 截图前后视觉一样 | v1 MoG ckpt 走 fallback path，我 fix 在 LayeredGaussians 分支里 = dead code |

## 14. 任务结束 checklist

- [ ] viser 加载 single-cam ckpt（或重训新 ckpt）能看到清晰场景 OR 文档化 limitation
- [ ] cleanup viewer + Mac tunnel
- [ ] **restore inceptio 主仓库**（坑 #1）：`cp /tmp/engine.py.bak ~/repo/3dgrut2/threedgrut_playground/engine.py; cp /tmp/viser_gui_4d.py.bak ~/repo/3dgrut2/threedgrut_playground/viser_gui_4d.py`（如果走路径 A 重训成功 + 不需要保留 fix code）；或者 merge fix 到 main commit 一份
- [ ] 更新 `inceptio_4cabad44_3dgrut_vs_nre.md` 把 viewer fix 结果 + 新 ckpt PSNR 加进去
- [ ] commit + push 完整 branch
- [ ] 写 1-2 句结论给用户（成功/失败/limitation）
