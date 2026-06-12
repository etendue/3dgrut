# E0.6 — 官方链 actor 编辑能力/限制清单（自有 clip 重建）

> **状态**：🟡 run-book 就绪、待执行回填。
> **场景**：E0.3 自有 clip 官方训练产物 `inceptio:~/work/nurec_e0/train_out/PVG7YYV72YKPLumogi7F7U/artifacts/last.usdz`（clip 9ae151dc，nre 26.4.146）。
> **目的**：走通官方 actor 编辑工作流（删 / 插收割车 / 替换）+ Harmonizer 协调，产出能力/限制清单喂 E2.5（v4_plan.md E0.6 任务卡）。
> **Schema 来源**：上游 `nurec-skills/skills/nre/references/asset-editing.md`（inceptio `~/repo/nurec-skills/`，2026-06-12 克隆）。

## 0. 前置（已验证，2026-06-12）

| 项 | 状态 | 证据 |
|---|---|---|
| last.usdz 含 `sequence_tracks.json`（编辑硬前置） | ✅ | `unzip -p last.usdz sequence_tracks.json` 可枚举 track ids（"13","16","165","18","2",…） |
| `rig_trajectories.json` | ✅ | unzip -l 在档 |
| AH 资产 bundle（3 车 + 3 人 PLY + metadata.yaml） | ✅ | `~/work/nurec_e0/assets/bundle/`；**注意**：metadata 里 `ply_file` 是嵌套路径（`consumer_vehicles/<id>/gaussians.ply`），上传的是扁平文件名 → 执行前按 §1 重排目录 |
| 镜像 | ✅ | nre-ga:latest（26.4.146）/ nre-tools-ga / harmonizer-cosmos-env（33.1GB） |
| viewer 竞态 patch | ✅ | `~/work/nurec_e0/patches/av_patched.py` |
| NGC key | 预期不需要 | R-v4.1 已关闭（E0 全链无 key 跑通）；export-external-assets 文档写需 key，**先无 key 试**，撞 auth 再记录 |

## 1. 资产目录重排（一次性，执行时跑）

```bash
ssh inceptio 'cd ~/work/nurec_e0/assets/bundle && python3 - <<EOF
import os, re, shutil, yaml
meta = yaml.safe_load(open("metadata.yaml"))
for tid, a in meta["assets"].items():
    dst = a["ply_file"]                      # e.g. consumer_vehicles/<id>/gaussians.ply
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = [f for f in os.listdir(".") if f.endswith(f"{tid}.ply")]
    assert src, tid
    shutil.copy(src[0], dst)
    print(tid, "->", dst)
EOF'
```

## 2. 操作序列（serve-grpc 与训练/eval 错峰，用完即收）

> 渲染器选择：**自训 USDZ 用 `--renderer default`**（torch 路径；`--no-enable-nrend` 已弃用、26.04+ 等价写法）。E0.2 实测 nrend 快路径对自训 USDZ 间歇 fail。

1. **R3a 重打包**（产 `target-external-assets.usdz` + `edit-assets.json` 骨架）：

```bash
docker run --rm --gpus all --shm-size=64g \
  -v ~/work/nurec_e0:/workdir/data \
  nvcr.io/nvidia/nre/nre-ga:latest \
  export-external-assets \
  --artifact-path /workdir/data/train_out/PVG7YYV72YKPLumogi7F7U/artifacts/last.usdz \
  --external-assets-dir /workdir/data/assets/bundle \
  --output-edit-file /workdir/data/e06/edit-assets.json \
  --output-artifact-path /workdir/data/e06/target-external-assets.usdz
```

2. **R3b 选编辑对象**：`unzip -p last.usdz sequence_tracks.json | python3 -c "..."` 选 1 辆近景常在车（原轨迹可见时段长）记 `<DEL_ID>` / `<REP_ID>`；插入位姿取某帧 GT cuboid 邻位（空车道），从 `~/work/e11/test_split_manifest.json` 取时间戳区间。
3. **R4 四档渲染**（每档 `render-grpc --edit-assets`，同一段 val 帧；`frames_base` 用空编辑 JSON）：
   - `frames_base/`：`{"metadata":…,"replace":[],"remove":[],"insert":{"asset_ids":[],"data":{}}}`
   - `frames_del/`：`"remove": ["<DEL_ID>"]`
   - `frames_ins/`：`"insert"`：1 辆收割车（`asset_ids:["<AH_CAR_ID>"]` + tracks_poses `[x,y,z,qx,qy,qz,qw]` 序列 + timestamps_us + label_class car + cuboids_dims 取 metadata）
   - `frames_rep/`：`"replace": [{"original_id":"<REP_ID>","replacement_id":"<AH_CAR_ID>","object_size":[]}]`（size 回落 cuboid_dims）

```bash
# server（一次）
docker run -d --name e06_grpc --gpus all --shm-size=64g --net=host \
  -v ~/work/nurec_e0:/workdir/data \
  nvcr.io/nvidia/nre/nre-ga:latest \
  serve-grpc --artifact-glob "/workdir/data/e06/target-external-assets.usdz" \
  --renderer default --enable-editing-actors --test-scenes-are-valid
# 每档
docker run --rm --gpus all --net=host -v ~/work/nurec_e0:/workdir/data \
  nvcr.io/nvidia/nre/nre-ga:latest \
  render-grpc --artifact-path /workdir/data/e06/target-external-assets.usdz \
  --output-dir /workdir/data/e06/frames_<tag> \
  --enable-editing-actors --edit-assets /workdir/data/e06/edit_<tag>.json
```

4. **R5 Harmonizer 协调**：每档时间模式（`--entrypoint python … inference_pix2pix_turbo_harmonizer.py --timestep 250 --resolution 1024 --use_sched`，**每 run 独立输出目录**）；抽 3 帧 `--nontemporal` 单帧对照。
5. **R6 量化**：FID/KID 双口径（编辑档 vs `frames_base` 残留口径；vs GT 帧分布）——用 `scripts/eval_frames_dir.py --kid`（interpolated 模式喂 frames 目录）或 E0.2 同链 FID 脚本；插入车跑 `--nta-iou` 验"被检出且框齐"（GT box = 插入位姿手工 cuboid）。**FID 解读带 E0.2 教训注**（编辑残留是局部伪影，FID 钝感 → 目视为主）。
6. 收服：`docker rm -f e06_grpc`。

## 3. 能力/限制清单（执行后回填）

| 维度 | 观察项 | 记录 |
|---|---|---|
| 删除 | 被删 actor 原位路面/人行道是否出洞、阴影残影、遮挡区如何补全 | ⬜ |
| 删除 | `remove` 仅 render 时过滤（per-frame filter，不动模型参数）→ 多视角一致性 | ⬜ |
| 插入 | 收割 PLY+metadata 兼容性（AssetBank 直接吃下？转换报错面） | ⬜ |
| 插入 | 阴影来源（无 / 烘焙在资产里 / 场景级）、光照失配程度（与 P1.4 spiky/失配对照） | ⬜ |
| 插入 | 位姿参数化（逐帧轨迹必填？静止单 pose 写法）、新 track_id 防冲突 | ⬜ |
| 替换 | 新旧尺寸不一致的露馅模式（object_size 回落 cuboid_dims 的缩放行为） | ⬜ |
| 协调 | Harmonizer 前后 FID/KID 双口径 + 目视（修掉什么/引入什么）、时间模式闪烁 | ⬜ |
| 工程 | edit 全程内存态（`restore_model_parameters` 自动回滚）→ 编辑不落盘、可重复实验 | ⬜（机制已自 schema 文档确认，跑通后打 ✅） |
| 工程 | gRPC 接口面（edit_assets 一次性 vs 逐帧改位姿能力）、时延、`--renderer default` 稳定性 | ⬜ |
| 工程 | export-external-assets 是否要 NGC key（R-v4.1 残留问号） | ⬜ |

## 4. 已知机制要点（来自 schema 文档，免实测）

- 编辑发生在**服务器内存中的 renderable model**，`render-grpc` 完成后自动 `restore_model_parameters` 回滚——官方编辑形态是"会话级"而非"产物级"（产物级要走 `--output-artifact-path` 重打包）。
- `remove` 不是 gRPC 请求的一部分——CLI 在 build 每帧 `dynamic_objects` 列表时本地跳过；自写客户端须自行过滤。
- `replace.object_size` 缺省回落 `external_assets_metadata.cuboid_dims`；两处都没有 → 断言失败。
- `insert.data` 即 `CuboidTracks.to_dict()` 布局；`tracks_id` 与现有 id 冲突直接报错。
- 验证锚（官方清单）：删除后帧无该 actor / 替换显示新几何于原轨迹 / 无 `not found` 警告 / 无 `--edit-assets` 的二次渲染复现未编辑场景（证明 restore 生效）。

## 5. 产物路径（执行后回填）

- 渲染四档：`inceptio:~/work/nurec_e0/e06/frames_{base,del,ins,rep}/` ⬜
- 协调后帧：`…/frames_<tag>_harmonized/` ⬜
- FID/KID 数字 + 目视截图：⬜
