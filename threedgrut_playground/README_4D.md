# viser_gui_4d — 4D 场景可视化（Stage 8）

`viser_gui_4d.py` 是一个浏览器版交互可视化器，专为 v2 LayeredGaussians 训出的 4D 场景设计：用户在时间轴上拖动，即可看到 Gaussian 背景里的车随 dynamic_rigids 层动起来，配合 ego 轨迹折线、cuboid wireframe、LiDAR 点云一起。

原静态 3D 版本 `viser_gui.py` 保持不变，仍可用于 v1 ckpt。

---

## Quick Start

### 1. 训练（A800）

要让 ckpt 携带 4D 元数据，使用专用 config：

```bash
ssh a800-x2
conda activate 3dgrut
cd /root/work/yusun/repo/3dgrut

python train.py --config-name apps/ncore_3dgut_mcmc_v2_full_4dviz \
    dataset.path=/root/work/yusun/ncore-nurec/data/ncore/clips/<clip_id> \
    output_path=/root/work/yusun/ncore-nurec/output/4dviz_smoke
```

该 config 继承 `ncore_3dgut_mcmc_v2_full_exposure`，额外打开 `viz_4d.enabled=true`，从而在 `Trainer.save_checkpoint` 时把 ego 轨迹、tracks、LiDAR 元数据打包进 `ckpt["viz_4d"]`。

任何其他 v2 config 都可以用 CLI override 临时开启：

```bash
python train.py --config-name apps/ncore_3dgut_mcmc_v2_full \
    viz_4d.enabled=true \
    viz_4d.lidar_road_subsample=200000 \
    ...
```

### 2. 启动 4D viewer

**有 RT cores**（所有 RTX 卡 + **Hopper datacenter H100/H800/H200** + Blackwell B100/B200 + Ampere workstation RTX A5000/A6000）：

```bash
python -m threedgrut_playground.viser_gui_4d \
    --gs_object /path/to/ckpt_last.pt \
    --default_gs_config apps/ncore_3dgut_mcmc.yaml \
    --port 8080
```

**无 RT cores**（**Ampere datacenter A100 / A800** —— NVIDIA 故意把 RT cores 阉割，给 RTX 系列让位）：必须加 `--no_gaussian_render`，否则 OptiX 扩展 dlopen 时 segfault。注意 H100 跟 A100 同属 datacenter 但**架构不同**（Hopper vs Ampere），H100 保留了第 3 代 RT cores，**不需要**此 flag。

```bash
python -m threedgrut_playground.viser_gui_4d \
    --gs_object /path/to/ckpt_last.pt \
    --no_gaussian_render \
    --port 8080
```

此模式跳过 Engine3DGRUT 全套（不渲染 Gaussian 背景），只保留 scene primitives + timeline。Mac 远程看：

```bash
ssh -L 8080:localhost:8080 a800-x2
# 浏览器 http://localhost:8080
```

### 2b. vast.ai 租 RT cores GPU 完整 Gaussian 渲染（T8.12 实测路径）

如果手里只有 Ampere datacenter A100/A800 但需要看完整 Gaussian 渲染（不只 scene primitives），最便宜路径是租 vast.ai 上带 RT cores 的 GPU。**RTX 4090 24GB** 完全够用且最便宜（$0.5-0.8/hr，1h 即可完成验证），Ada Lovelace 第 3 代 RT cores OptiX 兼容。

```bash
# Mac 端: vastai CLI 创建实例
VASTAI=/path/to/.venv/bin/vastai
API_KEY=<your-vast-api-key>

# 1. 搜便宜 4090 (按 $/hr 排序)
"$VASTAI" search offers \
    "gpu_name=RTX_4090 num_gpus=1 cpu_cores>=8 cpu_ram>=32 disk_space>=80 reliability>0.95 rentable=true" \
    --storage 80 --raw --api-key "$API_KEY"

# 2. 创建实例 (image 必须是 cuda12.1 + devel; cuda_helper.sh 已加 12.1 case)
"$VASTAI" create instance <OFFER_ID> \
    --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
    --disk 80 --label my_viser_4d --ssh --direct --api-key "$API_KEY"

# 3. 注入 ssh key, ssh -tt 进去, 装 uv + opencv libs + 跑 install_env_uv.sh, 装 viser
#    (详见 docs/T8.12_handover_day1.md §5.2 完整脚本)

# 4. 启动 viser (必须 setsid 脱离 ssh 会话; nohup 在 vast.ai container 不持久)
setsid bash -c 'python -m threedgrut_playground.viser_gui_4d \
    --gs_object /root/ckpt_with_viz_4d_v2.pt \
    --default_gs_config apps/ncore_3dgut_mcmc.yaml \
    --port 8080 > /tmp/viser.log 2>&1' < /dev/null &
disown

# 5. Mac 端 ssh -L 转发, 浏览器开 http://localhost:8080
ssh -N -T -o ControlMaster=no -o ControlPath=none \
    -L 8080:localhost:8080 my-vast-instance
```

**踩坑 / 实测要点**（T8.12 整理）:

- vast.ai pytorch:cu121 镜像 CUDA 12.1 不在原 `scripts/cuda_helper.sh` 支持列表（11.8/12.4/12.6/12.8/13.0），已加 12.1 case
- **必须用 `setsid` 而非 `nohup`** 启动 viser；vast.ai container `nohup` 会被回收
- rsync over ssh 必须 `-e "ssh -T"`，否则 RequestTTY force 跟二进制流不兼容
- Reset View 按钮真的能 snap camera 回 ckpt 训练相机位置（之前只重置 up_direction，T8.12 fix 后完整重置 position + wxyz + look_at + up_direction）
- viser server / client subprotocol 版本必须严格匹配；中间 pip downgrade 会让 JS bundle 跟 server 版本不一致 → WebSocket 被拒只看到 /WorldAxes

**T8.12 实测 RTX 4090 Norway $0.630/hr @ 1024×~600 → 87 FPS 稳态**。

### FTheta fisheye 渲染（T8.13）

✅ **FTheta-trained ckpts 现已支持视觉匹配 render.py**。schema_v2 起 `viz_4d.ego` 持久化完整 8-key FTheta polynomial intrinsics (`resolution / shutter_type / principal_point / reference_poly / pixeldist_to_angle_poly / angle_to_pixeldist_poly / max_angle / linear_cde`) + `primary_camera_resolution`。当 ckpt 含这些字段时，viser_gui_4d 自动启用 FTheta 分支，调用 3dgut UT rasterizer 走 `Batch.intrinsics_FThetaCameraModelParameters` 投影路径（kernel 在 `threedgut_tracer/tracer.py:471` 已原生支持，本任务全 Python 改动）。

⚠️ **限制**: render W×H 锁定到训练分辨率（FTheta principal_point 是像素坐标，分辨率改了就错位）。`resolution_slider` 在 FTheta 模式下自动隐藏 + GUI 显示提示。

**启动日志区分**:
- `[T8.13] FTheta intrinsics 已加载 (resolution=(W,H), max_angle=...)` —— v2 FTheta 路径
- `[T8.13] 无 FTheta intrinsics, 走 pinhole approximation 路径 (T8.12 行为)` —— v1 ckpt / 非 FTheta 相机

**老 v1 ckpt 升级**:

```bash
python -m threedgrut.viz.inject \
    --ckpt /path/to/old_v1_ckpt.pt \
    --dataset_path /path/to/scene_manifest.json \
    --out /path/to/new_v2_ckpt.pt
```

注入完 ckpt 即含 FTheta 字段，后续启动 viser 自动走 FTheta 路径。

T8.12 实际产出（commit 价值）：
- 修了 2 个真实 Stage 8 集成 bug (engine.py 缺 camera intrinsics → viser 一连即崩; layered_model.py sky_envmap 残留 CPU → addmm 报错)
- Reset View 真重置到 initial_c2w
- Infra: `cuda_helper.sh` 加 CUDA 12.1 case, viser setsid 启动模板, vast.ai 实例创建脚本
- 完整 Day 1→2 交接文档 `docs/T8.12_handover_day1.md`


### 3. 旧 ckpt（无 viz_4d）的两种 fallback

如果手里只有训好的旧 v2 ckpt（没有 `viz_4d` 块），有两条路径可选。

**方案 A — viewer 端 lazy fallback**（最简单，每次启动现场提取）：

```bash
python -m threedgrut_playground.viser_gui_4d \
    --gs_object /path/to/old_v2_ckpt.pt \
    --dataset_path /path/to/scene_manifest.json \
    --port 8080
```

该路径会 lazy import `NCoreDataset` + `extract_4d_metadata`，因此**无 NCore SDK 的机器只要不传 `--dataset_path` 就不会崩**。每次启动都会重新加载 dataset（首次 ~30s）。

**方案 B — 一次性注入 viz_4d 块到 ckpt**（推荐，省每次 NCore SDK reload，让 ckpt 在无 SDK 机器上可用）：

```bash
python -m threedgrut.viz.inject \
    --ckpt /path/to/old_v2_ckpt.pt \
    --dataset_path /path/to/scene_manifest.json \
    --out /path/to/new_ckpt_with_viz_4d.pt    # 或省略 --out 原地覆盖（留 .bak 备份）
```

注入后的 ckpt 字节级保留所有原字段（model / strategy / post_processing / exposure_state / sky_envmap_state 不变），只新增顶层 `viz_4d` 块。之后启动 viewer 不需要 `--dataset_path`：

```bash
python -m threedgrut_playground.viser_gui_4d \
    --gs_object /path/to/new_ckpt_with_viz_4d.pt \
    --port 8080
```

适用场景：经常分享/回放同一个 ckpt、想给没装 NCore SDK 的同事看效果。

### 4. v1 ckpt

直接传 v1 ckpt 即可，viewer 自动降级为静态 3D 模式（等价 `viser_gui.py`），timeline / visibility 控件不出现。

---

## GUI 控件说明

### Static Render（永远显示）

| 控件 | 作用 |
|---|---|
| Reset View | 重置 camera up direction |
| Resolution | 渲染分辨率（384–4096） |
| Near / Far | 视锥体裁面 |
| FPS | 实时 GPU render 时间倒数 |

### Timeline（仅 4D 模式）

| 控件 | 作用 |
|---|---|
| Time (us) | 当前时间戳，按真实微秒推进。范围 = `ckpt["viz_4d"]["viewer_defaults"]["t_us_first"/last"]` |
| Frame | 整数帧号（0..F-1），与 Time 双向绑定 |
| ▶ Play / ⏸ Pause | 切换播放状态。Play 时 `wallclock_dt × speed × 1e6` 推进 `t_us` |
| Loop | 到末尾是否回到 `t_us_first` |
| Speed | 播放倍速（0.1–4.0），1.0 = 实时 |

### Visibility（仅 4D 模式）

| 控件 | 默认 | 内容 |
|---|---|---|
| Ego trajectory | ✓ | 整段 ego polyline (绿色 catmull-rom) |
| Ego frustum | ✓ | 当前 t 时刻相机视锥 (绿色) |
| Track trajectories | ✓ | 所有 dynamic track 的历史折线，class 着色 |
| Active cuboids | ✓ | 当前 t 时刻所有 active 的 cuboid wireframe，instance 着色 |
| Road LiDAR | ✓ (when present) | 静态道路点云 |
| Dynamic LiDAR | ✗ | 动态物体点云（默认关闭，避免拥挤） |
| World axes | ✗ | 原点坐标轴 |

---

## ckpt schema (`ckpt["viz_4d"]`, schema_version=1)

```python
{
    "schema_version":             1,
    "dataset_type":               "ncore",
    "sequence_id":                str,

    "ego": {
        "poses_c2w":              Tensor[N, 4, 4] float32,  # primary cam C2W (world frame)
        "frame_timestamps_us":    Tensor[N]      int64,
        "primary_camera_id":      str,
        "primary_camera_fov_y_rad": float,
        "primary_camera_aspect":    float,
    },

    "tracks": {
        "<tid>": {
            "poses":      Tensor[F, 4, 4] float32,
            "size":       Tensor[3]      float32,    # cuboid LWH full extent
            "frame_info": Tensor[F]      bool,        # active per frame
            "class":      "automobile" | "heavy_truck" | "bus" | ...,
        }, ...
    },
    "tracks_camera_timestamps_us": Tensor[F] int64,

    "lidar": {
        # Road LiDAR: static, world frame
        "road_xyz":               Tensor[M_road, 3] | None,
        "road_rgb":               Tensor[M_road, 3] | None,
        "road_n_total":           int | None,
        "road_subsample":         int | None,
        # T8.11 — dynamic LiDAR: per-track object-local frame so viewer
        # can transform back to world every frame, keeping points glued
        # to the moving cuboid.
        "dynamic_local_xyz":      Tensor[N, 3] | None,   # object-local
        "dynamic_track_ids":      Tensor[N]    | None,   # idx into track_names
        "dynamic_track_names":    list[str]    | None,   # idx → tid
        "dynamic_pts_per_track":  int          | None,
        # Legacy world-frame union (pre-T8.11 ckpts / fallback only)
        "dynamic_xyz":            Tensor[M_dyn, 3] | None,
        "dynamic_rgb":            Tensor[M_dyn, 3] | None,
        "dynamic_n_total":        int | None,
        "dynamic_subsample":      int | None,
    },

    "viewer_defaults": {
        "initial_c2w":  Tensor[4, 4] float32,
        "near":         float,
        "far":          float,
        "resolution":   int,
        "t_us_first":   int,
        "t_us_last":    int,
    },
}
```

体积估算（F=1500 / K=179 / road 200K / dyn 100K）：**~35 MB**。v2 ckpt 主体通常 500 MB–2 GB，可接受。

### 控制体积

`configs/base_gs.yaml` 默认值：

```yaml
viz_4d:
  enabled: false                       # 默认关闭，开启需要在 app config 中翻
  include_lidar: true                  # 关掉则 lidar 全是 None
  lidar_road_subsample: 200000         # 默认 200K road pts (从 629K subsample)
  lidar_dynamic_subsample: 100000      # 默认 100K dynamic pts
```

---

## Troubleshooting

### `viser not installed`

```bash
pip install viser==1.0.0
```

或在你的 conda env 里。注意 `threedgrut_playground/requirements.txt` 已 pin viser==1.0.0。

### `kaolin` 缺失 / 报错

`viser_gui_4d.py` 复用 `Engine3DGRUT`，依赖 kaolin。按主 README 装 3dgrut 完整 env 即可。

### ckpt 没有 `viz_4d` 块

两种原因：
1. **训练时没开** `viz_4d.enabled=true` —— 用 `apps/ncore_3dgut_mcmc_v2_full_4dviz.yaml` 重训，或对旧 ckpt 用 `--dataset_path` fallback。
2. **不是 v2 LayeredGaussians ckpt**（是 v1 flat MoG）—— viewer 自动降级为静态 3D，行为等价 `viser_gui.py`。

### `--dataset_path` 模式启动失败 / ImportError

只有传了 `--dataset_path` 才会 import NCore SDK + cv2 + kornia 等重依赖。检查：
- conda env 是否包含 NCore SDK（仅 NVIDIA 内部可用）
- ckpt 中的 `config.dataset` 是否与提供的 manifest 兼容

不需要 4D fallback 时不要传 `--dataset_path`，那样就走纯 ckpt 路径，不会触发这些 import。

### Active cuboid 不动 / 永远停在 frame 0

排查清单：
1. `ckpt["viz_4d"]["tracks"]` 是否非空？`python -c "import torch; c=torch.load(<path>, weights_only=False); print(len(c['viz_4d']['tracks']))"`
2. `tracks_camera_timestamps_us` 是否非空且单调递增？
3. `Engine3DGRUT._trace_scene_mog` 分支是否进入了 LayeredGaussians 路径？（Engine 实例 `isinstance(engine.scene_mog, LayeredGaussians)` 应为 True）
4. `model.tracks_poses` 是否在 `load_3dgrt_object` 中被 `populate_tracks` 重建？

### 帧率太低 / 浏览器卡

- 调低 Resolution
- 关掉 Track trajectories / Cuboids（visibility checkbox）
- 确认 conda env 用的是 CUDA，不是 CPU fallback

### Multi-camera frustum 没显示其他相机

设计取舍：当前只渲染 `primary_camera_id` 的 frustum，避免 7 相机环视拥挤。多相机的 pose 仍 concat 进 `poses_c2w`（用于 ego polyline 完整轨迹），未来可扩展为 togglable per-camera frustum.

### `OptiX dlopen segfault` / `lib3dgrt_cc.so` 加载崩溃

GPU 没 RT cores。具体来说：**Ampere datacenter SKU（A100 / A800）NVIDIA 故意把 RT cores 阉割**，所以 OptiX BVH traversal 在这两个卡上 segfault。加 `--no_gaussian_render` 跳过 Engine3DGRUT 即可。

| 架构 | 卡型 | RT cores | OptiX | 需要 `--no_gaussian_render` |
|---|---|---|---|---|
| Ampere datacenter | **A100 / A800** | ❌ | ❌ | ✅ 必加 |
| Hopper datacenter | H100 / H800 / H200 | ✅ 第 3 代 | ✅ | ❌ |
| Blackwell datacenter | B100 / B200 | ✅ 第 4 代 | ✅ | ❌ |
| RTX consumer | 3090 / 4090 / 5090 | ✅ | ✅ | ❌ |
| Workstation | RTX A5000 / A6000 | ✅ | ✅ | ❌ |

NVIDIA NRE-GA 容器（`nvcr.io/nvidia/nre/nre-ga`）官方推荐 H100，正是因为 H100 有 RT cores 能跑 viser playground 完整 Gaussian 渲染。这是 3dgrut 不是 bug 而是 GPU 硬件 spec 差异。

### Dynamic LiDAR 点云不随 cuboid 动（T8.11 之前 bug）

旧 ckpt（T8.11 之前 inject 的）没存 per-track object-local 字段，只有 `dynamic_xyz` 世界帧聚合。重新 inject 或重训即可：

```bash
python -m threedgrut.viz.inject \
    --ckpt /path/to/old.pt \
    --dataset_path /path/to/manifest.json \
    --out /path/to/new.pt
```

新 ckpt 字段验证：`viz_4d.lidar.dynamic_local_xyz` 应该非 None，`dynamic_track_names` 应该是 list[str]。

### Cuboid 朝向不对（拐弯/掉头时偏）

这是已知限制：`tracks_loader.py:195` 当前 `pose[:3,:3]=I`（translate-only），尚未集成 `bbox.rot`。直行车视觉 OK，急拐弯/掉头偏。这是独立子任务（涉及训练侧 `dynamic_rigid_init` 重训），未列入 Stage 8 主线。

---

## 实现说明（开发者）

### 文件清单

```
threedgrut/
  viz/
    __init__.py
    metadata.py            # extract_4d_metadata + 子函数（纯 CPU）
  layers/layered_model.py  # _populate_tracks_impl 加 tracks_metadata
  trainer.py               # save_checkpoint 注入 viz_4d

threedgrut_playground/
  engine.py                # load_3dgrt_object 加 LayeredGaussians 分支 + render_pass timestamp_us
  viser_gui_4d.py          # Viser4DViewer 主体 + main 入口
  utils/
    viz4d_metadata.py      # FourDMetadata dataclass + lookup helpers（纯 numpy）
    cuboid.py              # UNIT_CUBE_EDGES + cuboid_world_edges + class_color + instance_color
    viser_math.py          # mat_to_wxyz (Shepperd/Markley quaternion)
  README_4D.md             # 本文件

configs/
  base_gs.yaml             # 加 viz_4d 默认配置块
  apps/ncore_3dgut_mcmc_v2_full_4dviz.yaml   # 开 viz_4d.enabled=true 的训练 preset
```

### 单元测试（Mac CPU）

```bash
source .venv/bin/activate
pytest threedgrut/tests/test_engine_layered_load.py \
       threedgrut/tests/test_viz_4d_metadata.py \
       threedgrut/tests/test_viz4d_metadata_loader.py \
       threedgrut/tests/test_viser_math.py \
       threedgrut/tests/test_cuboid_wireframe.py -v
```

预期：32 PASS，覆盖 ckpt schema roundtrip / subsample / `include_lidar=false` / `mat_to_wxyz` round-trip / cuboid 几何变换 / `lookup_frame_idx` 二分 / `active_tracks_at` 边界。

### A800 烟测（T8.8，待跑）

```bash
ssh a800-x2
conda activate 3dgrut && export CUDA_VISIBLE_DEVICES=0
cd /root/work/yusun/repo/3dgrut

python train.py --config-name apps/ncore_3dgut_mcmc_v2_full_4dviz \
    dataset.path=/root/work/yusun/ncore-nurec/data/ncore/clips/<clip_id> \
    output_path=/tmp/4dviz_smoke \
    training.iterations=100

python -c "
import torch
c = torch.load('/tmp/4dviz_smoke/ckpt_last.pt', weights_only=False)
v = c['viz_4d']
print('schema:', v['schema_version'], 'tracks:', len(v['tracks']),
      'ego_N:', v['ego']['poses_c2w'].shape[0],
      'road_pts:', v['lidar']['road_xyz'].shape if v['lidar']['road_xyz'] is not None else None)
"
```

期望：`schema:1 tracks:>0 ego_N:>0 road_pts:(200000, 3)`
