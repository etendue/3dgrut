# 从高斯场景推理 LiDAR 点云 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重训的前提下，从训练好的（3D/4D）高斯场景**推理**出一帧仿真 LiDAR 点云——用 3DGRT ray-tracing 内核对一组 LiDAR 扫描射线渲染命中距离 `pred_dist`，反投影成 XYZ `.ply`，并与 GT LiDAR 点云算 range-L1 / Chamfer。

**Architecture:** 复用 3DGRT 内核「接受任意射线 + 已输出 `pred_dist`」的两个既有能力（`threedgrt_tracer/tracer.py:227,99`），只在外围新增三块纯几何 Python：① 球面 LiDAR 射线表构造、② 复用 GT 点方向的射线表（评测用 apples-to-apples）、③ range→XYZ 反投影；再在 `render.py` eval loop 旁挂一条「渲 LiDAR 点云并存盘/评测」分支。intensity 与 ray-drop 需要改 OptiX/CUDA 内核，**不在本 MVP 卡内**，列为 Phase A3 backlog。

**Tech Stack:** 3DGRT (`threedgrt_tracer`, `lib3dgrt_cc`/OptiX), PyTorch, 复用 `scripts/dump_lidar_depth_map.py` 的世界点累积逻辑、`utils/eval_metrics.py::compute_lidar_psnr`。

**⚠️ 头号风险（必须先过 gate）：** 当前训练配方走 **3DGUT 光栅化器**（`lib3dgut_cc`），而本路线要用 **3DGRT ray tracer** 去渲同一组高斯。NVIDIA 原版 3DGRUT 支持「3DGUT 训 / 3DGRT 渲」，但本 fork 是否通必须由 Phase A0 spike 证实——**A0 不过则整条 ray-tracing 路线作废，改走 3DGUT 全景光栅化（另立计划）。**

---

## File Structure

- Create `threedgrut/utils/lidar_rays.py` — 纯几何：`build_spherical_lidar_rays(...)`、`build_rays_from_world_points(...)`、`reproject_range_to_xyz(...)`。无 GPU、无渲染依赖，Mac 可 TDD。
- Create `threedgrut/utils/lidar_render.py` — 把 LiDAR 射线表喂 3DGRT tracer、取 `pred_dist`、反投影、存 `.ply`。GPU 侧薄封装。
- Create `tests/test_lidar_rays.py` — Mac pytest，合成几何验证射线构造与反投影。
- Modify `threedgrut/render.py` — eval loop（~:481-600）新增 `--render_lidar` 分支；metrics 段写 `mean_lidar_range_l1` / `mean_chamfer`。
- (A3, backlog only) `threedgrut/model/model.py:152-165` + 3DGRT OptiX 内核 — intensity SH / ray-drop logit。

**关键复用事实（勘探所得）：**
- `threedgrt_tracer/tracer.py:49` `class Tracer`；`:216` `render()`；forward 接 `ray_ori[B,H,W,3]`+`ray_dir[B,H,W,3]`（`:227-228`），**不写死针孔相机**；输出含 `pred_dist[B,H,W,1]`（`:99`，沿射线 α-composite 命中距离）。射线张量可重塑为 `[B, N_rays, 1, 3]` 当作 H=N_rays/W=1 的"图"。
- `threedgut_tracer/tracer.py:158` 是 3DGUT（光栅化，写死相机模型 `:308-490`）——**不要用它走 LiDAR**。
- Batch dataclass：`threedgrut/datasets/protocols.py:24-76`（`rays_ori`/`rays_dir`/`T_to_world`）。
- GT LiDAR 世界点累积：`scripts/dump_lidar_depth_map.py:92-147` `_accumulate_lidar_world_points()`；点云源 `datasetNcore.py:288-289`。
- 4D 动态变换：`layered_model.py` `_resolve_pose_idx()(:1114)` / `_transform_means_and_active()(:1148)` / `fused_view(timestamp_us)(:848)`。
- LiDAR 评测既有：`utils/eval_metrics.py:14-36 compute_lidar_psnr`；render eval loop `threedgrut/render.py:481-600`，metrics.json 写盘在末尾。

---

## Phase A0 — 可行性 GATE（spike，stop/go；不写产品代码）

**Files:** 无（只读 + 一次性脚本）

- [ ] **Step 1: 确认 3DGUT 训出的 ckpt 能用 3DGRT tracer 渲一帧 RGB**

在一台 GPU 机（inceptio/A800）上，加载一个已有的 3DGUT multilayer ckpt，构造一组**相机针孔射线**（直接复用 eval batch 的 `rays_ori/rays_dir`），但把渲染后端切到 3DGRT `Tracer`，对比与 3DGUT 渲出的 RGB 的 PSNR。

Run（inceptio）:
```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2 && python scripts/spike_3dgrt_render_3dgut_ckpt.py --checkpoint <ckpt.pt> 2>&1 | tail -30'
```
其中 spike 脚本（一次性，不入库）核心：加载 ckpt 的 MoG 参数 → 实例化 `threedgrt_tracer.tracer.Tracer` → 喂一帧相机射线 → 取 `pred_rgb` → 与同帧 3DGUT 渲的 `pred_rgb` 比 PSNR + 各存一张 png 肉眼对照。

- [ ] **Step 2: GO / NO-GO 判据**

- **GO**：3DGRT 渲出的 RGB 与 3DGUT 渲的 PSNR ≳ 30dB（或肉眼一致、仅 AA/边缘差异）。→ 继续 Phase A1。
- **NO-GO**：崩溃 / 几何错位 / PSNR 很低（说明两后端高斯约定不一致，如 density 激活、SH 约定、scale 定义不同）。→ **停止本计划**，记录差异到 Done Log，改走「3DGUT 全景（panoramic）光栅化出深度图 → 反投影」的替代方案（另立 spec）。

- [ ] **Step 3: 确认 LiDAR 传感器世界原点的来源**

定位 `scripts/dump_lidar_depth_map.py:92-147` 里 `_accumulate_lidar_world_points()` 如何拿到 LiDAR 传感器位姿（每帧 `T_lidar2world` 的平移即射线 origin）。确认它是 per-frame 可索引的；若拿不到精确传感器原点，记录 fallback：用相机原点 `T_to_world[:3,3]` 近似（range 会有传感器-相机基线偏差，MVP 容忍，A2 评测时标注）。

- [ ] **Step 4: 记录 spike 结论**

把 GO/NO-GO、PSNR 数字、传感器原点来源写进 `v3_plan_revised.md` § 6 Done Log（即使 NO-GO 也要记，避免重复踩）。

---

## Phase A1 — LiDAR 射线表 + 反投影（纯几何，Mac TDD）

**Files:**
- Create: `threedgrut/utils/lidar_rays.py`
- Test: `tests/test_lidar_rays.py`

- [ ] **Step 1: 写失败测试（球面射线方向 + 反投影自洽）**

```python
# tests/test_lidar_rays.py
import torch
from threedgrut.utils.lidar_rays import (
    build_spherical_lidar_rays, build_rays_from_world_points, reproject_range_to_xyz,
)

def test_spherical_rays_unit_and_count():
    o = torch.tensor([1.0, 2.0, 3.0])
    ori, dir = build_spherical_lidar_rays(o, n_beams=8, h_res_deg=2.0,
                                          vfov_deg=(-10.0, 10.0), hfov_deg=(0.0, 360.0))
    n = 8 * int(360 / 2.0)
    assert ori.shape == (n, 3) and dir.shape == (n, 3)
    assert torch.allclose(ori, o.expand(n, 3))                 # 所有射线同 origin
    assert torch.allclose(dir.norm(dim=1), torch.ones(n), atol=1e-5)  # 单位向量

def test_rays_from_points_recover_range():
    o = torch.zeros(3)
    pts = torch.tensor([[10.0, 0, 0], [0, 5.0, 0], [0, 0, -3.0]])
    ori, dir, rng = build_rays_from_world_points(o, pts)
    assert torch.allclose(rng, torch.tensor([10.0, 5.0, 3.0]), atol=1e-5)
    xyz = reproject_range_to_xyz(ori, dir, rng)
    assert torch.allclose(xyz, pts, atol=1e-4)                 # 反投影回原点

def test_reproject_matches_pred_dist():
    o = torch.tensor([2.0, 0, 0])
    dir = torch.tensor([[1.0, 0, 0]])
    rng = torch.tensor([5.0])
    xyz = reproject_range_to_xyz(o.unsqueeze(0), dir, rng)
    assert torch.allclose(xyz[0], torch.tensor([7.0, 0.0, 0.0]), atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_lidar_rays.py -v`
Expected: FAIL（模块未定义）。

- [ ] **Step 3: 实现纯几何函数**

```python
# threedgrut/utils/lidar_rays.py
"""LiDAR 射线表构造与 range->XYZ 反投影（纯几何，无渲染依赖）。"""
from __future__ import annotations
import math
import torch


def build_spherical_lidar_rays(origin, n_beams=64, h_res_deg=0.2,
                               vfov_deg=(-25.0, 15.0), hfov_deg=(0.0, 360.0)):
    """旋转式 LiDAR 扫描射线（世界系）。返回 (ori[N,3], dir[N,3])，N=n_beams*n_az。"""
    origin = origin.reshape(3).float()
    el = torch.linspace(math.radians(vfov_deg[0]), math.radians(vfov_deg[1]), n_beams)
    n_az = max(1, int(round((hfov_deg[1] - hfov_deg[0]) / h_res_deg)))
    az = torch.linspace(math.radians(hfov_deg[0]), math.radians(hfov_deg[1]), n_az + 1)[:-1]
    el2 = el.repeat_interleave(n_az)            # [N]
    az2 = az.repeat(n_beams)                    # [N]
    ce = torch.cos(el2)
    dir = torch.stack([ce * torch.cos(az2), ce * torch.sin(az2), torch.sin(el2)], dim=1)
    dir = dir / dir.norm(dim=1, keepdim=True)
    ori = origin.unsqueeze(0).expand(dir.shape[0], 3).contiguous()
    return ori, dir


def build_rays_from_world_points(origin, points_world):
    """复用 GT LiDAR 世界点，反推每点的射线。返回 (ori[N,3], dir[N,3], gt_range[N])。"""
    origin = origin.reshape(3).float()
    v = points_world.float() - origin.unsqueeze(0)
    rng = v.norm(dim=1)
    dir = v / rng.clamp_min(1e-8).unsqueeze(1)
    ori = origin.unsqueeze(0).expand(dir.shape[0], 3).contiguous()
    return ori, dir, rng


def reproject_range_to_xyz(rays_ori, rays_dir, ranges):
    """p = o + range * d。返回 [N,3]。"""
    return rays_ori + rays_dir * ranges.reshape(-1, 1)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_lidar_rays.py -v`
Expected: PASS（全 3 个）。

- [ ] **Step 5: Commit**

```bash
git add threedgrut/utils/lidar_rays.py tests/test_lidar_rays.py
git commit -m "feat(lidar-sim): spherical/point-derived LiDAR ray tables + range reprojection"
```

---

## Phase A2 — 渲染命中距离 + 出点云 + 评测（GPU 集成）

**Files:**
- Create: `threedgrut/utils/lidar_render.py`
- Modify: `threedgrut/render.py`

- [ ] **Step 1: 实现 LiDAR 渲染薄封装**

```python
# threedgrut/utils/lidar_render.py
"""把 LiDAR 射线表喂 3DGRT tracer，取命中距离，反投影成点云并存 .ply。"""
from __future__ import annotations
import torch
from threedgrut.utils.lidar_rays import reproject_range_to_xyz


def render_lidar_range(model, tracer, rays_ori, rays_dir, timestamp_us=None):
    """rays_ori/dir: [N,3] 世界系。返回 pred_range[N]（无命中处为 0 或 NaN，依内核约定）。"""
    N = rays_ori.shape[0]
    ro = rays_ori.reshape(1, N, 1, 3).to(model.device)
    rd = rays_dir.reshape(1, N, 1, 3).to(model.device)
    # 4D：若动态场景，先把高斯变换到该时间戳（见 layered_model.fused_view）
    gaussians = model.fused_view(timestamp_us) if hasattr(model, "fused_view") and timestamp_us is not None else model
    out = tracer.render(gaussians, rays_ori=ro, rays_dir=rd)   # 以 tracer.render 真实签名为准
    pred_dist = out["pred_dist"].reshape(N)
    return pred_dist


def save_ply(path, xyz, intensity=None):
    xyz = xyz.detach().cpu().numpy()
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
```

注：`tracer.render(...)` 的确切签名/入参名以 `threedgrt_tracer/tracer.py:216` 源码为准（A0 spike 已实际调过一次，照搬其调用形式）。

- [ ] **Step 2: render.py 增加 `--render_lidar` 分支（球面扫描出 .ply + GT 点方向算 range-L1）**

在 `threedgrut/render.py` eval loop（~:481-600）按帧处理处插入（受 `self.render_lidar` 开关保护，默认关）：
```python
if getattr(self, "render_lidar", False):
    from threedgrut.utils.lidar_rays import (
        build_spherical_lidar_rays, build_rays_from_world_points, reproject_range_to_xyz)
    from threedgrut.utils.lidar_render import render_lidar_range, save_ply
    sensor_o = self._lidar_origin_for_frame(gpu_batch)        # A0 Step3 确认的来源（或相机原点 fallback）
    ts = getattr(gpu_batch, "timestamp_us", None)
    # (a) 模拟扫描 -> 点云 .ply（肉眼/Chamfer）
    so, sd = build_spherical_lidar_rays(sensor_o, n_beams=64, h_res_deg=0.4)
    pr = render_lidar_range(self.model, self.tracer, so, sd, ts)
    valid = pr > 0
    xyz = reproject_range_to_xyz(so[valid], sd[valid], pr[valid])
    save_ply(f"{self.out_dir}/ours_{global_step}/lidar/{iteration:05d}.ply", xyz)
    # (b) GT 点方向 -> apples-to-apples range-L1
    gt_pts = self._lidar_gt_world_points(gpu_batch)           # 复用 dump_lidar_depth_map 累积逻辑
    if gt_pts is not None and gt_pts.shape[0] > 0:
        go, gd, gt_rng = build_rays_from_world_points(sensor_o, gt_pts)
        pr_gt = render_lidar_range(self.model, self.tracer, go, gd, ts)
        m = pr_gt > 0
        if m.any():
            lidar_range_l1.append(float((pr_gt[m] - gt_rng[m].to(pr_gt)).abs().mean()))
```
在累加器初始化处加 `lidar_range_l1 = []`；并实现两个小 helper `_lidar_origin_for_frame` / `_lidar_gt_world_points`（直接调用/搬运 `scripts/dump_lidar_depth_map.py` 的世界点累积，A0 Step3 已定位）。

- [ ] **Step 3: metrics 段写盘**

metrics 聚合段追加：
```python
if lidar_range_l1:
    metrics_json["mean_lidar_range_l1"] = float(sum(lidar_range_l1) / len(lidar_range_l1))
    metrics_json["lidar_range_l1_n_frames"] = int(len(lidar_range_l1))
```

- [ ] **Step 4: GPU smoke —— 出一帧 .ply + range-L1**

Run（inceptio）:
```bash
ssh inceptio 'export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH && export CUDA_VISIBLE_DEVICES=0 && cd ~/repo/3dgrut2 && python render.py --checkpoint <ckpt.pt> --render_lidar --out_dir ~/work/output/lidar_smoke 2>&1 | tail -40'
ssh inceptio 'ls -la ~/work/output/lidar_smoke/ours_*/lidar/ | head && python3 -c "import json,glob; m=json.load(open(glob.glob(\"/home/inceptio/work/output/lidar_smoke/ours_*/metrics.json\")[0])); print(\"range_l1\", m.get(\"mean_lidar_range_l1\"))"'
```
Expected: `lidar/00000.ply` 等文件存在且非空；`mean_lidar_range_l1` 为合理米级数值（典型 < 1m，强依赖 A0 fallback 是否用了相机原点）。用 viser/CloudCompare 肉眼看 .ply 是否成街景形状（道路/车/墙）。

- [ ] **Step 5: Commit**

```bash
git add threedgrut/utils/lidar_render.py threedgrut/render.py
git commit -m "feat(lidar-sim): render LiDAR point cloud from GS via 3DGRT + range-L1 vs GT"
```

---

## Phase A3 — intensity / ray-drop（backlog，需改 CUDA 内核，本卡不展开）

**不在本 MVP 范围**，依赖 A0–A2 全通后再单独立 spec。要点（供后续规划）：
- 高斯参数容器 `threedgrut/model/model.py:152-165` 新增 `intensity_sph[N,k]` + `ray_drop_logit[N,1]`（trivial），并入 `_FORWARD_PARAM_NAMES`、优化器组、layered param loop。
- **真正的难点**：3DGRT OptiX/CUDA 内核需新增「沿射线 α-composite intensity」与「输出 ray-drop 概率」两个通道——需改 `threedgrt_tracer/src/optixTracer.cpp` 与 device 代码。
- 评测扩展：range-L1 之外加 intensity-L2、ray-drop IoU；训练加 intensity/ray-drop 监督（GT 强度来自 NCore 点云源属性）。

---

## 文档同步（CLAUDE.md 强制）

- [ ] **Step 1: v3_plan_revised.md**

新增任务卡（建议归一条新「传感器仿真」线，编号如 P-LIDAR 或并入 v4 backlog——本路线偏「资产/仿真能力」而非 per-class 质量轴，请大g 定编号）。§ 6 Done Log 记 A0 GO/NO-GO + A2 实测 `mean_lidar_range_l1` + commit hash。mermaid 卡片括号全角。

- [ ] **Step 2: v2_architecture.md**

§ 6 文件清单加 `lidar_rays.py` / `lidar_render.py`；§ 7 不变量加：「LiDAR 推理走 3DGRT ray tracer（任意射线 + pred_dist），不走 3DGUT 光栅化；A0 已验证 3DGUT ckpt 可被 3DGRT 渲（PSNR=<实测>）」。

- [ ] **Step 3: Commit**

```bash
git add v3_plan_revised.md v2_architecture.md
git commit -m "docs(plan): log LiDAR-sim A0 gate + A2 range-L1; docs(arch): register lidar-sim modules"
```

---

## Self-Review notes
- **Gate 前置**：A0 把最大未知（3DGUT↔3DGRT 渲染兼容）作为 stop/go，避免在错误前提上写一堆代码——符合 CLAUDE.md「A800 贵、先验证」原则。
- **绕开未知**：MVP 用「复用 GT 世界点反推射线」做量化评测（apples-to-apples range-L1），把对「逆向 LiDAR 扫描参数 API」的依赖降到只在 (a) 球面扫描出 .ply 时用默认参数；精确扫描参数留待 A3。
- **纯几何先行**：射线构造/反投影（A1）全部 Mac TDD，零 GPU 成本；GPU 只用于 A0 spike 与 A2 集成 smoke。
- **4D**：`render_lidar_range` 接 `timestamp_us` 调 `fused_view`，per-frame pose 正确；per-beam rolling-shutter（按方位角时间戳二次变换）属精化项，MVP 用单帧 pose，A3 再做。
- **类型一致**：`build_*` 返回 `[N,3]`/`[N]`；`render_lidar_range` 内部 reshape 成 `[1,N,1,3]` 喂 tracer，取 `pred_dist` reshape 回 `[N]`；`reproject_range_to_xyz` 消费同形状。一致。
- **待确认点**：①A0 的 GO/NO-GO（决定全局）；②传感器原点来源（A0 Step3，否则相机原点 fallback）；③`tracer.render` 精确签名（以源码 + A0 实调为准）。
