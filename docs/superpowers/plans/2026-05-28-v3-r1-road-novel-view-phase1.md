# V3-R1 Road Novel-View Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 1 周窗口内把 road 层车道线在 ±2m 横移 / ±10° yaw novel-view 下的模糊变形显著减弱，5k smoke novel-view raw PSNR 从 14.4 dB → ≥ 16.5 dB（+2.0 dB），守护 cc_psnr_masked ≥ 24 dB。

**Architecture:** 在 3DGRUT2 现有分层高斯 (LayeredGaussians + LayeredMCMCStrategy) 基础上，对 road 层施加 4 层"轻量级"约束（零 rasterizer 修改）：
1. **SH 降阶**：road 层 max SH degree 从 3 → 1，靠扩展 `LayerSpec.sh_degree` + 在 `LayeredGaussians.create_layer_params` 注入 per-layer SH 维度。
2. **Scale 上限 + anisotropy clamp**：road 层 scale 在 MCMC 步后被 in-place clamp（XY ≤ 0.3m、Z ≤ 0.05m、max/min ≤ 8x）。
3. **Effective Rank 正则**：road 高斯的 log scale eigenvalue 谱熵 reg，惩罚针状高斯。
4. **Virtual-view depth-TV 正则**：每 K iter 在 ±0.5m 随机虚拟相机上渲染 road region 深度，加 TV 平滑 reg。

每个改动都是 loss / clamp / config，零 CUDA kernel 修改。

**Nvidia Nurec 外部实证背书**（详见调研报告 §8.6）：3dgrut2 现有 road 层已无意识地吸收 Nurec 6/8 个 road-specific 参数（`scale_prior=(0.1,0.1,0.001)`、`max_n_particles=200K`、`mask_field=road_mask`、`scale_lr_mult=0.2`、`perturb_scale_mask=(1,1,0)` 锁 Z）。Phase 1 的 P1.1（SH 降阶）正是补 Nurec `fourier_features_dim=1 vs bg 5` 的 view-dep 表达力缺口；P1.2（scale clamp）正是补"运行期保持扁平"——Nurec 文档原话：「**体积型粒子无法在地表实现亚像素级清晰度**」。P1.1 + P1.2 从"理论推断"升级为"外部实证强背书"，不可互相替代。

**Tech Stack:** Python 3.10 + PyTorch + Hydra/OmegaConf + 3DGRUT CUDA rasterizer (unchanged) + pytest。

**关联文档:** [调研报告](road-gaussian-background-zesty-dragonfly.md), [v3_plan.md](/Users/etendue/repo/3dgrut2/v3_plan.md), [v2_architecture.md](/Users/etendue/repo/3dgrut2/v2_architecture.md)

---

## File Structure

### Files to CREATE

| 路径 | 责任 | ~LoC |
|---|---|---|
| `threedgrut/model/road_reg.py` | 纯函数：effective rank loss、scale clamp、depth-TV loss | ~80 |
| `threedgrut/tests/test_road_reg.py` | 上述纯函数的单测 | ~120 |
| `threedgrut/tests/test_road_sh_degree.py` | road 层 SH 降阶集成测试 | ~60 |
| `threedgrut/tests/test_road_scale_clamp.py` | scale clamp 与 MCMC 集成测试 | ~80 |

### Files to MODIFY

| 路径 | 改动 | 说明 |
|---|---|---|
| `threedgrut/layers/layer_spec.py` | 加 4 个 optional 字段 | `sh_degree`, `scale_xy_max`, `scale_z_max`, `anisotropy_ratio_max` |
| `threedgrut/layers/registry.py:27-32` | road LayerSpec 填新字段 | `sh_degree=1, scale_xy_max=0.3, scale_z_max=0.05, anisotropy_ratio_max=8.0` |
| `threedgrut/layers/layered_model.py:1148` | per-layer SH dim 来自 spec | `sh_degree_to_specular_dim(spec.sh_degree or layer.max_n_features)` |
| `threedgrut/layers/layered_strategy.py` | post-optimizer hook 调 `clamp_layer_scales` | 每个 road sub 层调一次 |
| `threedgrut/trainer.py:1052-1072` (`get_losses`) | 加 road effective rank loss + depth-TV loss | 通过 `trainer.lambda_road_*` 开关 |
| `threedgrut/trainer.py:1700-1730` (`run_train_iter`) | 每 K iter 渲染虚拟视角 + 计算 depth-TV | 通过 `trainer.virtual_view_frequency` 开关 |
| `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` | 加 trainer.lambda_road_eff_rank / virtual_view_frequency 等 | 与现有 dynfix 兼容 |
| `configs/base_gs.yaml` | trainer 段加新字段默认值（off） | 保证 v1 byte-identical |

### Files NOT touched (确保零 rasterizer 改动)

- `threedgrut/strategy/src/` (CUDA 源)
- `threedgrut/render/` (渲染管线)
- `threedgrut/model/model.py` (MoG 核心)
- gsplat / CUDA 内核

---

## Task 1: Add `sh_degree` field to LayerSpec

> **Why `sh_degree=1` for road**: Nvidia Nurec pipeline 实证 `fourier_features_dim=1` (vs background 5) 已足以表达路面 view-dep 颜色，避免高对比黑白车道线在 SH 高阶系数上 overfit 训练视角。SH degree 1 = DC + 3 linear coef，与 Nurec 1-dim Fourier 概念等价。详见 [研究报告 §8.6](road-gaussian-background-zesty-dragonfly.md)。

**Files:**
- Modify: `threedgrut/layers/layer_spec.py`
- Test: `threedgrut/tests/test_road_sh_degree.py` (new)

- [ ] **Step 1.1: Write the failing test**

新建 `threedgrut/tests/test_road_sh_degree.py`：

```python
# SPDX-License-Identifier: Apache-2.0
"""V3-R1.1 unit tests for per-layer SH degree override via LayerSpec.sh_degree."""
from __future__ import annotations

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS


def test_layerspec_default_sh_degree_is_none():
    """sh_degree defaults to None → use global progressive_training.max_n_features."""
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100)
    assert spec.sh_degree is None


def test_layerspec_accepts_explicit_sh_degree():
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100, sh_degree=1)
    assert spec.sh_degree == 1


def test_road_layer_sh_degree_is_1():
    """V3-R1.1 acceptance: road layer caps SH at degree 1 (DC + 3 linear)."""
    assert STANDARD_LAYERS["road"].sh_degree == 1


def test_background_layer_sh_degree_default():
    """Background layer keeps default (None = use global)."""
    assert STANDARD_LAYERS["background"].sh_degree is None
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd /Users/etendue/repo/3dgrut2 && source .venv/bin/activate
pytest threedgrut/tests/test_road_sh_degree.py -v
```

Expected: 4 个 test 全部 FAIL — `LayerSpec.__init__() got an unexpected keyword argument 'sh_degree'`

- [ ] **Step 1.3: Add `sh_degree` to LayerSpec**

Edit `threedgrut/layers/layer_spec.py`，在 `perturb_scale_mask` 行后插入：

```python
    # V3-R1.1: per-layer SH degree cap. None = use global
    # conf.model.progressive_training.max_n_features. Road layer uses 1
    # (DC + 3 linear) so the high-contrast lane-marking color does not
    # over-fit to the training camera frustum and degrade under ±2m
    # lateral / ±10° yaw novel-view perturbation.
    sh_degree: int | None = None
```

Then edit `threedgrut/layers/registry.py:27-32` — road entry 加 `sh_degree=1`：

```python
    "road": LayerSpec(
        name="road", layer_id=1, max_n_particles=200_000,
        scale_prior=(0.1, 0.1, 0.001), scale_lr_mult=0.2,
        mask_field="road_mask",
        perturb_scale_mask=(1.0, 1.0, 0.0),  # T3.4 D1: Z lock during MCMC perturb
        sh_degree=1,  # V3-R1.1: reduce view-dep overfit on lane markings
    ),
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
pytest threedgrut/tests/test_road_sh_degree.py -v
```

Expected: 4 个 test 全部 PASS

- [ ] **Step 1.5: Verify no regression in existing tests**

```bash
pytest threedgrut/tests/test_layered_loss.py threedgrut/tests/test_novel_view.py -v
```

Expected: 全部 PASS（未触动现有 LayerSpec 字段）

- [ ] **Step 1.6: Commit**

```bash
git add threedgrut/layers/layer_spec.py threedgrut/layers/registry.py threedgrut/tests/test_road_sh_degree.py
git commit -m "$(cat <<'EOF'
feat(V3-R1.1): add sh_degree field to LayerSpec; road defaults to 1

LayerSpec gains optional sh_degree (default None = use global). Registry
road entry sets sh_degree=1 (DC + 3 linear) to reduce view-dependent SH
overfit on high-contrast lane markings under novel-view perturbation.

This commit only carries the spec/registry plumbing; threading into
LayeredGaussians.create_layer_params lands in Task 2.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Thread per-layer SH degree through LayeredGaussians

**Files:**
- Modify: `threedgrut/layers/layered_model.py:1148`
- Modify: `threedgrut/tests/test_road_sh_degree.py` (extend with integration test)

- [ ] **Step 2.1: Extend the test file**

Append to `threedgrut/tests/test_road_sh_degree.py`:

```python
import torch

from threedgrut.utils.misc import sh_degree_to_specular_dim


def test_create_layer_params_uses_spec_sh_degree_when_set(monkeypatch):
    """V3-R1.1: when spec.sh_degree is set, features_specular dim matches
    sh_degree_to_specular_dim(spec.sh_degree) not layer.max_n_features.
    """
    # Build a CPU stub LayeredGaussians via the same path
    # diagnose_bg_in_cuboid.py uses (memory: obs 1107, May 26).
    # Mock-build a single road layer; assert features_specular shape.
    from omegaconf import OmegaConf
    from threedgrut.layers.layered_model import LayeredGaussians
    from threedgrut.layers.registry import specs_from_config

    conf = OmegaConf.create({
        "layers": {"enabled": ["road"]},
        "use_layered_model": True,
        "model": {
            "progressive_training": {
                "feature_type": "sh", "init_n_features": 0,
                "max_n_features": 3,  # global default = 3
                "increase_frequency": 1000, "increase_step": 1,
            },
            "default_density": 0.1, "default_scale_factor": 1.0,
            "density_activation": "sigmoid", "scale_activation": "exp",
            "optimize_density": True, "optimize_features_albedo": True,
            "optimize_features_specular": True, "optimize_position": True,
            "optimize_rotation": True, "optimize_scale": True,
            "bvh_update_frequency": 1, "print_stats": False,
        },
        "render": {"particle_radiance_sph_degree": 3},
        "optimizer": {"type": "adam", "lr": 0.0, "eps": 1e-15,
                      "params": {"positions": {"lr": 1e-4},
                                 "density": {"lr": 1e-2},
                                 "features_albedo": {"lr": 1e-3},
                                 "features_specular": {"lr": 5e-5},
                                 "rotation": {"lr": 1e-3},
                                 "scale": {"lr": 5e-3}}},
    })
    specs = specs_from_config(conf)
    lg = LayeredGaussians(conf=conf, specs=specs, device="cpu")

    N = 16
    positions = torch.randn(N, 3)
    lg.create_layer_params("road", positions, setup_optimizer=False)

    road_layer = lg.layers["road"]
    expected_specular = sh_degree_to_specular_dim(1)  # spec.sh_degree=1
    assert road_layer.features_specular.shape == (N, expected_specular), (
        f"road features_specular dim = {road_layer.features_specular.shape}, "
        f"expected ({N}, {expected_specular}) for sh_degree=1"
    )
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
pytest threedgrut/tests/test_road_sh_degree.py::test_create_layer_params_uses_spec_sh_degree_when_set -v
```

Expected: FAIL — `features_specular dim = (16, 45), expected (16, 9)` (现状是用 layer.max_n_features=3 → 45 维)

- [ ] **Step 2.3: Implement spec-driven SH dim in create_layer_params**

Edit `threedgrut/layers/layered_model.py:1148` — 把：

```python
        features_albedo = (colors.to(dtype=dtype, device=device) - 0.5) / _SH_C0
        num_specular_dims = sh_degree_to_specular_dim(layer.max_n_features)
```

改为：

```python
        features_albedo = (colors.to(dtype=dtype, device=device) - 0.5) / _SH_C0
        # V3-R1.1: per-layer SH cap via spec.sh_degree (None = inherit global).
        # Road layer uses sh_degree=1 to reduce view-dep overfit on lane
        # markings under novel-view perturbation (±2m lateral / ±10° yaw).
        effective_sh = spec.sh_degree if spec.sh_degree is not None else layer.max_n_features
        num_specular_dims = sh_degree_to_specular_dim(effective_sh)
```

- [ ] **Step 2.4: Run integration test to verify it passes**

```bash
pytest threedgrut/tests/test_road_sh_degree.py -v
```

Expected: 5 个 test 全部 PASS

- [ ] **Step 2.5: Run full regression suite**

```bash
pytest threedgrut/tests/ -v --timeout=120
```

Expected: 全部 PASS（≥ 58 个现有 test 不退化）

- [ ] **Step 2.6: Commit**

```bash
git add threedgrut/layers/layered_model.py threedgrut/tests/test_road_sh_degree.py
git commit -m "$(cat <<'EOF'
feat(V3-R1.1): thread spec.sh_degree through LayeredGaussians.create_layer_params

When LayerSpec.sh_degree is set (road layer = 1), features_specular is
allocated with sh_degree_to_specular_dim(spec.sh_degree) instead of
layer.max_n_features. Other layers (background, dynamic_rigids) keep
sh_degree=None → inherit conf.model.progressive_training.max_n_features
(byte-identical with pre-V3-R1 path).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add scale clamp + anisotropy field to LayerSpec

> **Why scale clamp for road**: Nurec 文档明示「每个高斯被压扁成一个贴地的扁平定向圆盘。这是车道线、井盖、路面补丁在新视角下保持锐利的根本原因 —— 体积型粒子无法在地表实现亚像素级清晰度」。3dgrut2 现有 `scale_prior=(0.1, 0.1, 0.001)` 仅保证**初始**扁盘，但 MCMC perturb 中 XY 无上限会让粒子在平面内"胖化"。本任务通过 LayerSpec 新增的 3 个字段实现**运行期持续压扁**，与 Nurec 设计意图对齐。详见 [研究报告 §8.6](road-gaussian-background-zesty-dragonfly.md)。

**Files:**
- Modify: `threedgrut/layers/layer_spec.py`
- Modify: `threedgrut/layers/registry.py:27-32`
- Test: `threedgrut/tests/test_road_scale_clamp.py` (new)

- [ ] **Step 3.1: Write the failing test for spec fields**

New file `threedgrut/tests/test_road_scale_clamp.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""V3-R1.2 unit tests for per-layer scale clamp + anisotropy via LayerSpec."""
from __future__ import annotations

import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.layers.registry import STANDARD_LAYERS


def test_layerspec_default_scale_clamps_are_none():
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=100)
    assert spec.scale_xy_max is None
    assert spec.scale_z_max is None
    assert spec.anisotropy_ratio_max is None


def test_road_layer_clamps_set():
    """V3-R1.2 acceptance: road layer caps scale (XY ≤ 0.3m, Z ≤ 0.05m)
    and anisotropy ratio (max/min eigenvalue ≤ 8x)."""
    s = STANDARD_LAYERS["road"]
    assert s.scale_xy_max == 0.3
    assert s.scale_z_max == 0.05
    assert s.anisotropy_ratio_max == 8.0


def test_background_layer_clamps_not_set():
    s = STANDARD_LAYERS["background"]
    assert s.scale_xy_max is None
    assert s.scale_z_max is None
    assert s.anisotropy_ratio_max is None
```

- [ ] **Step 3.2: Verify failures**

```bash
pytest threedgrut/tests/test_road_scale_clamp.py -v
```

Expected: 3 FAIL — `LayerSpec.__init__() got an unexpected keyword argument 'scale_xy_max'`

- [ ] **Step 3.3: Add the 3 fields to LayerSpec**

Append in `threedgrut/layers/layer_spec.py` after `sh_degree`:

```python
    # V3-R1.2: per-layer scale upper bounds applied as in-place clamp
    # after every MCMC post_optimizer_step. Physical units (meters in log
    # space → exp). None disables. Road layer uses (0.3, 0.05) to keep
    # lane-marking-sized particles tight; XY 0.3m ≈ lane-stripe width × 2,
    # Z 0.05m keeps the disc thin so it stays on the LiDAR-Z surface.
    scale_xy_max: float | None = None
    scale_z_max: float | None = None
    # V3-R1.2: per-layer anisotropy ratio cap (max scale eigenvalue /
    # min scale eigenvalue). Prevents needle-shaped Gaussians that
    # overfit to a single training-camera direction. None disables.
    # Road layer uses 8.0 — generous enough for elongated lane stripes
    # yet bounded enough to suppress hair-thin novel-view artifacts.
    anisotropy_ratio_max: float | None = None
```

- [ ] **Step 3.4: Update registry road entry**

Edit `threedgrut/layers/registry.py:27-32` — append the new fields:

```python
    "road": LayerSpec(
        name="road", layer_id=1, max_n_particles=200_000,
        scale_prior=(0.1, 0.1, 0.001), scale_lr_mult=0.2,
        mask_field="road_mask",
        perturb_scale_mask=(1.0, 1.0, 0.0),  # T3.4 D1: Z lock during MCMC perturb
        sh_degree=1,                          # V3-R1.1
        scale_xy_max=0.3, scale_z_max=0.05,   # V3-R1.2
        anisotropy_ratio_max=8.0,             # V3-R1.2
    ),
```

- [ ] **Step 3.5: Verify spec tests pass**

```bash
pytest threedgrut/tests/test_road_scale_clamp.py -v
```

Expected: 3 PASS

- [ ] **Step 3.6: Commit (no behavior change yet)**

```bash
git add threedgrut/layers/layer_spec.py threedgrut/layers/registry.py threedgrut/tests/test_road_scale_clamp.py
git commit -m "$(cat <<'EOF'
feat(V3-R1.2): add scale_xy_max/scale_z_max/anisotropy_ratio_max to LayerSpec

Road LayerSpec gains 3 optional clamp bounds. Spec/registry only — the
clamp hook itself lands in Task 4. Other layers keep all 3 as None →
no behavior change (byte-identical).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Pure function `clamp_layer_scales` + MCMC post-step hook

**Files:**
- Create: `threedgrut/model/road_reg.py`
- Create: `threedgrut/tests/test_road_reg.py`
- Modify: `threedgrut/layers/layered_strategy.py` (call hook after post-optimizer step)

- [ ] **Step 4.1: Write the failing test for `clamp_layer_scales`**

New file `threedgrut/tests/test_road_reg.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""V3-R1 unit tests for road_reg pure functions."""
from __future__ import annotations

import math

import pytest
import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.road_reg import clamp_layer_scales


def _make_road_spec(**kwargs):
    base = dict(
        name="road", layer_id=1, max_n_particles=100,
        scale_xy_max=0.3, scale_z_max=0.05, anisotropy_ratio_max=8.0,
    )
    base.update(kwargs)
    return LayerSpec(**base)


def test_clamp_no_op_when_spec_has_no_clamps():
    """When all 3 clamp fields are None, scale_log returned unchanged."""
    spec = LayerSpec(name="x", layer_id=0, max_n_particles=10)
    scale_log = torch.randn(10, 3)
    out = clamp_layer_scales(scale_log, spec)
    assert torch.equal(out, scale_log)


def test_clamp_xy_upper_bound():
    """XY scales above scale_xy_max get clamped down; Z untouched if no Z max."""
    spec = LayerSpec(name="r", layer_id=1, max_n_particles=10, scale_xy_max=0.3)
    # log(1.0) = 0 → exp = 1.0m, well above 0.3m
    scale_log = torch.zeros(4, 3)
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    assert torch.all(out_exp[:, 0] <= 0.3 + 1e-6)
    assert torch.all(out_exp[:, 1] <= 0.3 + 1e-6)
    # Z untouched — input exp = 1.0
    assert torch.allclose(out_exp[:, 2], torch.tensor(1.0))


def test_clamp_z_upper_bound():
    spec = LayerSpec(name="r", layer_id=1, max_n_particles=10, scale_z_max=0.05)
    scale_log = torch.zeros(4, 3)  # exp = 1.0 all
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    assert torch.all(out_exp[:, 2] <= 0.05 + 1e-6)


def test_clamp_anisotropy_ratio():
    """When max/min > ratio_max, the smallest axis gets raised."""
    spec = _make_road_spec(scale_xy_max=None, scale_z_max=None,
                            anisotropy_ratio_max=4.0)
    # log scale = [log(1.0), log(1.0), log(0.05)] → ratio 1.0/0.05 = 20x
    scale_log = torch.tensor([[0.0, 0.0, math.log(0.05)]])
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    ratio = out_exp.max(dim=-1).values / out_exp.min(dim=-1).values
    assert torch.all(ratio <= 4.0 + 1e-5), f"ratio {ratio} > 4.0"


def test_clamp_combined_xy_z_anisotropy_road():
    """End-to-end road spec: XY≤0.3, Z≤0.05, ratio≤8."""
    spec = _make_road_spec()
    # Mix: some too-big XY, some too-small Z, some both
    scale_log = torch.log(torch.tensor([
        [0.5, 0.5, 0.001],   # XY too big; ratio 500x
        [0.2, 0.2, 0.04],    # OK
        [0.1, 0.1, 0.5],     # Z too big
    ]))
    out = clamp_layer_scales(scale_log, spec)
    out_exp = torch.exp(out)
    assert torch.all(out_exp[:, :2] <= 0.3 + 1e-6)
    assert torch.all(out_exp[:, 2] <= 0.05 + 1e-6)
    ratio = out_exp.max(dim=-1).values / out_exp.min(dim=-1).values
    assert torch.all(ratio <= 8.0 + 1e-5), f"ratios {ratio}"


def test_clamp_returns_same_dtype_device():
    spec = _make_road_spec()
    scale_log = torch.zeros(4, 3, dtype=torch.float32)
    out = clamp_layer_scales(scale_log, spec)
    assert out.dtype == torch.float32
    assert out.device == scale_log.device
```

- [ ] **Step 4.2: Verify failures**

```bash
pytest threedgrut/tests/test_road_reg.py -v
```

Expected: All 6 FAIL — `ModuleNotFoundError: No module named 'threedgrut.model.road_reg'`

- [ ] **Step 4.3: Implement `road_reg.py`**

New file `threedgrut/model/road_reg.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""V3-R1 road-layer regularizers (pure functions).

Three orthogonal regularizers used by V3-R1 Phase 1 to suppress
lane-marking blur and deformation under novel-view perturbation:

1. ``clamp_layer_scales`` — in-place scale upper bounds + anisotropy
   ratio cap applied AFTER MCMC post-optimizer step (V3-R1.2).
2. ``compute_effective_rank_loss`` — entropy-of-scale-spectrum penalty
   that softens needle-shaped Gaussians (V3-R1.3).
3. ``compute_depth_tv_loss`` — total-variation smoothness on a rendered
   depth map restricted to a road mask, used at virtual viewpoints
   (V3-R1.4).

Pure functions / no Trainer / no CUDA — safe to unit-test on Mac CPU.
"""
from __future__ import annotations

from typing import Optional

import torch

from threedgrut.layers.layer_spec import LayerSpec


def clamp_layer_scales(scale_log: torch.Tensor, spec: LayerSpec) -> torch.Tensor:
    """Clamp a per-particle log-space scale tensor by the per-layer bounds.

    Args:
        scale_log: ``[N, 3]`` log-space scale parameter (model.scale).
        spec: layer descriptor; reads ``scale_xy_max`` / ``scale_z_max`` /
            ``anisotropy_ratio_max``. Any field that is None disables that
            clamp.

    Returns:
        ``[N, 3]`` clamped log-space scale (same dtype/device as input).

    Notes:
        - XY/Z clamps are absolute upper bounds in physical units (exp(log)).
        - Anisotropy clamp raises the smallest eigenvalue if max/min >
          ratio_max, leaving the largest untouched. This biases toward
          larger Gaussians rather than shrinking the in-plane extent.
        - All three clamps are applied in order: XY → Z → ratio.
    """
    if (
        spec.scale_xy_max is None
        and spec.scale_z_max is None
        and spec.anisotropy_ratio_max is None
    ):
        return scale_log

    out = scale_log.clone()

    if spec.scale_xy_max is not None:
        cap = float(torch.log(torch.tensor(spec.scale_xy_max)).item())
        out[:, 0].clamp_(max=cap)
        out[:, 1].clamp_(max=cap)

    if spec.scale_z_max is not None:
        cap = float(torch.log(torch.tensor(spec.scale_z_max)).item())
        out[:, 2].clamp_(max=cap)

    if spec.anisotropy_ratio_max is not None:
        # Work in physical space for the ratio check.
        s = torch.exp(out)
        s_max, _ = s.max(dim=-1, keepdim=True)
        floor = s_max / float(spec.anisotropy_ratio_max)
        s = torch.maximum(s, floor)
        out = torch.log(s.clamp_min(1e-12))

    return out


def compute_effective_rank_loss(
    scale_log: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Spectral-entropy regularizer encouraging isotropic-ish Gaussians.

    For each particle, normalize its 3 scale eigenvalues to a probability
    simplex and compute Shannon entropy. Loss = -entropy.mean() so that
    minimizing → push entropy up → push Gaussians toward isotropy.

    Args:
        scale_log: ``[N, 3]`` log-scale parameter.
        mask: optional ``[N]`` bool/float mask selecting which particles
            contribute (e.g. road layer only). None = all particles.

    Returns:
        Scalar tensor on the same device/dtype as scale_log.
    """
    s = torch.exp(scale_log)  # [N, 3]
    s = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(s * (s.clamp_min(1e-12).log())).sum(dim=-1)  # [N]
    if mask is not None:
        m = mask.to(entropy.dtype)
        denom = m.sum().clamp_min(1.0)
        return -(entropy * m).sum() / denom
    return -entropy.mean()


def compute_depth_tv_loss(
    depth: torch.Tensor,
    road_mask: torch.Tensor,
    min_pixels: int = 100,
) -> torch.Tensor:
    """Total-variation smoothness on a rendered depth map, road-region only.

    Args:
        depth: ``[B, H, W]`` or ``[H, W]`` rendered depth (meters).
        road_mask: same spatial shape, binary {0, 1}.
        min_pixels: when road_mask.sum() < min_pixels, returns 0
            (graceful no-op for edge frames with no road).

    Returns:
        Scalar TV loss in meters, normalized by the number of road
        boundary pairs.
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        road_mask = road_mask.unsqueeze(0)
    rm = road_mask.to(depth.dtype)
    if rm.sum().item() < min_pixels:
        return torch.zeros((), device=depth.device, dtype=depth.dtype)

    # Horizontal & vertical neighbor pair masks: both endpoints inside road.
    pair_h = rm[:, :, :-1] * rm[:, :, 1:]
    pair_v = rm[:, :-1, :] * rm[:, 1:, :]

    diff_h = (depth[:, :, :-1] - depth[:, :, 1:]).abs() * pair_h
    diff_v = (depth[:, :-1, :] - depth[:, 1:, :]).abs() * pair_v

    num = diff_h.sum() + diff_v.sum()
    den = pair_h.sum() + pair_v.sum() + 1e-6
    return num / den
```

- [ ] **Step 4.4: Verify pure-function tests pass**

```bash
pytest threedgrut/tests/test_road_reg.py -v
```

Expected: 6 PASS

- [ ] **Step 4.5: Add MCMC integration test**

Append to `threedgrut/tests/test_road_scale_clamp.py`:

```python
import torch

from threedgrut.layers.layer_spec import LayerSpec
from threedgrut.model.road_reg import clamp_layer_scales


def test_clamp_is_inplace_safe_on_param():
    """clamp_layer_scales returns a fresh tensor; doesn't mutate input."""
    spec = LayerSpec(name="r", layer_id=1, max_n_particles=10,
                      scale_xy_max=0.3, scale_z_max=0.05,
                      anisotropy_ratio_max=8.0)
    scale_log = torch.zeros(4, 3, requires_grad=True)
    out = clamp_layer_scales(scale_log, spec)
    # Input untouched (so nn.Parameter detection still works in caller)
    assert torch.all(scale_log == 0.0)
    # Output is clamped
    assert (torch.exp(out) <= 0.3 + 1e-6).all() or (torch.exp(out)[:, 2] <= 0.05 + 1e-6).all()
```

- [ ] **Step 4.6: Wire clamp into LayeredMCMCStrategy**

Locate the post-optimizer-step hook in `threedgrut/layers/layered_strategy.py` (search for `_post_optimizer_step` in that file). Add at the end of the layered post-step loop:

```python
# V3-R1.2: in-place clamp road-layer scale params after MCMC perturb /
# relocate. Other layers (background, dynamic_rigids) have all 3 spec
# clamp fields = None → clamp_layer_scales is a no-op.
from threedgrut.model.road_reg import clamp_layer_scales
for spec in self.specs:
    if not spec.is_particle_layer:
        continue
    sub_strategy = self._sub_strategies[spec.name]  # field name must match
    sub_model = sub_strategy.model
    if (
        spec.scale_xy_max is not None
        or spec.scale_z_max is not None
        or spec.anisotropy_ratio_max is not None
    ):
        with torch.no_grad():
            sub_model.scale.copy_(
                clamp_layer_scales(sub_model.scale.detach(), spec)
            )
```

(Implementation note: the executing agent must verify the exact attribute names — `_sub_strategies` may be `self.sub_strategies` or stored on `self.model.layers`. Read the file first; the pattern above mirrors how `perturb_scale_mask` is plumbed in `MCMCStrategy._get_perturb_mask`.)

- [ ] **Step 4.7: Run full regression**

```bash
pytest threedgrut/tests/ -v --timeout=120
```

Expected: All existing tests PASS + new tests PASS

- [ ] **Step 4.8: Commit**

```bash
git add threedgrut/model/road_reg.py threedgrut/tests/test_road_reg.py threedgrut/tests/test_road_scale_clamp.py threedgrut/layers/layered_strategy.py
git commit -m "$(cat <<'EOF'
feat(V3-R1.2): road scale clamp + anisotropy hook in LayeredMCMCStrategy

New pure function clamp_layer_scales applies per-layer scale upper bound
and anisotropy ratio cap. LayeredMCMCStrategy invokes it after every
post-optimizer step for layers whose spec has any of the 3 clamp fields
set. Road layer gets (XY≤0.3m, Z≤0.05m, ratio≤8x); other layers stay
no-op (byte-identical).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Effective Rank loss wired into Trainer.get_losses

**Files:**
- Modify: `threedgrut/trainer.py:931-1072` (`get_losses`)
- Modify: `configs/base_gs.yaml` (add `trainer.lambda_road_eff_rank` default 0.0)
- Modify: `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` (set `lambda_road_eff_rank: 0.01`)
- Test: extend `threedgrut/tests/test_road_reg.py`

- [ ] **Step 5.1: Write the failing test for effective rank loss**

Append to `threedgrut/tests/test_road_reg.py`:

```python
from threedgrut.model.road_reg import compute_effective_rank_loss


def test_effective_rank_loss_isotropic_minimum():
    """Isotropic Gaussians (s_x == s_y == s_z) attain MIN entropy loss."""
    iso = torch.zeros(4, 3)  # log(1,1,1) -> isotropic
    needle = torch.tensor([[0.0, 0.0, -5.0]] * 4)  # huge anisotropy
    L_iso = compute_effective_rank_loss(iso)
    L_needle = compute_effective_rank_loss(needle)
    assert L_iso.item() < L_needle.item(), (
        f"isotropic loss {L_iso.item()} should be < needle loss {L_needle.item()}"
    )


def test_effective_rank_loss_mask_selects_subset():
    """Mask selects which particles contribute."""
    log_scale = torch.zeros(10, 3)
    # Make particles 5..9 needles
    log_scale[5:, 2] = -5.0
    mask_isos = torch.tensor([1.0] * 5 + [0.0] * 5)
    mask_needles = torch.tensor([0.0] * 5 + [1.0] * 5)
    L_iso = compute_effective_rank_loss(log_scale, mask=mask_isos)
    L_needle = compute_effective_rank_loss(log_scale, mask=mask_needles)
    assert L_iso.item() < L_needle.item()


def test_effective_rank_loss_grad_flows():
    log_scale = torch.zeros(4, 3, requires_grad=True)
    L = compute_effective_rank_loss(log_scale)
    L.backward()
    assert log_scale.grad is not None
```

- [ ] **Step 5.2: Verify pass (function already implemented in Task 4)**

```bash
pytest threedgrut/tests/test_road_reg.py -v
```

Expected: All PASS (9 tests now)

- [ ] **Step 5.3: Add config default in `configs/base_gs.yaml`**

Inside `trainer:` block (around the `bg_dyn_cuboid_penalty` block, line ~63), add:

```yaml
  # V3-R1.3: effective rank reg on road layer scales. Penalizes needle-
  # shaped Gaussians that overfit to a single training-camera direction
  # and break under novel-view ±2m / ±10° perturbation. 0.0 = off
  # (v2 byte-identical). Typical: 0.005 - 0.02.
  lambda_road_eff_rank: 0.0
```

- [ ] **Step 5.4: Wire into Trainer.get_losses**

Edit `threedgrut/trainer.py` — at top of file with other model imports:

```python
from threedgrut.model.road_reg import compute_effective_rank_loss
```

Inside `get_losses` (after `loss_pose_smooth = self._compute_pose_smoothness_term(...)` around line 1050), insert:

```python
        # V3-R1.3: effective-rank reg on road-layer scales (suppresses
        # needle-shaped lane-marking Gaussians that overfit to training
        # camera direction).
        loss_road_eff_rank = torch.zeros(1, device=self.device)
        lambda_road_eff_rank = float(
            trainer_conf.get("lambda_road_eff_rank", 0.0) if hasattr(trainer_conf, "get")
            else getattr(trainer_conf, "lambda_road_eff_rank", 0.0)
        )
        if lambda_road_eff_rank > 0.0 and hasattr(self.model, "layers") and "road" in self.model.layers:
            with torch.cuda.nvtx.range("loss-road-eff-rank"):
                loss_road_eff_rank = compute_effective_rank_loss(
                    self.model.layers["road"].scale
                )
```

Then update the `loss = (...)` sum and `return dict(...)`:

```python
        loss = (
            lambda_l1 * loss_l1
            + lambda_ssim * loss_ssim
            + lambda_opacity * loss_opacity
            + lambda_scale * loss_scale
            + lambda_sky * loss_sky
            + loss_bg_cuboid
            + loss_pose_smooth
            + lambda_road_eff_rank * loss_road_eff_rank  # V3-R1.3
        )
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            opacity_loss=lambda_opacity * loss_opacity,
            scale_loss=lambda_scale * loss_scale,
            sky_loss=lambda_sky * loss_sky,
            bg_cuboid_loss=loss_bg_cuboid,
            pose_smooth_loss=loss_pose_smooth,
            road_eff_rank_loss=lambda_road_eff_rank * loss_road_eff_rank,  # V3-R1.3
        )
```

- [ ] **Step 5.5: Enable in multilayer config**

Edit `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`, inside the `trainer:` block, add:

```yaml
  # V3-R1.3 — road-layer effective rank reg (active, V3-R1 Phase 1).
  lambda_road_eff_rank: 0.01
```

- [ ] **Step 5.6: Mac 1k smoke (verify training still runs)**

```bash
cd /Users/etendue/repo/3dgrut2 && source .venv/bin/activate
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
  n_iterations=200 \
  path=<test_clip_path>/pai_<clip>.json \
  trainer.sky_backend=mlp \
  experiment_name=v3r1_p1_3_smoke_mac \
  2>&1 | tee /tmp/v3r1_p1_3_mac.log
```

Look for `road_eff_rank_loss` in the training log every iter. If grep returns hits, wiring works.

- [ ] **Step 5.7: Commit**

```bash
git add threedgrut/trainer.py configs/base_gs.yaml configs/apps/ncore_3dgut_mcmc_multilayer.yaml threedgrut/tests/test_road_reg.py
git commit -m "$(cat <<'EOF'
feat(V3-R1.3): road effective-rank reg loss wired into trainer

Add compute_effective_rank_loss tests; expose lambda_road_eff_rank in
configs/base_gs.yaml (default 0.0) and configs/apps/multilayer.yaml
(active at 0.01). Trainer.get_losses computes the entropy penalty on
LayeredGaussians.layers["road"].scale when lambda > 0; the loss is
included in total_loss and surfaced in the loss dict for logging.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Virtual-view depth-TV regularization

**Files:**
- Modify: `threedgrut/trainer.py:1700-1730` (`run_train_iter`)
- Modify: `configs/base_gs.yaml` + `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`
- Test: extend `threedgrut/tests/test_road_reg.py`

- [ ] **Step 6.1: Write the failing test for depth-TV loss**

Append to `threedgrut/tests/test_road_reg.py`:

```python
from threedgrut.model.road_reg import compute_depth_tv_loss


def test_depth_tv_zero_for_constant_depth():
    """Flat road plane → TV loss = 0."""
    depth = torch.ones(1, 8, 8) * 10.0
    road = torch.ones(1, 8, 8)
    L = compute_depth_tv_loss(depth, road, min_pixels=1)
    assert L.item() == pytest.approx(0.0, abs=1e-6)


def test_depth_tv_positive_for_high_freq():
    """Checkerboard depth → positive TV in road region."""
    depth = torch.zeros(1, 4, 4)
    depth[:, ::2, ::2] = 10.0
    depth[:, 1::2, 1::2] = 10.0
    road = torch.ones(1, 4, 4)
    L = compute_depth_tv_loss(depth, road, min_pixels=1)
    assert L.item() > 0.0


def test_depth_tv_respects_mask():
    """TV outside road region must not contribute."""
    depth = torch.zeros(1, 4, 4)
    depth[:, 0, 0] = 1000.0  # huge gradient at (0,0)
    road = torch.zeros(1, 4, 4)
    road[:, 2:, 2:] = 1.0  # road mask doesn't include (0,0)
    L = compute_depth_tv_loss(depth, road, min_pixels=1)
    assert L.item() == pytest.approx(0.0, abs=1e-6)


def test_depth_tv_zero_when_no_road():
    """Empty road mask → 0 (no NaN)."""
    depth = torch.randn(1, 4, 4)
    road = torch.zeros(1, 4, 4)
    L = compute_depth_tv_loss(depth, road, min_pixels=1)
    assert L.item() == 0.0
```

- [ ] **Step 6.2: Verify pass**

```bash
pytest threedgrut/tests/test_road_reg.py -v
```

Expected: ALL PASS (13 tests now)

- [ ] **Step 6.3: Add config defaults**

In `configs/base_gs.yaml` under `trainer:`:

```yaml
  # V3-R1.4: virtual-view depth-TV reg. Every virtual_view_frequency
  # iterations, render road region at a small random pose perturbation
  # and add TV smoothness on the rendered depth. 0.0 = off (default).
  lambda_road_virtual_tv: 0.0
  virtual_view_frequency: 5
  # Magnitude (uniform sample range): lateral ∈ ±lat_max meters,
  # yaw ∈ ±yaw_max degrees. Picked from one or the other each iter.
  virtual_view_lat_max: 0.5
  virtual_view_yaw_max: 2.0
```

- [ ] **Step 6.4: Implement virtual-view render + loss in run_train_iter**

In `threedgrut/trainer.py` add helper method near other private methods:

```python
def _maybe_compute_virtual_view_tv(self, gpu_batch, outputs, trainer_conf):
    """V3-R1.4: every virtual_view_frequency iters, render road region at
    a small random virtual c2w and return a depth-TV loss in the road
    mask. Returns zero tensor when disabled / not on the cadence step /
    no road mask. Pure-eval no-grad render — only the TV loss carries
    grad through the Gaussian params via the implicit dependency on the
    rendered depth (handled by gradient checkpointing of the renderer).
    """
    import random
    import numpy as np
    from threedgrut.utils.novel_view import perturb_shutter_pair_torch
    from threedgrut.model.road_reg import compute_depth_tv_loss

    lam = float(getattr(trainer_conf, "lambda_road_virtual_tv", 0.0) or 0.0)
    freq = int(getattr(trainer_conf, "virtual_view_frequency", 5) or 5)
    if lam <= 0.0 or self.global_step % freq != 0:
        return torch.zeros((), device=self.device)

    image_infos = getattr(gpu_batch, "image_infos", None)
    if image_infos is None or "road_mask" not in image_infos:
        return torch.zeros((), device=self.device)

    # Sample perturbation magnitude
    lat_max = float(getattr(trainer_conf, "virtual_view_lat_max", 0.5))
    yaw_max = float(getattr(trainer_conf, "virtual_view_yaw_max", 2.0))
    use_lateral = random.random() < 0.5
    mag = (random.random() * 2 - 1)  # [-1, 1]

    # Build a one-off perturbed batch via the existing primitive.
    # If lateral_<x>m / yaw_<y>deg ladder is too coarse, switch to
    # the lower-level perturb_c2w with a custom delta.
    if use_lateral:
        mode = "lateral_1m"  # placeholder; below we override magnitude
        scale = lat_max * mag
    else:
        mode = "yaw_5deg"
        scale = yaw_max * mag / 5.0  # scale factor of the 5° step

    # ... render at virtual view, extract depth, compute TV
    # NOTE: the executing agent must (a) verify whether self.model.forward
    # already returns a depth map (3DGUT does — outputs["depth"]) and
    # (b) use the renderer's tensor-friendly entry point. See render.py
    # for the eval-time call pattern (already wired with novel_view=True
    # in V3-E4.1).
    # Pseudocode:
    #     virtual_batch = perturb_batch_dict(gpu_batch, mode, scale)
    #     virt_out = self.model(virtual_batch)  # no grad on photometric
    #     return compute_depth_tv_loss(
    #         virt_out["depth"].squeeze(-1),
    #         image_infos["road_mask"].to(self.device),
    #     )
    raise NotImplementedError(
        "Virtual-view render hook — wire to renderer in Step 6.5"
    )
```

- [ ] **Step 6.5: Concrete renderer wiring**

Read `threedgrut/render.py` for the existing `novel_view=True` path that already calls the renderer with a perturbed c2w (memory: obs 2048 confirms this exists since T9.3). Reuse the same primitive in the trainer:

1. Locate the `novel_view` flag handling in `Renderer.render_all` / `Renderer.from_preloaded_model`.
2. Add a `Trainer._render_at_virtual_view(gpu_batch, mode, scale)` method that calls `perturb_batch_shutter_pair_torch` (existing in `threedgrut/utils/novel_view.py:140-157`) with the sampled magnitude, then runs `self.model(virtual_batch)` under `torch.no_grad()` for the photometric output but **keeps depth differentiable** (depth comes from a separate forward — see `model.forward(..., return_depth=True)` or equivalent).
3. Replace the `NotImplementedError` in `_maybe_compute_virtual_view_tv` with the concrete render + depth extraction.

(If the agent finds that depth gradient isn't easily reachable, an acceptable fallback for Phase 1 is to make the **virtual-view photometric self-consistency** loss instead: render at virtual view with grad, compare to depth-warped original GT in road region. See [`scripts/diagnose_bg_in_cuboid.py`](/Users/etendue/repo/3dgrut2/scripts/diagnose_bg_in_cuboid.py) for the LayeredGaussians CPU-load pattern, memory obs 1107.)

Then wire into `run_train_iter` (around line 1720) before `batch_losses = self.get_losses(...)`:

```python
        # V3-R1.4: virtual-view depth-TV reg (gated by lambda_road_virtual_tv).
        loss_virtual_tv = self._maybe_compute_virtual_view_tv(
            gpu_batch, outputs, trainer_conf
        )
```

And include it in `batch_losses["total_loss"]`:

```python
        if trainer_conf and float(getattr(trainer_conf, "lambda_road_virtual_tv", 0.0) or 0.0) > 0.0:
            lam = float(trainer_conf.lambda_road_virtual_tv)
            batch_losses["total_loss"] = batch_losses["total_loss"] + lam * loss_virtual_tv
            batch_losses["road_virtual_tv_loss"] = lam * loss_virtual_tv
```

- [ ] **Step 6.6: Mac 1k smoke with `lambda_road_virtual_tv: 0.05`**

Add to `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` under `trainer:`:

```yaml
  # V3-R1.4 — virtual-view depth-TV reg (Phase 1).
  lambda_road_virtual_tv: 0.05
  virtual_view_frequency: 5
  virtual_view_lat_max: 0.5
  virtual_view_yaw_max: 2.0
```

Run:

```bash
python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
  n_iterations=200 \
  path=<test_clip>/pai_<clip>.json \
  trainer.sky_backend=mlp \
  experiment_name=v3r1_p1_4_smoke_mac \
  2>&1 | tee /tmp/v3r1_p1_4_mac.log
```

Verify: `grep "road_virtual_tv_loss" /tmp/v3r1_p1_4_mac.log` returns iterations divisible by 5.

- [ ] **Step 6.7: Commit**

```bash
git add threedgrut/trainer.py configs/base_gs.yaml configs/apps/ncore_3dgut_mcmc_multilayer.yaml threedgrut/tests/test_road_reg.py
git commit -m "$(cat <<'EOF'
feat(V3-R1.4): virtual-view depth-TV reg on road region

Every virtual_view_frequency iters, render the current frame at a
random small perturbation (lateral ±0.5m or yaw ±2°) and apply a
total-variation loss on the rendered depth restricted to the road
mask. Reuses perturb_batch_shutter_pair_torch from utils/novel_view.py
(introduced in T8.5.3). Active in multilayer.yaml at lambda=0.05,
freq=5; off by default in base_gs.yaml.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Mac → A800 sync + 5k smoke validation

**Files:**
- None (validation only)

- [ ] **Step 7.1: Push to remote-tracked branch**

```bash
git push origin <current-branch>
```

- [ ] **Step 7.2: Sync to A800 via rsync**

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /Users/etendue/repo/3dgrut2/threedgrut/ \
  a800-x2:/root/work/yusun/repo/3dgrut/threedgrut/
rsync -avz /Users/etendue/repo/3dgrut2/configs/ \
  a800-x2:/root/work/yusun/repo/3dgrut/configs/
```

- [ ] **Step 7.3: Verify code present on remote (per CLAUDE.md §A 严格清单)**

```bash
ssh a800-x2 "grep -n 'V3-R1' /root/work/yusun/repo/3dgrut/threedgrut/layers/registry.py"
ssh a800-x2 "grep -n 'compute_effective_rank_loss' /root/work/yusun/repo/3dgrut/threedgrut/trainer.py"
ssh a800-x2 "head -25 /root/work/yusun/repo/3dgrut/render.py"
```

Expected: all 3 grep return hits; `render.py` head shows `import argparse` (correct top-level entry).

- [ ] **Step 7.4: A800 5k smoke run**

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 \
  && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=5000 \
    path=/root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/pai_<clip>.json \
    trainer.sky_backend=mlp \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=v3r1_p1_full_5k_a800 \
    2>&1 | tee /tmp/v3r1_p1_5k.log'
```

Run in background; the harness will notify on completion. Do NOT poll.

- [ ] **Step 7.5: Inspect metrics**

```bash
ssh a800-x2 "cat /root/work/yusun/ncore-nurec/output/v3r1_p1_full_5k_a800/*/metrics.json"
```

Check the KPI matrix (see § 8.4 of the research report). Required:

| Metric | Baseline (T9.3 5k) | V3-R1 Phase 1 exit |
|---|---|---|
| novel-view raw PSNR | 14.4 dB | **≥ 16.5** |
| novel-view LPIPS | 0.602 | **≤ 0.55** |
| cc_psnr_masked | 26.0 dB | **≥ 24.0** (守护) |
| road particle count | 200K | within ±10% |

- [ ] **Step 7.6: Visual diff against baseline**

Render the 4 novel-view perturbation modes from the new ckpt:

```bash
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && cd /root/work/yusun/repo/3dgrut \
  && python render.py \
    --checkpoint=/root/work/yusun/ncore-nurec/output/v3r1_p1_full_5k_a800/.../ckpt_last.pt \
    --novel_view=true \
    --out_dir=/root/work/yusun/ncore-nurec/output/v3r1_p1_renders'
```

Pull a sample image grid (lateral_2m, yaw_10deg) back to Mac:

```bash
rsync -avz a800-x2:/root/work/yusun/ncore-nurec/output/v3r1_p1_renders/sample_grid.png \
  /tmp/v3r1_p1_renders/
```

Eye-check: car道线边界更锐利、纵向断裂减少。

---

## Task 8: Documentation sync (per CLAUDE.md project rules)

**Files:**
- Modify: `v3_plan.md`
- Modify: `v2_architecture.md`

- [ ] **Step 8.1: Update v3_plan.md kanban**

Find the V3-P section in `v3_plan.md` and insert a new "V3-R1 Road Novel-View Phase 1" subsection between V3-P1 (BilateralGrid, done) and Stage 11 (LiDAR + DepthAnything):

- 4 tasks V3-R1.a / V3-R1.b / V3-R1.c / V3-R1.d listed in the task table with commit short hashes and status ✅.
- Done Log entry with date 2026-05-XX + commit hashes + 5k smoke PSNR numbers.

- [ ] **Step 8.2: Update v2_architecture.md**

Add a node `road_reg.py:::done` in the relevant mermaid diagram (model/ subgraph). Update §6.x file table: `threedgrut/model/road_reg.py` ✅, `LayerSpec` new fields ✅.

- [ ] **Step 8.3: Final commit**

```bash
git add v3_plan.md v2_architecture.md
git commit -m "$(cat <<'EOF'
docs(V3-R1): Phase 1 road novel-view exit — plan + architecture sync

Mark V3-R1.a..d as ✅ in v3_plan.md kanban + Done Log with 5k smoke
metrics. Update v2_architecture.md model/ subgraph to include the new
road_reg module and per-layer SH / scale clamp / anisotropy plumbing.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 与深度监督任务（Stage 11）的关系

**v3_plan.md 现状**（kanban "In Progress 🟡 = 0"）：
- T11.1 V3-T8 trainer 每步 LiDAR ray batch — ⬜ Todo
- T11.2 V3-T9 LiDAR depth/intensity ray loss head — ⬜ Todo
- T11.3 V3-R2 lidar_divergence cone 抗锯齿 — ⬜ Todo
- **T11.4 V3-D1 DepthAnythingV2 metric depth prior reader + depth loss head** — ⬜ Todo
- T11.5 V3-E1 val_lidar=true — LiDAR PSNR 独立报告 — ⬜ Todo
- T11.6 A800 Stage 11 出口 — ⬜ Todo

**Phase 1 与 Stage 11 的执行顺序**：

```
Phase 1 (本 plan, 1 周, +1.5-4.0 dB)
   │   SH 降阶 + scale clamp + effective rank + virtual-view TV
   │   零 LiDAR / 零 DepthAnythingV2 依赖
   ▼
Stage 11 (T11.1-T11.6, 2-3 周, +3.0 dB)
   │   LiDAR ray loss + DepthAnythingV2 depth prior + lidar_divergence
   │   引入几何监督底座
   ▼
Phase 2A — road-normal 扩展 (1-2 周, +1-2 dB)
       │   复用 T11.4 DepthAnythingV2 reader → 加 mono depth → normal head
       │   约束 road 高斯 rotation (normal 朝上)
       │   复用 T11.2 LiDAR ray head → 加局部 plane fitting prior
```

**为什么 Phase 1 不等 Stage 11**：
1. Phase 1 全是 loss/clamp/config 级改动，无 LiDAR/DepthAnythingV2 依赖，可独立 1 周窗口验证
2. Phase 1 的 P1.1（SH 降阶）+ P1.2（scale clamp）是 Nurec 实证背书的"必做"基础设施，**Stage 11 几何监督也需要它们打底**——没有扁平 disc + 低 SH 维度，DepthAnything 法向监督会被高斯的几何漂移抵消
3. Stage 11 引入 LiDAR ray batch 会让训练 it/s 跌（v3_plan §2.3 风险 R3 + R7 已识别），先在 baseline + Phase 1 上跑通再加 Stage 11 风险更低

**潜在合流点**：本 plan 的 Task 6（虚拟视角 depth-TV reg）使用 3DGUT 内部 rendered depth；Stage 11 引入 DepthAnythingV2 后可考虑把 P1.4 升级为"虚拟视角 vs DepthAnythingV2 depth alignment loss"——这属于 Phase 2A 范围，本 plan 不实施。

---

## Self-Review Checklist

**Spec coverage:** ✅ 4 个 phase 1 步骤 (P1.1–P1.4) 各对应至少一个 Task；KPI 验收矩阵在 Task 7 直接对应研究报告 §8.4。

**Placeholder scan:** ⚠️ Task 6 Step 6.5 包含 `NotImplementedError` 占位 — 这是有意的"骨架"留给执行 agent 在读取 `render.py` 实际 novel-view 路径后填实；plan 明确指明读取位置（`Renderer.render_all` + `perturb_batch_shutter_pair_torch`）和 fallback（warp-based photometric loss）。executing agent 必须先读 `render.py` 的 novel-view 实现再实施 Step 6.5。

**Type consistency:** ✅ 所有 `LayerSpec` 新字段类型一致（`int | None` 或 `float | None`），所有 loss 函数返回 `torch.Tensor` 标量，所有 mask 形状约定 `[B, H, W]` 或 `[H, W]`。

**Rollback policy:** 任一步骤 5k smoke 触发守护线 `cc_psnr_masked < 24.0 dB`，立即 git revert 对应 commit，分析后再继续下一步。

---

## Execution Handoff

**Plan complete and saved to `/Users/etendue/.claude/plans/2026-05-28-v3-r1-road-novel-view-phase1.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — 每个 Task 由独立 subagent 执行，主会话做 review，Task 间快速迭代。每个 Task 自包含（创建/修改的文件、测试、commit message 都已列明），适合 fresh subagent。

**2. Inline Execution** — 在当前 session 顺序执行 Task 1 → Task 8，到 Task 7 (A800 smoke) 时切到 background 运行并等待完成通知。

**Which approach?**
