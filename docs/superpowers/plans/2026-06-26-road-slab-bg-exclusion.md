# Road-slab Background Exclusion (A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hard, gradient-free per-step rule that clamps the opacity of background-layer Gaussians inside a thin road slab to ~0, so the frozen road layer becomes the sole renderer of the road surface (fixing road→background "takeover").

**Architecture:** Two pure geometry helpers in `threedgrut/model/road_reg.py` (BEV road-height grid + slab membership), wired into `LayeredMCMCStrategy._post_optimizer_step` via a new `_maybe_exclude_bg_from_road_slab()` method (mirrors the existing `_maybe_clamp_road_scales`). Gated behind a default-off config so it is byte-identical to the current baseline when disabled.

**Tech Stack:** PyTorch, OmegaConf/Hydra config, pytest (CPU-only tests).

## Global Constraints

- Default `strategy.bg_road_slab_exclude.enabled: false` → new code is a strict no-op (baseline byte-identical).
- Hard clamp only; NEVER a loss term (E3.2.6 soft penalty was negated).
- Device-agnostic helpers (work on CPU in tests, CUDA in training).
- `mode: clamp` is the only implemented mode in A1; `prune` raises NotImplementedError (reserved follow-up).
- `xy_dilate` config key reserved but a no-op in A1 (footprint = exact road BEV support).
- road layer is ~frozen → build the BEV grid ONCE and cache on the strategy.

---

### Task 1: Road BEV height grid + slab membership helpers

**Files:**
- Modify: `threedgrut/model/road_reg.py` (append helpers + `RoadBev` dataclass)
- Test: `threedgrut/tests/test_road_slab_bg_exclude.py` (create)

**Interfaces:**
- Produces:
  - `RoadBev` dataclass: fields `origin: Tensor[2]`, `cell: float`, `height: Tensor[H,W]`, `mask: Tensor[H,W] bool`.
  - `build_road_bev_height(road_xyz: Tensor[N,3], cell: float = 0.20) -> RoadBev`
  - `bg_in_road_slab_mask(bg_xyz: Tensor[M,3], bev: RoadBev, band_z: float = 0.15) -> Tensor[M] bool`

- [ ] **Step 1: Write the failing test**

```python
# threedgrut/tests/test_road_slab_bg_exclude.py
# SPDX-License-Identifier: Apache-2.0
"""A1 road-slab background exclusion: geometry + strategy clamp (CPU)."""
import torch


def test_road_slab_mask_geometry():
    from threedgrut.model.road_reg import build_road_bev_height, bg_in_road_slab_mask

    # Flat road patch at z=0 over an 11x11 grid in [0,1]^2.
    xs = torch.linspace(0.0, 1.0, 11)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    road = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(121)], dim=-1)

    bev = build_road_bev_height(road, cell=0.2)
    assert bool(bev.mask.any())

    bg = torch.tensor([
        [0.5, 0.5, 0.05],   # on the road, within +/-0.15 band -> True
        [0.5, 0.5, 0.50],   # car-height above road -> False
        [5.0, 5.0, 0.00],   # outside road footprint -> False
        [0.5, 0.5, -0.10],  # slightly below surface, within band -> True
    ])
    m = bg_in_road_slab_mask(bg, bev, band_z=0.15)
    assert m.tolist() == [True, False, False, True]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest threedgrut/tests/test_road_slab_bg_exclude.py::test_road_slab_mask_geometry -q`
Expected: FAIL with ImportError (`build_road_bev_height` not defined).

- [ ] **Step 3: Write minimal implementation**

Append to `threedgrut/model/road_reg.py`:

```python
from dataclasses import dataclass


@dataclass
class RoadBev:
    """Cached BEV road-surface height field built from the (frozen) road layer."""
    origin: "torch.Tensor"   # [2] (min_x, min_y)
    cell: float
    height: "torch.Tensor"   # [H, W] mean road z per cell (0 where unsupported)
    mask: "torch.Tensor"     # [H, W] bool, True where road support exists


def build_road_bev_height(road_xyz, cell: float = 0.20) -> RoadBev:
    """Build a BEV height grid from road-layer gaussian centers.

    Each occupied cell stores the mean road z of the centers that fall in it.
    The road layer is ~frozen, so the caller builds this once and caches it.
    """
    import torch

    xy = road_xyz[:, :2]
    z = road_xyz[:, 2]
    origin = xy.min(dim=0).values  # [2]
    ij = torch.floor((xy - origin) / cell).long()  # [N, 2]
    H = int(ij[:, 0].max().item()) + 1
    W = int(ij[:, 1].max().item()) + 1
    flat = ij[:, 0] * W + ij[:, 1]
    n_cells = H * W
    sum_z = torch.zeros(n_cells, device=z.device, dtype=z.dtype).scatter_add_(0, flat, z)
    cnt = torch.zeros(n_cells, device=z.device, dtype=z.dtype).scatter_add_(0, flat, torch.ones_like(z))
    height = torch.where(cnt > 0, sum_z / cnt.clamp(min=1.0), torch.zeros_like(sum_z)).reshape(H, W)
    mask = (cnt > 0).reshape(H, W)
    return RoadBev(origin=origin, cell=float(cell), height=height, mask=mask)


def bg_in_road_slab_mask(bg_xyz, bev: RoadBev, band_z: float = 0.15):
    """Bool[M]: background centers inside the road footprint AND within +/-band_z
    of that cell's road height."""
    import torch

    H, W = bev.height.shape
    xy = bg_xyz[:, :2]
    z = bg_xyz[:, 2]
    ij = torch.floor((xy - bev.origin) / bev.cell).long()
    ix, iy = ij[:, 0], ij[:, 1]
    in_bounds = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    ixc = ix.clamp(0, H - 1)
    iyc = iy.clamp(0, W - 1)
    cell_supported = bev.mask[ixc, iyc] & in_bounds
    cell_z = bev.height[ixc, iyc]
    return cell_supported & (torch.abs(z - cell_z) < band_z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest threedgrut/tests/test_road_slab_bg_exclude.py::test_road_slab_mask_geometry -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add threedgrut/model/road_reg.py threedgrut/tests/test_road_slab_bg_exclude.py
git commit -m "feat(road_reg): BEV road-height grid + bg slab membership helpers (A1)"
```

---

### Task 2: Wire exclusion into LayeredMCMCStrategy + config

**Files:**
- Modify: `threedgrut/strategy/layered_mcmc.py` (add `_maybe_exclude_bg_from_road_slab`, call it in `_post_optimizer_step`, init `_road_bev` cache)
- Modify: `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` (add `strategy.bg_road_slab_exclude` block, default off)
- Test: `threedgrut/tests/test_road_slab_bg_exclude.py` (append strategy test)

**Interfaces:**
- Consumes: `build_road_bev_height`, `bg_in_road_slab_mask`, `RoadBev` from Task 1.
- Produces: `LayeredMCMCStrategy._maybe_exclude_bg_from_road_slab(self) -> None`; reads `self.conf.strategy.bg_road_slab_exclude.{enabled,band_z,cell,mode,clamp_value}`, `self.model.layers["background"|"road"]`, caches `self._road_bev`.

- [ ] **Step 1: Write the failing test**

Append to `threedgrut/tests/test_road_slab_bg_exclude.py`:

```python
def _layer(pos, dens):
    import torch
    L = type("L", (), {})()
    L.positions = torch.nn.Parameter(pos)
    L.density = torch.nn.Parameter(dens)
    return L


def test_bg_road_slab_exclude_clamps_in_slab_only():
    import torch
    from omegaconf import OmegaConf
    from threedgrut.strategy.layered_mcmc import LayeredMCMCStrategy

    xs = torch.linspace(0.0, 1.0, 6)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    road = _layer(torch.stack([gx.flatten(), gy.flatten(), torch.zeros(36)], -1),
                  torch.zeros(36, 1))
    bg_pos = torch.tensor([[0.5, 0.5, 0.05], [0.5, 0.5, 0.60], [9.0, 9.0, 0.0]])
    bg = _layer(bg_pos.clone(), torch.zeros(3, 1))
    model = type("M", (), {"layers": {"background": bg, "road": road}})()

    strat = LayeredMCMCStrategy.__new__(LayeredMCMCStrategy)
    strat.model = model
    strat._road_bev = None
    strat.conf = OmegaConf.create({"strategy": {"bg_road_slab_exclude": {
        "enabled": True, "band_z": 0.15, "cell": 0.2,
        "xy_dilate": 0.0, "mode": "clamp", "clamp_value": -50.0}}})

    strat._maybe_exclude_bg_from_road_slab()
    assert bg.density[0].item() == -50.0   # in slab -> clamped
    assert bg.density[1].item() == 0.0     # above road -> untouched
    assert bg.density[2].item() == 0.0     # off footprint -> untouched

    # disabled -> strict no-op
    bg2 = _layer(bg_pos.clone(), torch.zeros(3, 1))
    model2 = type("M", (), {"layers": {"background": bg2, "road": road}})()
    strat2 = LayeredMCMCStrategy.__new__(LayeredMCMCStrategy)
    strat2.model = model2
    strat2._road_bev = None
    strat2.conf = OmegaConf.create({"strategy": {"bg_road_slab_exclude": {"enabled": False}}})
    strat2._maybe_exclude_bg_from_road_slab()
    assert bool(torch.all(bg2.density == 0.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest threedgrut/tests/test_road_slab_bg_exclude.py::test_bg_road_slab_exclude_clamps_in_slab_only -q`
Expected: FAIL with AttributeError (`_maybe_exclude_bg_from_road_slab` not defined).

- [ ] **Step 3: Write minimal implementation**

In `threedgrut/strategy/layered_mcmc.py`, add the import near the existing `from threedgrut.model.road_reg import clamp_layer_scales`:

```python
from threedgrut.model.road_reg import (
    clamp_layer_scales,
    build_road_bev_height,
    bg_in_road_slab_mask,
)
```

Add the method (place next to `_maybe_clamp_road_scales`):

```python
    def _maybe_exclude_bg_from_road_slab(self) -> None:
        """A1: hard-clamp opacity of background gaussians inside the road slab
        to ~0 so the frozen road layer owns the road surface. Gradient-free,
        runs every optimizer step. No-op unless
        strategy.bg_road_slab_exclude.enabled is true.
        """
        import torch

        strat = getattr(self.conf, "strategy", None)
        cfg = getattr(strat, "bg_road_slab_exclude", None) if strat is not None else None
        if cfg is None or not getattr(cfg, "enabled", False):
            return
        layers = getattr(self.model, "layers", None)
        if not layers or "background" not in layers or "road" not in layers:
            return
        bg = layers["background"]
        road = layers["road"]
        if road.positions.shape[0] == 0 or bg.positions.shape[0] == 0:
            return

        mode = str(getattr(cfg, "mode", "clamp"))
        if mode != "clamp":
            raise NotImplementedError(
                f"bg_road_slab_exclude.mode='{mode}' not implemented in A1 (use 'clamp')"
            )

        if getattr(self, "_road_bev", None) is None:
            self._road_bev = build_road_bev_height(
                road.positions.detach(), cell=float(getattr(cfg, "cell", 0.20))
            )
        mask = bg_in_road_slab_mask(
            bg.positions.detach(), self._road_bev, band_z=float(getattr(cfg, "band_z", 0.15))
        )
        if not bool(mask.any()):
            return
        with torch.no_grad():
            bg.density[mask] = float(getattr(cfg, "clamp_value", -50.0))
```

In `_post_optimizer_step`, add the call right after `self._maybe_clamp_road_scales()`:

```python
        self._maybe_clamp_road_scales()
        self._maybe_exclude_bg_from_road_slab()
```

In `LayeredMCMCStrategy.__init__`, initialize the cache (near the other instance attrs):

```python
        self._road_bev = None
```

- [ ] **Step 4: Add the config block**

In `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`, under the existing `strategy:` mapping, add:

```yaml
  bg_road_slab_exclude:
    enabled: false        # A1 A/B switch (off = current depth-off baseline)
    band_z: 0.15          # m, half-thickness of the exclusion slab
    cell: 0.20            # m, BEV grid cell for the road height field
    xy_dilate: 0.0        # reserved (no-op in A1)
    mode: clamp           # clamp (prune reserved)
    clamp_value: -50.0    # pre-sigmoid density sentinel
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest threedgrut/tests/test_road_slab_bg_exclude.py -q`
Expected: 2 passed.

Then config-loads cleanly (Hydra compose smoke):
Run: `python3 -c "from hydra import compose, initialize_config_dir; import os; d=os.path.abspath('configs'); initialize_config_dir(config_dir=d, version_base=None).__enter__(); c=compose(config_name='apps/ncore_3dgut_mcmc_multilayer'); print('enabled=', c.strategy.bg_road_slab_exclude.enabled)"`
Expected: prints `enabled= False`.

- [ ] **Step 6: Run the broader strategy test suite (no regressions)**

Run: `python3 -m pytest threedgrut/tests/test_layered_mcmc.py threedgrut/tests/test_road_scale_clamp.py -q`
Expected: all pass (baseline path untouched when disabled).

- [ ] **Step 7: Commit**

```bash
git add threedgrut/strategy/layered_mcmc.py configs/apps/ncore_3dgut_mcmc_multilayer.yaml threedgrut/tests/test_road_slab_bg_exclude.py
git commit -m "feat(layered_mcmc): A1 road-slab background opacity exclusion (default off)"
```

---

## Self-Review

- **Spec coverage:** mechanism (Task 2 method) ✓; road slab def via BEV height + band_z (Task 1) ✓; per-step hook in `_post_optimizer_step` (Task 2) ✓; config single-toggle default-off (Task 2 Step 4) ✓; clamp mode + clamp_value (Task 2) ✓; prune reserved (NotImplementedError) ✓; CPU regression test (Tasks 1+2) ✓; A/B + off-track eval = runtime/experiment step, out of code scope ✓.
- **Placeholder scan:** no TBD/TODO; all code shown.
- **Type consistency:** `RoadBev`, `build_road_bev_height`, `bg_in_road_slab_mask` names identical across Task 1 and Task 2; `_road_bev` cache name consistent.

## Out of scope (later)

- GPU A/B run on 9ae (depth-off) + `scripts/eval_road_offtrack.py` evaluation — runtime, after code lands and a GPU frees.
- Step B (un-freeze road appearance + road-only sparse depth) and A+B — separate specs.
- A2 render-time per-pixel mask gating — escalation only if A1 center-test leak is visually bad.
