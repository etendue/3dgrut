# Protected Warm-Start Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect asset-harvester warm-start (diffusion-completed, unobserved-face) Gaussians from MCMC relocate/perturb and the opacity L1 decay, so the injected 360° geometry survives training while observed faces still refine via Adam gradients.

**Architecture:** Tag warm-started tracks with a small persistent `_warmstart_protected_track_ids` buffer on the `dynamic_rigids` layer MoG (written at injection, round-trips through ckpt). The `dynamic_rigids` MCMC sub-strategy then (a) drops protected-track particles out of `relocate_gaussians`' dead set and (b) zeroes their `perturb_gaussians` noise. The opacity L1 decay already supports per-layer exemption (`exempt_layers_opacity_reg`), so we add `dynamic_rigids` to it. All changes are **no-ops when no protected set exists** (every non-warm layer, every baseline run stays byte-identical).

**Tech Stack:** Python 3.14 / PyTorch (CPU on Mac for unit tests, CUDA on A800/inceptio for the A/B), Hydra configs, pytest 9.0.3. Repo: `3dgrut2`, branch `claude/interesting-mccarthy-03c6be` (worktree `.claude/worktrees/interesting-mccarthy-03c6be`, **not yet merged to main**).

---

## Source spec & traceability

- Design spec: [`docs/superpowers/specs/2026-06-05-protected-warmstart-design.md`](../specs/2026-06-05-protected-warmstart-design.md) (C1–C5 配方 + §4 C2 方案 + §5 验证 + §6 文件清单 + §7 风险).
- Depends on P1.4 warm-start injection engine (commit `1237111`, same branch).
- Maps spec changes → tasks: **C1** → Task 6, **C2** → Tasks 1–5 (helpers→relocate→perturb→buffer→source), **C3** → Task 7, **C4** (keep Adam refine) → no code (already default; verified in Task 8), **C5** (iters) → Task 8.

## Key code anchors (already verified against the worktree)

| What | Location | Note |
|---|---|---|
| MCMC `relocate_gaussians` (dead set) | [`threedgrut/strategy/mcmc.py:110-152`](../../../threedgrut/strategy/mcmc.py) | `self.model` = the per-layer MoG (`model.layers["dynamic_rigids"]`), which carries `track_ids`. CUDA (`_mcmc_plugin`) inside `sample_new_gaussians`. |
| MCMC `perturb_gaussians` | [`threedgrut/strategy/mcmc.py:207-231`](../../../threedgrut/strategy/mcmc.py) | **No CUDA plugin** — pure torch ops + model getters → CPU-testable. Axis mask at line 229, `positions.add_(noise)` at 231. |
| Per-layer sub-strategy | [`threedgrut/strategy/layered_mcmc.py:36-43`](../../../threedgrut/strategy/layered_mcmc.py) | `sub_strategies["dynamic_rigids"]` is a bare `MCMCStrategy` bound to the layer MoG → our mcmc.py edits apply per-layer automatically. |
| `init_layer_from_points` (buffer reg) | [`threedgrut/layers/layered_model.py:1260-1338`](../../../threedgrut/layers/layered_model.py) | `track_ids` registered persistent at line 1336-1337; signature kwargs at 1260-1272. |
| `get_density_excluding` (opacity-reg exempt) | [`threedgrut/layers/layered_model.py:796`](../../../threedgrut/layers/layered_model.py) | already wired; road uses it. |
| opacity-reg exempt read site | [`threedgrut/trainer.py:1196-1208`](../../../threedgrut/trainer.py) | `exempt = conf.loss.exempt_layers_opacity_reg`; uses `get_density_excluding`. |
| warm-start injection seam | [`threedgrut/trainer.py:486-533`](../../../threedgrut/trainer.py) | `_warm_merged = build_warmstart_layer_inputs(...)` → `model.init_layer_from_points("dynamic_rigids", ...)`. `warmstart_max_pts_per_track` fallback `5_000` at line 511-512. **LiDAR-only path's `5_000` at line 489 must NOT change.** |
| warm-start orchestrator | [`threedgrut/layers/warmstart_inject.py:41-98`](../../../threedgrut/layers/warmstart_inject.py) | `aligned_list` entries are `(name_to_id[track_key], aligned)`; returns merged dict (`_MERGE_KEYS`). |
| opacity-reg exempt config | [`configs/apps/ncore_3dgut_mcmc_multilayer.yaml:132`](../../../configs/apps/ncore_3dgut_mcmc_multilayer.yaml) | `exempt_layers_opacity_reg: [road]`. |
| CPU test pattern (no CUDA) | [`threedgrut/tests/test_learnable_pose_param.py:224-241`](../../../threedgrut/tests/test_learnable_pose_param.py), [`threedgrut/tests/conftest.py`](../../../threedgrut/tests/conftest.py) | `LayeredGaussians(conf, specs=specs, scene_extent=1.0)`; conftest installs the `MCMCStrategy.__init__` no-CUDA patch + sys.modules stubs. `MCMCStrategy.__new__(MCMCStrategy)` bypasses CUDA init. |

## Test environment (Mac, CPU)

All unit tests run on the **main repo venv** (worktree has none of its own):

```bash
source /Users/etendue/repo/3dgrut2/.venv/bin/activate
cd /Users/etendue/repo/3dgrut2/.claude/worktrees/interesting-mccarthy-03c6be
python -m pytest threedgrut/tests/test_protected_warmstart.py -v
```

Run that activation + `cd` once per shell; the per-task `Run:` lines assume it.

## File change map

| File | Change | Task |
|---|---|---|
| `threedgrut/strategy/mcmc.py` | add 2 pure module helpers; wire into `relocate_gaussians` + `perturb_gaussians` | 1, 2, 3 |
| `threedgrut/layers/layered_model.py` | `init_layer_from_points` gains `protected_track_ids` kwarg → persistent buffer | 4 |
| `threedgrut/layers/warmstart_inject.py` | `build_warmstart_layer_inputs` returns `warm_track_ids` | 5 |
| `threedgrut/trainer.py` | C1 default `5_000`→`50_000` (warm path only); pass `protected_track_ids` to injection | 6 |
| `configs/apps/ncore_3dgut_mcmc_multilayer.yaml` | `exempt_layers_opacity_reg: [road, dynamic_rigids]` | 7 |
| `threedgrut/tests/test_protected_warmstart.py` | **new** — all CPU unit tests for Tasks 1,3,4,5,7 | 1,3,4,5,7 |

## Baked-in decisions (resolved from the spec's open choices)

1. **C2 = per-track protection** (spec §4 recommended), implemented as a per-particle `track_id ∈ protected` filter inside the shared `MCMCStrategy` — automatically scoped to `dynamic_rigids` because that layer's sub-strategy is the only one whose MoG carries a protected buffer. The 65 LiDAR-only tracks in the same layer keep full MCMC. No whole-layer disable, so no per-track-eval fallback needed.
2. **`add_new_gaussians` left untouched.** With C1=50k×5≈250k warm + ~125k LiDAR ≈ 375k injected and `dynamic_rigids.max_n_particles=300k`, `_get_add_cap()` math makes `add` a no-op (`target=min(300k,1.05·375k)=300k < current` → 0 added). Keep `max_n_particles=300000` for the A/B so growth stays disabled with zero code. (Verified: `init_layer_from_points` does not clamp to the spec cap — it allocates Parameters of the injected length.)
3. **Protected set source of truth** = the mapped warm track integer ids (`name_to_id[track_key]` for each asset-mapped track) returned by `build_warmstart_layer_inputs` as `warm_track_ids`. (Authoritative; the viser `--replaced_track_ids 14,16,17,27,67` string in the spec is a viz display list and is NOT used here.)
4. **Validation scope** = code + Mac CPU unit tests (Tasks 1–7), then one A800 (preferred; inceptio fallback) 10k A/B (Task 8), then doc writeback. CLAUDE.md authorizes the GPU run without further confirmation.

---

### Task 1: Pure protected-index helpers in `mcmc.py`

**Files:**
- Modify: `threedgrut/strategy/mcmc.py` (add 2 module-level functions after the imports / before `load_mcmc_plugin`, around line 33)
- Test: `threedgrut/tests/test_protected_warmstart.py` (new)

- [ ] **Step 1: Write the failing test**

Create `threedgrut/tests/test_protected_warmstart.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Protected warm-start (C2): MCMC must not relocate/perturb asset-injected
(unobserved-face) particles. CPU-only unit tests — the dynamic relocate path
itself is CUDA (_mcmc_plugin) and is validated by the A800 A/B; here we pin the
pure index/mask logic + the buffer plumbing + the warm-track-id source.
"""
from __future__ import annotations

import torch

from threedgrut.strategy.mcmc import (
    _filter_protected_indices,
    _protected_particle_mask,
)


def test_filter_protected_indices_drops_matching():
    track_ids = torch.tensor([10, 20, 10, 30, 20])
    protected = torch.tensor([20])
    idxs = torch.tensor([0, 1, 2, 3, 4])
    out = _filter_protected_indices(idxs, track_ids, protected)
    assert out.tolist() == [0, 2, 3]  # particles 1 and 4 (track 20) removed


def test_filter_protected_indices_subset_input():
    track_ids = torch.tensor([10, 20, 10, 30, 20])
    protected = torch.tensor([20, 30])
    idxs = torch.tensor([1, 3, 4])  # all protected
    out = _filter_protected_indices(idxs, track_ids, protected)
    assert out.numel() == 0


def test_filter_protected_indices_noop_when_no_protected():
    track_ids = torch.tensor([10, 20, 10])
    idxs = torch.tensor([0, 1, 2])
    assert torch.equal(_filter_protected_indices(idxs, track_ids, None), idxs)
    empty = torch.tensor([], dtype=torch.long)
    assert torch.equal(_filter_protected_indices(idxs, track_ids, empty), idxs)


def test_filter_protected_indices_noop_when_no_track_ids():
    idxs = torch.tensor([0, 1, 2])
    out = _filter_protected_indices(idxs, None, torch.tensor([20]))
    assert torch.equal(out, idxs)


def test_protected_particle_mask_marks_protected():
    track_ids = torch.tensor([10, 20, 10, 30, 20])
    protected = torch.tensor([20])
    mask = _protected_particle_mask(track_ids, protected, n=5, device=torch.device("cpu"))
    assert mask is not None
    assert mask.tolist() == [False, True, False, False, True]


def test_protected_particle_mask_none_when_unprotected():
    track_ids = torch.tensor([10, 20, 10])
    assert _protected_particle_mask(track_ids, None, 3, torch.device("cpu")) is None
    empty = torch.tensor([], dtype=torch.long)
    assert _protected_particle_mask(track_ids, empty, 3, torch.device("cpu")) is None
    assert _protected_particle_mask(None, torch.tensor([20]), 3, torch.device("cpu")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -v`
Expected: FAIL at import — `ImportError: cannot import name '_filter_protected_indices' from 'threedgrut.strategy.mcmc'`.

- [ ] **Step 3: Write minimal implementation**

In `threedgrut/strategy/mcmc.py`, after the imports block (after line 32, before `_mcmc_plugin = None` at line 34), insert:

```python
def _filter_protected_indices(
    indices: torch.Tensor,
    track_ids: Optional[torch.Tensor],
    protected_track_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Drop entries of ``indices`` whose per-particle track id is protected.

    Protected = asset-harvester warm-start tracks whose unobserved-face geometry
    must not be relocated by MCMC. No-op (returns ``indices`` unchanged) when
    there is no protected set or no ``track_ids`` — keeps every non-warm layer
    and every baseline run byte-identical with v1.
    """
    if (
        protected_track_ids is None
        or protected_track_ids.numel() == 0
        or track_ids is None
        or indices.numel() == 0
    ):
        return indices
    prot = protected_track_ids.to(track_ids.device)
    keep = ~torch.isin(track_ids[indices], prot)
    return indices[keep]


def _protected_particle_mask(
    track_ids: Optional[torch.Tensor],
    protected_track_ids: Optional[torch.Tensor],
    n: int,
    device,
) -> Optional[torch.Tensor]:
    """Bool ``[n]`` mask, True where the particle's track id is protected.

    Returns ``None`` when nothing is protected so callers skip masking entirely
    (byte-identical default). ``n`` is the current particle count (asserted
    against ``track_ids`` length by the caller's own indexing).
    """
    if (
        protected_track_ids is None
        or protected_track_ids.numel() == 0
        or track_ids is None
    ):
        return None
    prot = protected_track_ids.to(track_ids.device)
    return torch.isin(track_ids, prot).to(device)
```

(`Optional` and `torch` are already imported at the top of `mcmc.py` — lines 25 & 27.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add threedgrut/strategy/mcmc.py threedgrut/tests/test_protected_warmstart.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C2 — pure track-id filter/mask helpers

_filter_protected_indices + _protected_particle_mask in mcmc.py: no-op when
no protected set (byte-identical for non-warm layers). CPU unit tests pin the
logic ahead of the relocate/perturb wiring.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire protected filter into `relocate_gaussians`

**Files:**
- Modify: `threedgrut/strategy/mcmc.py:114-116` (relocate dead set)

> The dynamic relocate path runs `_mcmc_plugin.compute_relocation_tensor` (CUDA),
> so this wiring is validated end-to-end by the A800 A/B (Task 8), not a Mac unit
> test. The *logic* it relies on is already pinned by Task 1's helper tests. This
> task is a 4-line edit + a static verification.

- [ ] **Step 1: Edit `relocate_gaussians`**

In `threedgrut/strategy/mcmc.py`, the current lines 113-116 are:

```python
        # Find the dead indices
        dead_idxs = torch.where(densities <= self.conf.strategy.opacity_threshold)[0]
        alive_idxs = torch.where(densities > self.conf.strategy.opacity_threshold)[0]
        n_dead_gaussians = len(dead_idxs)
```

Replace with:

```python
        # Find the dead indices
        dead_idxs = torch.where(densities <= self.conf.strategy.opacity_threshold)[0]
        alive_idxs = torch.where(densities > self.conf.strategy.opacity_threshold)[0]
        # Protected warm-start (C2): never relocate asset-injected particles —
        # their unobserved faces get no photometric gradient, so MCMC would
        # erode them onto high-error observed regions. No-op for layers with no
        # protected buffer (byte-identical v1).
        dead_idxs = _filter_protected_indices(
            dead_idxs,
            getattr(self.model, "track_ids", None),
            getattr(self.model, "_warmstart_protected_track_ids", None),
        )
        n_dead_gaussians = len(dead_idxs)
```

(The protected particles stay in `alive_idxs`, where they may serve as donors — harmless, since donor params are copied *into* dead spots, never moving the donor.)

- [ ] **Step 2: Static verification (no behavior change for baseline)**

Run: `python -m pytest threedgrut/tests/test_layered_mcmc.py -v`
Expected: PASS (all existing structural MCMC tests still green — the edit is a no-op when `_warmstart_protected_track_ids` is absent, which is the case for every test model).

- [ ] **Step 3: Confirm the edit landed**

Run: `grep -n "_filter_protected_indices" threedgrut/strategy/mcmc.py`
Expected: two lines — the definition (Task 1) and the new call inside `relocate_gaussians`.

- [ ] **Step 4: Commit**

```bash
git add threedgrut/strategy/mcmc.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C2 — exclude warm tracks from relocate dead set

relocate_gaussians drops protected-track particles out of dead_idxs so MCMC
never moves asset-injected unobserved-face geometry. No-op without a protected
buffer; existing test_layered_mcmc structural tests unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire protected mask into `perturb_gaussians` (+ CPU integration test)

**Files:**
- Modify: `threedgrut/strategy/mcmc.py:229-231` (perturb noise → positions)
- Test: `threedgrut/tests/test_protected_warmstart.py` (append)

> Unlike relocate, `perturb_gaussians` uses **no CUDA plugin**, so we can drive it
> end-to-end on CPU with a duck-typed stub model and assert protected positions
> are frozen.

- [ ] **Step 1: Write the failing test (append to `test_protected_warmstart.py`)**

```python
# --- Task 3: perturb_gaussians freezes protected particles (CPU integration) ---

from types import SimpleNamespace

import torch.nn as nn

from threedgrut.strategy.mcmc import MCMCStrategy


class _StubMoG:
    """Minimal duck-typed model exposing exactly what perturb_gaussians reads."""

    def __init__(self, positions, densities, track_ids, protected):
        self.positions = nn.Parameter(positions.clone())
        self._density = densities
        self.track_ids = track_ids
        self._warmstart_protected_track_ids = protected
        n = positions.shape[0]
        # identity covariance per particle → bmm leaves the noise vector intact
        self._cov = torch.eye(3).unsqueeze(0).expand(n, 3, 3).contiguous()
        self.optimizer = SimpleNamespace(
            param_groups=[{"name": "positions", "lr": 1.0}]
        )

    def get_covariance(self):
        return self._cov

    def get_positions(self):
        return self.positions.detach()

    def get_density(self):
        return self._density


def test_perturb_freezes_protected_particles():
    torch.manual_seed(0)
    n = 6
    positions = torch.zeros(n, 3)
    # low density everywhere → op_sigmoid(1-d) is non-trivial → real noise
    densities = torch.full((n, 1), 0.02)
    track_ids = torch.tensor([10, 20, 10, 30, 20, 10])
    protected = torch.tensor([20])  # particles 1 and 4 must not move
    model = _StubMoG(positions, densities, track_ids, protected)

    strat = MCMCStrategy.__new__(MCMCStrategy)  # bypass CUDA __init__
    strat.model = model
    strat.conf = SimpleNamespace(
        strategy=SimpleNamespace(perturb=SimpleNamespace(noise_lr=1.0))
    )

    before = model.positions.detach().clone()
    strat.perturb_gaussians()
    after = model.positions.detach()

    protected_rows = torch.tensor([1, 4])
    moved_rows = torch.tensor([0, 2, 3, 5])
    assert torch.equal(after[protected_rows], before[protected_rows])  # frozen
    assert not torch.allclose(after[moved_rows], before[moved_rows])   # perturbed


def test_perturb_byte_identical_without_protected():
    torch.manual_seed(0)
    n = 4
    model = _StubMoG(
        torch.zeros(n, 3), torch.full((n, 1), 0.02),
        torch.tensor([10, 20, 10, 30]), None,  # no protected buffer
    )
    strat = MCMCStrategy.__new__(MCMCStrategy)
    strat.model = model
    strat.conf = SimpleNamespace(
        strategy=SimpleNamespace(perturb=SimpleNamespace(noise_lr=1.0))
    )
    before = model.positions.detach().clone()
    strat.perturb_gaussians()
    # every particle moved (no freezing path taken)
    assert not torch.allclose(model.positions.detach(), before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py::test_perturb_freezes_protected_particles -v`
Expected: FAIL — without the wiring, protected rows DO move (assert `torch.equal(... frozen)` fails).

- [ ] **Step 3: Write minimal implementation**

In `threedgrut/strategy/mcmc.py`, current lines 227-231:

```python
        # T3.4 D1: per-axis mask on positional noise. Default is ones (v1
        # byte-identical); LayeredMCMC road sub overrides to (1, 1, 0).
        noise = noise * self._get_perturb_mask().to(noise.device).to(noise.dtype)

        self.model.positions.add_(noise)
```

Replace with:

```python
        # T3.4 D1: per-axis mask on positional noise. Default is ones (v1
        # byte-identical); LayeredMCMC road sub overrides to (1, 1, 0).
        noise = noise * self._get_perturb_mask().to(noise.device).to(noise.dtype)

        # Protected warm-start (C2): zero the perturb noise for asset-injected
        # particles so their diffusion-completed geometry doesn't drift. None
        # (no protected buffer) → byte-identical v1.
        _prot_mask = _protected_particle_mask(
            getattr(self.model, "track_ids", None),
            getattr(self.model, "_warmstart_protected_track_ids", None),
            noise.shape[0],
            noise.device,
        )
        if _prot_mask is not None:
            noise[_prot_mask] = 0.0

        self.model.positions.add_(noise)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -v`
Expected: PASS (all Task 1 + Task 3 tests, 10 total).

- [ ] **Step 5: Commit**

```bash
git add threedgrut/strategy/mcmc.py threedgrut/tests/test_protected_warmstart.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C2 — freeze warm tracks in perturb_gaussians

perturb_gaussians zeroes positional noise for protected-track particles. CPU
integration test (stub MoG) verifies protected rows stay put while others move,
and that absence of a protected buffer is byte-identical.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Persistent `_warmstart_protected_track_ids` buffer in `init_layer_from_points`

**Files:**
- Modify: `threedgrut/layers/layered_model.py:1260-1272` (signature) + `:1336-1337` (buffer reg)
- Test: `threedgrut/tests/test_protected_warmstart.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
# --- Task 4: init_layer_from_points writes a persistent protected buffer ---

import os

import pytest
from hydra import compose, initialize_config_dir

from threedgrut.layers.layer_spec import LayerSpec

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "configs")
)


@pytest.fixture(scope="module")
def dyn_conf():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        return compose(config_name="apps/ncore_3dgut_mcmc")


def _dyn_model(conf):
    from threedgrut.layers.layered_model import LayeredGaussians

    specs = [
        LayerSpec(name="background", layer_id=0, max_n_particles=100),
        LayerSpec(name="dynamic_rigids", layer_id=1, max_n_particles=100),
    ]
    return LayeredGaussians(conf, specs=specs, scene_extent=1.0)


def _inject_dyn(model, n_per_track):
    """Inject n_per_track particles for tracks 0,1,2 into dynamic_rigids."""
    positions, track_ids = [], []
    for tid in (0, 1, 2):
        positions.append(torch.zeros(n_per_track, 3))
        track_ids.append(torch.full((n_per_track,), tid, dtype=torch.long))
    positions = torch.cat(positions)
    track_ids = torch.cat(track_ids)
    model.init_layer_from_points(
        "dynamic_rigids",
        positions,
        scales=torch.full((positions.shape[0], 3), -3.0),
        densities=torch.zeros(positions.shape[0], 1),
        colors=torch.full((positions.shape[0], 3), 0.5),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(positions.shape[0], 4).clone(),
        track_ids=track_ids,
        protected_track_ids=torch.tensor([0, 2], dtype=torch.long),
        setup_optimizer=False,
    )
    return model


def test_init_layer_writes_protected_buffer(dyn_conf):
    model = _inject_dyn(_dyn_model(dyn_conf), n_per_track=4)
    layer = model.layers["dynamic_rigids"]
    assert hasattr(layer, "_warmstart_protected_track_ids")
    assert layer._warmstart_protected_track_ids.tolist() == [0, 2]
    assert layer._warmstart_protected_track_ids.dtype == torch.long


def test_protected_buffer_persists_in_state_dict(dyn_conf):
    model = _inject_dyn(_dyn_model(dyn_conf), n_per_track=4)
    sd = model.state_dict()
    key = "layers.dynamic_rigids._warmstart_protected_track_ids"
    assert key in sd, f"{key} missing from state_dict (not persistent?)"
    assert sd[key].tolist() == [0, 2]


def test_init_layer_no_protected_buffer_when_omitted(dyn_conf):
    """Backward-compat: omitting protected_track_ids registers no buffer."""
    model = _dyn_model(dyn_conf)
    model.init_layer_from_points(
        "dynamic_rigids",
        torch.zeros(6, 3),
        scales=torch.full((6, 3), -3.0),
        densities=torch.zeros(6, 1),
        colors=torch.full((6, 3), 0.5),
        rotations=torch.tensor([1.0, 0, 0, 0]).expand(6, 4).clone(),
        track_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        setup_optimizer=False,
    )
    layer = model.layers["dynamic_rigids"]
    assert not hasattr(layer, "_warmstart_protected_track_ids")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -k "protected_buffer or writes_protected or no_protected_buffer" -v`
Expected: FAIL — `TypeError: init_layer_from_points() got an unexpected keyword argument 'protected_track_ids'`.

- [ ] **Step 3: Write minimal implementation**

In `threedgrut/layers/layered_model.py`, add the kwarg to the signature. Current lines 1269-1272:

```python
        track_ids: Optional[torch.Tensor] = None,
        observer_pts: Optional[torch.Tensor] = None,
        setup_optimizer: bool = True,
    ) -> None:
```

Replace with:

```python
        track_ids: Optional[torch.Tensor] = None,
        protected_track_ids: Optional[torch.Tensor] = None,
        observer_pts: Optional[torch.Tensor] = None,
        setup_optimizer: bool = True,
    ) -> None:
```

Then register the buffer. Current lines 1336-1337:

```python
        if track_ids is not None:
            layer.register_buffer("track_ids", track_ids.long(), persistent=True)
```

Replace with:

```python
        if track_ids is not None:
            layer.register_buffer("track_ids", track_ids.long(), persistent=True)

        if protected_track_ids is not None:
            # Protected warm-start (C2): the set of asset-injected track ids
            # whose particles MCMC must not relocate/perturb. Persistent so it
            # round-trips through the training checkpoint.
            layer.register_buffer(
                "_warmstart_protected_track_ids",
                protected_track_ids.long(),
                persistent=True,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -v`
Expected: PASS (Task 1+3+4 tests, 13 total).

- [ ] **Step 5: Commit**

```bash
git add threedgrut/layers/layered_model.py threedgrut/tests/test_protected_warmstart.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C2 — persistent protected-track-id buffer

init_layer_from_points gains protected_track_ids kwarg → registers
_warmstart_protected_track_ids (persistent, ckpt round-trips). Omitting it
registers nothing (backward-compat). CPU tests cover write + state_dict persist.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `build_warmstart_layer_inputs` returns `warm_track_ids`

**Files:**
- Modify: `threedgrut/layers/warmstart_inject.py:91-98`
- Test: `threedgrut/tests/test_protected_warmstart.py` (append)

- [ ] **Step 1: Write the failing test (append)**

The orchestrator needs the real demo bundle; skip cleanly when absent (mirrors `_needs_bundle` in `test_warmstart_ply_engine.py`).

```python
# --- Task 5: build_warmstart_layer_inputs surfaces the warm (protected) ids ---

from pathlib import Path

from threedgrut.layers.warmstart_inject import build_warmstart_layer_inputs

_BUNDLE = Path(
    os.environ.get(
        "WARMSTART_BUNDLE",
        os.path.join(os.path.dirname(__file__), "..", "..",
                     "asset_harvester", "demo_bundle"),
    )
)
_needs_bundle = pytest.mark.skipif(
    not (_BUNDLE / "metadata.yaml").exists(),
    reason=f"warm-start demo bundle not found at {_BUNDLE}",
)


@_needs_bundle
def test_build_warmstart_returns_warm_track_ids():
    """warm_track_ids == the integer ids of the asset-mapped tracks, derived
    from track_names order (name_to_id)."""
    bundle = load_bundle_metadata(_BUNDLE / "metadata.yaml")
    # Map the first available asset to a track named so it lands in track_names.
    asset_hash = next(iter(bundle.keys()))
    track_key = "carA"
    track_names = ["bg_ignore", track_key]  # name_to_id[track_key] == 1
    tracks = {track_key: {"size": torch.tensor([4.0, 2.0, 1.6])}}
    mapping = {track_key: asset_hash}

    lidar_positions = torch.zeros(0, 3)
    lidar_track_ids = torch.zeros(0, dtype=torch.long)

    out = build_warmstart_layer_inputs(
        bundle_path=_BUNDLE,
        mapping=mapping,
        tracks=tracks,
        track_names=track_names,
        lidar_positions=lidar_positions,
        lidar_track_ids=lidar_track_ids,
        scale_prior=(0.1, 0.1, 0.1),
        density_init=0.1,
        mode="replace",
        max_pts_per_track=500,
        seed=0,
    )
    assert out is not None
    assert "warm_track_ids" in out
    assert out["warm_track_ids"].tolist() == [1]   # name_to_id["carA"]
    assert out["warm_track_ids"].dtype == torch.long
    # Every protected id must actually appear among the merged particles.
    merged_ids = set(out["track_ids"].unique().tolist())
    assert set(out["warm_track_ids"].tolist()) <= merged_ids
```

> Note: this test imports `load_bundle_metadata` — add it to the test file's
> imports (it is already used by `test_warmstart_ply_engine.py`):
> `from threedgrut.layers.warmstart_metadata import load_bundle_metadata`.
> The mapping value form (`{track_key: asset_hash}`) must match what
> `map_assets_to_tracks` expects; if the real bundle uses a different mapping
> schema, adjust `mapping`/`track_key` to one mapped track and assert its id.
> If the demo bundle is absent on this machine, the test skips — the assertion
> is still re-validated implicitly by the Task 8 A/B (warm tracks survive).

- [ ] **Step 2: Run test to verify it fails (or skips)**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py::test_build_warmstart_returns_warm_track_ids -v`
Expected: FAIL with `KeyError: 'warm_track_ids'` **if** the demo bundle exists; otherwise SKIP. If it skips, eyeball-verify Step 3 by reading the diff instead.

- [ ] **Step 3: Write minimal implementation**

In `threedgrut/layers/warmstart_inject.py`, current lines 91-98:

```python
        aligned_list.append((name_to_id[track_key], aligned))

    warm = assets_to_layer_inputs(aligned_list)
    return merge_warmstart_with_lidar(
        lidar_positions, lidar_track_ids, warm,
        max_pts_per_track=max_pts_per_track, scale_prior=scale_prior,
        density_init=density_init, mode=mode, generator=gen,
    )
```

Replace with:

```python
        aligned_list.append((name_to_id[track_key], aligned))

    warm = assets_to_layer_inputs(aligned_list)
    merged = merge_warmstart_with_lidar(
        lidar_positions, lidar_track_ids, warm,
        max_pts_per_track=max_pts_per_track, scale_prior=scale_prior,
        density_init=density_init, mode=mode, generator=gen,
    )
    if merged is not None:
        # Protected warm-start (C2): the integer ids of the asset-mapped tracks.
        # Consumed by the trainer → init_layer_from_points(protected_track_ids=)
        # so MCMC leaves these tracks' injected geometry alone.
        warm_ids = sorted({tid for tid, _ in aligned_list})
        merged["warm_track_ids"] = torch.tensor(warm_ids, dtype=torch.long)
    return merged
```

- [ ] **Step 4: Run tests to verify they pass (or skip)**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -v`
Expected: PASS or SKIP for the bundle test; all other tests PASS.

- [ ] **Step 5: Commit**

```bash
git add threedgrut/layers/warmstart_inject.py threedgrut/tests/test_protected_warmstart.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C2 — surface warm_track_ids from orchestrator

build_warmstart_layer_inputs returns the asset-mapped track ids as
warm_track_ids (the protected set source of truth). Bundle-gated CPU test.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Trainer wiring — C1 default 50k + pass `protected_track_ids`

**Files:**
- Modify: `threedgrut/trainer.py:511-512` (C1 default) + `:517-527` (pass protected ids)

> No new unit test: the trainer injection path needs the dataset + GPU and is
> exercised by the Task 8 A/B. This task is two precise edits + static checks.
> **Do NOT touch line 489** (`max_pts_per_track=5_000` for the LiDAR-only
> `init_dynamic_rigid_layer`) — that path must stay byte-identical.

- [ ] **Step 1: C1 — bump the warm-start fallback default**

In `threedgrut/trainer.py`, current lines 511-512:

```python
                                        max_pts_per_track=int(_extra.get(
                                            "warmstart_max_pts_per_track", 5_000)),
```

Replace with:

```python
                                        max_pts_per_track=int(_extra.get(
                                            "warmstart_max_pts_per_track", 50_000)),
```

- [ ] **Step 2: C2 — pass `protected_track_ids` into the injection**

In `threedgrut/trainer.py`, current lines 517-527 (the warm-merged branch):

```python
                                if _warm_merged is not None:
                                    model.init_layer_from_points(
                                        "dynamic_rigids",
                                        _warm_merged["positions"].to(device),
                                        colors=_warm_merged["colors"].to(device),
                                        rotations=_warm_merged["rotations"].to(device),
                                        scales=_warm_merged["scales"].to(device),
                                        densities=_warm_merged["densities"].to(device),
                                        track_ids=_warm_merged["track_ids"].to(device),
                                        setup_optimizer=True,
                                    )
```

Replace with:

```python
                                if _warm_merged is not None:
                                    _warm_ids = _warm_merged.get("warm_track_ids")
                                    model.init_layer_from_points(
                                        "dynamic_rigids",
                                        _warm_merged["positions"].to(device),
                                        colors=_warm_merged["colors"].to(device),
                                        rotations=_warm_merged["rotations"].to(device),
                                        scales=_warm_merged["scales"].to(device),
                                        densities=_warm_merged["densities"].to(device),
                                        track_ids=_warm_merged["track_ids"].to(device),
                                        protected_track_ids=(
                                            _warm_ids.to(device)
                                            if _warm_ids is not None else None
                                        ),
                                        setup_optimizer=True,
                                    )
```

- [ ] **Step 3: Static verification**

Run:
```bash
grep -n "warmstart_max_pts_per_track\", 50_000" threedgrut/trainer.py
grep -n "protected_track_ids=" threedgrut/trainer.py
grep -n "max_pts_per_track=5_000" threedgrut/trainer.py   # line 489 must STILL be 5_000
```
Expected: line 1 matches (50k default); line 2 matches (protected passed); line 3 still shows the LiDAR-only `5_000` at ~489 (untouched).

- [ ] **Step 4: Import smoke (no syntax break)**

Run: `python -c "import ast; ast.parse(open('threedgrut/trainer.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add threedgrut/trainer.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C1+C2 wiring — 50k default + protected ids

trainer warm-start seam: warmstart_max_pts_per_track default 5k→50k (warm path
only; LiDAR-only init untouched) and forwards warm_track_ids to
init_layer_from_points(protected_track_ids=).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: C3 — exempt `dynamic_rigids` from opacity L1 decay

**Files:**
- Modify: `configs/apps/ncore_3dgut_mcmc_multilayer.yaml:132`
- Test: `threedgrut/tests/test_protected_warmstart.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
# --- Task 7: opacity-reg exemption includes dynamic_rigids ---

def test_multilayer_exempts_dynamic_rigids_from_opacity_reg():
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        conf = compose(config_name="apps/ncore_3dgut_mcmc_multilayer")
    exempt = list(conf.loss.exempt_layers_opacity_reg)
    assert "road" in exempt
    assert "dynamic_rigids" in exempt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py::test_multilayer_exempts_dynamic_rigids_from_opacity_reg -v`
Expected: FAIL — `assert 'dynamic_rigids' in ['road']`.

- [ ] **Step 3: Edit the config**

In `configs/apps/ncore_3dgut_mcmc_multilayer.yaml`, current line 132:

```yaml
  exempt_layers_opacity_reg: [road]
```

Replace with:

```yaml
  exempt_layers_opacity_reg: [road, dynamic_rigids]   # P1.4 protected warm-start (C3): don't decay injected unobserved-face opacity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest threedgrut/tests/test_protected_warmstart.py -v`
Expected: PASS (full file green; bundle test may SKIP).

- [ ] **Step 5: Commit**

```bash
git add configs/apps/ncore_3dgut_mcmc_multilayer.yaml threedgrut/tests/test_protected_warmstart.py
git commit -m "$(cat <<'EOF'
feat(P1.4): protected warm-start C3 — exempt dynamic_rigids from opacity L1

Adds dynamic_rigids to exempt_layers_opacity_reg so the injected unobserved-face
opacity isn't monotonically decayed (only observed faces get gradient to hold it
up). Mechanism (get_density_excluding) already wired for road.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: A800 10k A/B validation + doc writeback

**Files:**
- Run (GPU): A800 `a800-x2` (preferred) or `inceptio` RTX 4090 (fallback)
- Modify: `v3_plan_revised.md` (§1 kanban, §1.2 P1.4 row, §6 Done Log), `v2_architecture.md` (§7 invariants) after results land.

> This is the only GPU-cost step. Per CLAUDE.md it runs without further user
> confirmation. Follow the A800 strict checklist (rsync → grep-verify code →
> train → cat metrics.json). Whole-file unit suite must be green FIRST.

- [ ] **Step 1: Full Mac unit suite green (gate before any GPU spend)**

Run:
```bash
python -m pytest threedgrut/tests/test_protected_warmstart.py \
                 threedgrut/tests/test_layered_mcmc.py \
                 threedgrut/tests/test_warmstart_ply_engine.py -v
```
Expected: all PASS (bundle-gated tests may SKIP on Mac). If anything fails, fix before proceeding — do NOT rsync a red tree to the GPU.

- [ ] **Step 2: Sync code to A800 and grep-verify the changes landed**

```bash
rsync -az --exclude='.claude/worktrees' --exclude='.venv' --exclude='__pycache__' \
  /Users/etendue/repo/3dgrut2/.claude/worktrees/interesting-mccarthy-03c6be/ \
  a800-x2:/root/work/yusun/repo/3dgrut/
ssh a800-x2 "grep -n '_warmstart_protected_track_ids' /root/work/yusun/repo/3dgrut/threedgrut/strategy/mcmc.py /root/work/yusun/repo/3dgrut/threedgrut/layers/layered_model.py && grep -n 'dynamic_rigids' /root/work/yusun/repo/3dgrut/configs/apps/ncore_3dgut_mcmc_multilayer.yaml | grep exempt"
```
Expected: the buffer name appears in both mcmc.py (relocate + perturb + helpers) and layered_model.py; the exempt line shows `[road, dynamic_rigids]`.

- [ ] **Step 3: Discover the P1.4 harvest bundle + mapping + clip on the GPU host**

```bash
ssh a800-x2 "find /root/work/yusun -name metadata.yaml -path '*warm*' -o -name 'metadata.yaml' -path '*bundle*' 2>/dev/null | head; ls /root/work/yusun/ncore-nurec/data/ncore/clips/ | grep 9ae151dc"
```
Record the resolved `<BUNDLE>` dir, the `<MAPPING>` (the asset→track map used in commit `1237111`'s AH-2 run — tracks 24/244/259/316/7 on clip `9ae151dc…`), and `<CLIP_JSON>` = `/root/work/yusun/ncore-nurec/data/ncore/clips/9ae151dc.../pai_9ae151dc....json`. If the bundle/mapping can't be found, STOP and ask — do not guess paths.

- [ ] **Step 4: Launch the dual-GPU 10k A/B (background)**

Run A = LiDAR-only baseline (no warm bundle); Run B = protected warm-start (C1–C3). Use `run_in_background=true`.

```bash
# RUN A — LiDAR-only control (GPU 0)
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=10000 \
    path=<CLIP_JSON> trainer.sky_backend=mlp \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=protected_A_lidar_10k 2>&1 | tee /tmp/protA.log'

# RUN B — protected warm-start (GPU 1); ++ for layers.overrides (multilayer has defaults)
ssh a800-x2 'export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH \
  && export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  && cd /root/work/yusun/repo/3dgrut \
  && python train.py --config-name apps/ncore_3dgut_mcmc_multilayer \
    n_iterations=10000 \
    path=<CLIP_JSON> trainer.sky_backend=mlp \
    ++layers.overrides.dynamic_rigids.warmstart_ply_bundle=<BUNDLE> \
    ++layers.overrides.dynamic_rigids.warmstart_ply_mapping=<MAPPING> \
    ++layers.overrides.dynamic_rigids.warmstart_mode=replace \
    ++layers.overrides.dynamic_rigids.warmstart_max_pts_per_track=50000 \
    out_dir=/root/work/yusun/ncore-nurec/output \
    experiment_name=protected_B_warm_10k 2>&1 | tee /tmp/protB.log'
```

Watch (Monitor grep, key nodes only): `🚗 dynamic_rigids WARM-START injected|🎊 Training Statistics|⭐ Test Metrics|Traceback|OOM|FAILED`.
- Confirm Run B logs `🚗 dynamic_rigids WARM-START injected: <N> particles` with `N ≈ 5×50k + LiDAR`.
- If OOM on A800 is unexpected at 50k, fall back to inceptio is NOT advised (24GB tighter); instead lower to `warmstart_max_pts_per_track=30000` and note the cap in the writeback.

- [ ] **Step 5: Read metrics + visual gate**

```bash
ssh a800-x2 "cat /root/work/yusun/ncore-nurec/output/protected_A_lidar_10k/*/metrics.json; echo '---'; cat /root/work/yusun/ncore-nurec/output/protected_B_warm_10k/*/metrics.json"
```
Required (spec §5 出口):
- `automobile` `class_psnr`(B) ≥ first-round B (22.06) **and** > A(10k).
- `cc_psnr_masked`(B) ≥ 24.7 guard (background not degraded).
- Visual (viser, original 3dgut + `--replaced_track_ids 24,244,259,316,7` + dynamic_replaced toggle): unobserved faces preserved (no holes/erosion), observed faces sharp (no spiky MCMC floaters). Capture before/after screenshots.

If metrics don't clear the gate, STOP and report (do not mark done). Likely follow-ups: PR2 floaters → add a dynamic opacity floor; PR5 misalign → L4 covariance align (out of this plan's scope).

- [ ] **Step 6: Doc writeback (task is NOT done until this lands)**

Update on the branch:
- `v3_plan_revised.md` §1.1 kanban: move the protected-warm-start card to Done ✅ (full-width parens 全角 if any).
- `v3_plan_revised.md` §1.2: P1.4 提质 row → ✅ + commit short hashes.
- `v3_plan_revised.md` §6 Done Log: new entry — date `2026-06-05`, commits, summary, **measured per-class numbers** (A vs B automobile class_psnr, cc_psnr_masked, it/s, iters).
- `v2_architecture.md` §7 关键不变量: add a row pinning "protected warm-start: relocate/perturb skip `_warmstart_protected_track_ids`; opacity-reg exempts dynamic_rigids".
- Run the mermaid `(` self-check on `v3_plan_revised.md` (must be zero output):
  ```bash
  awk '/```mermaid/{i=1;next} /```/&&i{i=0;next} i&&/\(/{print FILENAME":"NR": "$0}' v3_plan_revised.md
  ```

- [ ] **Step 7: Commit the writeback**

```bash
git add v3_plan_revised.md v2_architecture.md
git commit -m "$(cat <<'EOF'
docs(plan): protected warm-start done — A/B 10k measured per-class

docs(plan): mark P1.4 protected warm-start ✅ in v3_plan_revised.md kanban + Done Log
docs(arch): pin protected-warmstart invariants in v2_architecture.md §7

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review (run against the spec)

**1. Spec coverage:**
- C1 (5k→50k) → Task 6 Step 1. ✅
- C2 (per-track relocate+perturb protection, buffer, ckpt persist) → Tasks 1,2,3,4,5,6. ✅
- C3 (opacity-reg exempt dynamic_rigids) → Task 7. ✅
- C4 (keep Adam geom+appearance refine) → no code (already default); asserted via Run B converging in Task 8. ✅
- C5 (≤10k iters) → Task 8 Step 4 (`n_iterations=10000`). ✅
- §4 per-track vs whole-layer → per-track chosen (decision #1); per-track eval fallback not needed. ✅
- §5 validation (metrics + visual) → Task 8 Steps 5. ✅
- §6 file list → file change map covers mcmc.py, layered_model.py, warmstart_inject.py, trainer.py, config. `per_class_eval.py`/`render.py` listed in §6 only for the whole-layer path — not needed (decision #1). ✅
- §7 risks: PR1/PR5 noted in Task 8 Step 5 follow-ups; PR4 (>cap variable-length inject) covered by decision #2 + Task 4 persist tests. PR2 floaters noted as a follow-up. ✅

**2. Placeholder scan:** GPU paths `<BUNDLE>/<MAPPING>/<CLIP_JSON>` are resolved by an explicit discovery step (Task 8 Step 3), not left as silent TODOs. Task 5's mapping-schema caveat is flagged inline. No code step contains TBD/"handle edge cases". ✅

**3. Type consistency:** `_filter_protected_indices` / `_protected_particle_mask` signatures identical across definition (Task 1) and call sites (Tasks 2, 3). Buffer name `_warmstart_protected_track_ids` identical in layered_model.py (Task 4), mcmc.py reads (Tasks 2, 3), state_dict key (Task 4 test), and A800 grep (Task 8). Returned-dict key `warm_track_ids` identical in warmstart_inject.py (Task 5), trainer.py (Task 6), and tests. ✅
