# A1 — Road-slab background exclusion (hard, gradient-free)

Date: 2026-06-26
Status: design (awaiting review)
Author: Claude (with 大g)

## Context & goal

Real goal: **novel-view synthesis** where the **road surface stays spatially
consistent** under camera rotation/translation. The multilayer model has a
`road` particle layer with special handling (zero-thickness disk + normal lock
+ freeze, E3.2.5) that gives good road *geometry* — but in the full model the
**background layer "takes over" the road region**: background Gaussians, not the
road layer, end up rendering the road appearance. Background has no geometric
anchor (inceptio runs are depth-off), so under novel-view motion its road fit
breaks and road consistency collapses.

This is **step A** of an agreed three-step plan: **A (solo) → B → A+B**.
- **A** = stop background from painting the road (this spec).
- **B** = re-arm the road layer to render road appearance itself (later spec).
- **A+B** = combine.

## Why this, not 30k or the prior failed fixes

- **30k will NOT fix takeover** (architecture/competition problem, not
  convergence). The renderer fuses all layers into one flat Gaussian cloud with
  a single global alpha-blend — there is **no per-pixel layer assignment**
  (`LayerSpec.mask_field` is declared but never consumed in rendering). More
  iterations only let high-capacity background fit training views better,
  deepening takeover (eff_rank report: 3k→30k dropped off-track band_psnr
  −1.7~−3.3 dB).
- **Do NOT repeat E3.2.6** (negated): a *soft* opacity-LOSS penalty on
  background. It failed because the optimizer trades it against cc, and it
  assumed the frozen road could MCMC-relocate to fill the gap (it can't).
- **A1 is a HARD, gradient-free, per-step clamp** (like the existing road
  `clamp_layer_scales`), NOT a loss term — so it is not subject to the
  cc-vs-penalty tradeoff that killed E3.2.6.

## Mechanism

Every MCMC `_post_optimizer_step` (the same hook that already runs
`_maybe_clamp_road_scales`), find **background-layer** Gaussians whose center
lies inside a thin **road slab** and clamp their opacity to ~0 so they cannot
contribute to the fused render. The frozen road discs then become the only
thing rendering the road surface.

**Road slab** = `(x,y) ∈ road BEV footprint` AND `|z − road_surface_z(x,y)| < band_z`.
- `band_z` tight (default **0.15 m**): excludes background only from the thin
  ground slab, NOT the airspace above (cars / signs / vegetation above the road
  legitimately belong to background and must be left alone).
- Road footprint + height field come from the **road layer's own positions**
  (`model.layers["road"].positions`), which are ~frozen (positions_lr=1e-6,
  in `exclude_layer_ids`). So the surface is effectively static → build a **BEV
  height grid once and cache it**; O(1) per-step lookup.

**Action** = `clamp` (default): set the in-slab background `density`
(pre-sigmoid opacity) to a large negative sentinel (e.g. −50) every step, so
`sigmoid(density) ≈ 0`. Non-destructive: a background particle that later drifts
out of the slab recovers, and MCMC is free to relocate suppressed particles
elsewhere (good — frees capacity for non-road background). `prune` (delete) is a
more aggressive mode kept behind the same flag for a later follow-up.

## Components / interfaces

- **New method** `LayeredMCMCStrategy._maybe_exclude_bg_from_road_slab(self)`
  in `threedgrut/strategy/layered_mcmc.py`, mirroring `_maybe_clamp_road_scales`
  (~L187-212). Called from `_post_optimizer_step` right after
  `self._maybe_clamp_road_scales()` (L134).
- **BEV road height grid** helper in `threedgrut/model/road_reg.py` (sibling of
  `clamp_layer_scales`): `build_road_bev_height(road_positions, cell, dilate)
  → (origin, cell, height_grid[H,W], mask_grid[H,W])`. Built lazily on first
  call, cached on the strategy (road is frozen, so static).
- **Slab membership + clamp** helper in `road_reg.py`:
  `bg_in_road_slab_mask(bg_xyz, bev) → Bool[N_bg]` (XY in footprint AND
  |z − cell_z| < band_z), then the strategy clamps
  `model.layers["background"].density[mask] = clamp_value`.

## Config (single toggle, single-variable A/B)

Add under `strategy` (sibling of `exclude_layer_ids`):

```yaml
strategy:
  bg_road_slab_exclude:
    enabled: false        # ★ A/B switch (off = current depth-off baseline)
    band_z: 0.15          # m, half-thickness of the exclusion slab
    cell: 0.20            # m, BEV grid cell for the road height field
    xy_dilate: 0.0        # m, optionally grow road footprint at edges (curbs)
    mode: clamp           # clamp | prune
    clamp_value: -50.0    # pre-sigmoid density sentinel when mode=clamp
```

Default `enabled: false` → byte-identical to the current baseline (the new code
is a no-op). Turn on via `++strategy.bg_road_slab_exclude.enabled=true`.

## Edge cases / risks

1. **3D approximation leak**: A1 tests the background *center*; a background
   Gaussian centered just above the slab with a large covariance can still paint
   road pixels. A1 accepts this (cheap first step); the per-pixel-precise variant
   (render-time mask gating, "A2") is the escalation if leak is visually bad.
2. **Road must actually cover the road** when background is excluded. De-risked
   by 大g's observation that the road-only view is geometrically good (road has
   non-trivial opacity on the surface) — so A1 should not punch holes, only lose
   stripe detail.
3. **Detail loss is EXPECTED**: the frozen road is DC-color + coarse grid, so
   it cannot render sharp white lane stripes. A1's road will be geometrically
   stable but **less detailed** (stripes blur to averaged gray). This is the
   known cost and the reason step **B** follows. A1's success criterion is
   *novel-view consistency up*, not detail.
4. **band_z too large** would wrongly suppress low background objects (curbs,
   barriers, low vegetation) → keep tight; expose as config for a small sweep.
5. **Footprint from a sloped/banked road** is handled per-cell by the BEV
   height grid (each cell has its own z), not a single global plane.

## Verification

- **Regression test (CPU, no GPU)**: unit test for `bg_in_road_slab_mask` +
  the clamp — given a synthetic road height grid and background positions
  (some in-slab, some above, some off-footprint), assert exactly the in-slab
  ones get the sentinel density and others are untouched. Pins the geometry.
- **A/B experiment** on **9ae PAI** (depth-off), single variable
  `bg_road_slab_exclude.enabled` on vs off, 5k smoke first then 30k if
  promising. 9ae chosen because the off-track eval + depth-off baselines are
  ready there (and it avoids the OpenCVPinhole multi-cam issues).
- **Metric** = `scripts/eval_road_offtrack.py`: lane_grad_corr (lat 3m/6m, yaw),
  band_psnr/lpips, with `cc_psnr_masked` as the on-track guard.
- **Expected**: off-track lane_grad_corr / band_psnr **up**; on-track cc maybe
  slightly down (acceptable); white-stripe detail down (expected — motivates B).
  If off-track does NOT improve, A is not the lever and we rethink before B.

## Out of scope (later steps)

- **B**: decouple geometric freeze from appearance freeze — keep Z/normal lock
  but take road out of `exclude_layer_ids`, allow in-plane (XY) densify + free
  color/opacity, anchored by a road-only sparse LiDAR depth term.
- **A+B**: combine.
- **A2** (render-time per-pixel mask gating) — escalation only if A1's
  center-test leak is visually unacceptable.
