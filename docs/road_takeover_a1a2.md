# Road→Background Takeover — A1 + A2 fix (image-space road-mask gating)

Date: 2026-06-27 · Clip: 9ae151dc (PAI/FTheta) · inceptio RTX 4090, depth-off, nw=10
Commits: `83aefb3` (A-bugfix) · `8b0fa92` (A1) · `d269b3b` (A2)

## Problem (= the P3.4 "air-zone floating bg" diagnosis, 2026-06-11)

In the multilayer model the renderer fuses ALL layers into ONE flat Gaussian
cloud with a single global alpha-blend — there is **no per-pixel layer
assignment** (`LayerSpec.mask_field` is declared but never consumed in
rendering). So gradient descent, minimizing training-view L1, lets the
high-capacity **background** layer own the road surface (incl. lane/zebra
stripes) instead of the frozen **road** layer. Background has no geometric
anchor (depth-off), so the bg gaussians that paint the road sit at the **wrong
depth** — fine at training poses, but under novel-view camera motion they shift
by parallax → the road **warps** → novel-view road consistency collapses.

This is the "road/bg 空间耦合 + 空气区悬浮 bg 零约束 = novel 鬼影" problem
recorded under P3.3/P3.4 (v3_plan_revised § 6, 2026-06-11). 30k does NOT fix it
(architecture/competition, not convergence; eff_rank even degraded 3k→30k).
Prior attempts negated: E3.2.5 freeze (created the ownership gap), E3.2.6 soft
opacity-loss penalty (optimizer trades it against cc; road can't densify to
fill in), E3.3 BEV texture (all metrics down). **Common failure: they assume
the frozen road can adapt to reclaim the pixels — it cannot.**

## A1 — 3D road-slab background clamp (cheap first try; insufficient)

Every MCMC step, hard-clamp (gradient-free, NOT a loss) the opacity of bg
gaussians whose **center** lies in a thin road slab (`(x,y)∈road BEV footprint`
AND `|z − road_surface_z| < band_z=0.15m`) to ~0, so the frozen road owns the
surface. Reuses the `_post_optimizer_step` clamp hook.

**Result (3k A/B, off-track eval):** grad_corr +0.08~0.11 (structure up), but
**band_psnr −1.5** (photometry DOWN), and **viser: stripes still in background.**

**Why insufficient (diagnostic on the A1 ckpt):** the stripe-painting bg
**float above** the thin slab — ~199k bg are 0.15–2.0m above the road footprint
(~74k alive). A1's center-in-slab test misses them. Widening the slab is wrong:
the 0.15–2m band also holds legitimate above-road objects (cars, signs) that the
3D test cannot distinguish from road-projecting floaters.

## A2 — image-space road-mask projection clamp (the fix)

Catch the floaters by **where they project, not their 3D height**: every MCMC
step, project bg centers into the current training camera
(`FthetaForwardProjector` / `PinholeForwardProjector`, identity world→cam flip
for NCore OpenCV poses), sample the **road sseg-mask** (`image_infos["road_mask"]`,
already per-pixel aligned + already passed to the strategy via `batch=`) at the
projected pixel, and hard-clamp density of bg landing on a road pixel.

- Image-space → floating bg over the road is caught; the road sseg-mask is
  road-CLASS only, so bg projecting onto cars/objects on the road is left alone.
- Gradient-free (NOT a loss → not subject to the cc-vs-penalty tradeoff that
  negated E3.2.6). Reuses the per-step clamp hook + the in-tree
  `_project_cuboids_to_dyn_mask` projection precedent.
- **No CUDA / tracer change** — the tracer is frozen (3dgrt OptiX-compiled,
  3dgut a precompiled .so; no per-pixel mask field, no per-gaussian layer-id).
  A true per-pixel tracer gate is impossible; projection-clamp is the realizable
  image-space equivalent. (This is the HARD realization of the planned P3.4
  "air-zone penalty".)

## Results — 3k single-variable A/B (off-track eval, 6-mode mean)

| metric | baseline | A1-only | **A1+A2** |
|---|---|---|---|
| lane_grad_corr (structure↑) | ~0.26 | ~0.36 (+0.10) | **~0.38 (+0.13)** |
| lane_band_psnr (photometry↑) | ~14.0 | ~12.4 (−1.5) | **~17.6 (+3.6)** |
| cc_psnr_masked (on-track guard) | 11.49 | 11.36 (−0.13) | **11.58 (+0.09)** |
| road_crop_psnr | 17.97 | 17.74 | **20.0 (+2.0)** |
| lane_band_lpips (↓ better) | 0.041 | 0.0395 | **0.0372** |

**A1+A2 beats baseline AND A1-only on every metric.** The band_psnr surprise
(+3.6, vs the feasibility prediction of "grey blank road") is because the gain
is mostly **road geometry correctness** — removing the wrong-depth floaters puts
the road band in the right place under novel views (asphalt no longer warps),
which dominates PSNR; stripe SHARPNESS is a small-area secondary factor.

**Float diagnostic (A1-only → A1+A2), bg over road footprint, alive (sigmoid>0.05):**
- in-slab |dz|<0.15m: 943 → **17** (road surface bg cleared)
- above 0.15–2.0m (floaters): 74,147 → **60,906** (~13k road-projecting floaters removed; the rest project onto non-road/object pixels, correctly kept)
- above >2.0m: 153k → **170k** (evicted bg relocated up — fine, renders non-road)

**viser (大g, 2026-06-27):** (1) stripes now in the **road** gaussian layer
(takeover fixed); (2) off-route consistency improved; (3) stripe sharpness "not
yet enough but acceptable, training-length related".

## Config (default off → byte-identical baseline)

```yaml
strategy:
  bg_road_slab_exclude:
    enabled: false              # A1 (3D slab)
    band_z: 0.15
    cell: 0.20
    mode: clamp
    clamp_value: -50.0
    projection_enabled: false   # A2 (image-space projection clamp)
```

Enable: `++strategy.bg_road_slab_exclude.enabled=true ++strategy.bg_road_slab_exclude.projection_enabled=true`.

## Open items

- **A1+A2 30k** (running): confirm gains hold 3k→30k (eff_rank degraded — verify
  A1+A2 does not overfit) + likely sharper stripes (大g's hypothesis).
- **B (deferred, 大g: acceptable)**: un-freeze road appearance / capacity so the
  road layer draws crisp stripes (not just correct-geometry grey). Revisit if 30k
  stripes still insufficient.
- **A2 perf**: numpy CPU projection of ~1M bg centers/step (~25% slowdown). Port
  `project_points` to torch/GPU if it becomes a bottleneck.
- **multi-cam**: a float is only tested when its source camera is the current
  batch cam (B=1/step) — rare-view floats get fewer clamp opportunities.
