# B2 Phase 0 Calibration Probe — Result Log

**Date:** 2026-05-24
**Probe script:** [scripts/probe_ftheta_overlay.py](../../scripts/probe_ftheta_overlay.py)
**Result JSON:** [B2_calibration_probe.json](B2_calibration_probe.json)
**Spec:** [docs/superpowers/specs/2026-05-24-b2-ftheta-cuboid-overlay-design.md](../superpowers/specs/2026-05-24-b2-ftheta-cuboid-overlay-design.md)

## Setup

- ckpt: `/home/yusun/work/ckpts/bug4_v2_full_30k/ckpt_with_ftheta_v2.pt`
- ThinkPad conda env: `3dgrut2`
- Probe cuboid: `tid=41 class=automobile`, world translation `(-72.33, +4.75, +1.55)`
- Probe ego pose: `ego_poses_c2w[0]` translation `(+2.17, +0.03, +1.44)`
- FTheta intrinsics: `resolution=(1920, 1080)`, `principal_point=(960.3, 545.4)`, `max_angle=1.221 rad = 69.9° half-FOV`
- `angle_to_pixeldist_poly`: `[0, 927.57, 5.75, -37.60, 24.83, -8.42]` (ascending order)
- `linear_cde`: `[1.0016, 0, 0]` — near-identity, skipped per ftheta_intrinsics.py:50-57

## Candidates Tested

| ID | c2w transform | poly order | n_visible/24 | Δv (bottom-top) | Verdict |
|---|---|---|---|---|---|
| A | `c2w @ diag([1,-1,-1,1])` (std GL→CV) | ascending | 24 | **-20.1** | ❌ bottom above top (violates OpenCV image +Y down) |
| B | identity (assume already OpenCV) | ascending | 0 | n/a | ❌ all behind camera (z=-76); ego pose is NOT raw OpenCV |
| C | `c2w @ diag([1,-1,-1,1])` | **descending** (np.polyval) | 24 | +2.4 | ❌ poly order wrong (all vertices collapse to ~principal_point) |
| **D** | `c2w @ diag([1, 1, -1, 1])` (Z-only flip) | **ascending** | **24** | **+20.1** | ✅ bottom below top, all visible, all in-FOV (2.6° - 4.4°) |

## Winning Combination (locked for B2 implementation)

```python
FLIP_C2W_TO_OPENCV = np.diag([1.0, 1.0, -1.0, 1.0])  # right-multiplied
c2w_opencv = c2w_viser @ FLIP_C2W_TO_OPENCV

# Polynomial (mirrors ftheta_intrinsics.py:69-70 ascending Horner)
def _horner_ascending(poly, x):
    out = 0.0 * x
    for k in range(len(poly) - 1, -1, -1):
        out = out * x + poly[k]
    return out
r_pix = _horner_ascending(ftheta["angle_to_pixeldist_poly"], angle)

# linear_cde: SKIP (≈ identity per ftheta_intrinsics.py:50-57 note)
```

## Interpretation

The viser/ckpt ego pose uses a **+Y down + Z backward** convention (Y already matches OpenCV image axis, only Z needs flipping). This is *not* standard OpenGL (+Y up + Z backward) — likely an artifact of NCore's native data convention being preserved through training and into the schema_v2 ckpt.

This matches `ftheta_intrinsics.py`'s inverse projection convention exactly (which uses `+V_pixel = +camera Y` directly without an extra Y flip).

## Sanity Numerics

Cuboid at world `(-72.3, +4.7, +1.5)`, ego at world `(+2.2, +0.03, +1.44)`:
- Relative: `(-74.5, +4.7, +0.06)` — ~74 m ahead-left, ~same height
- Cam-frame z under D: `+72.5 to +76.6 m` ✓ matches expected forward distance
- Cam-frame ray angle: `2.6° - 4.4°` ✓ matches `arctan2(sqrt(4.7² + 0.06²), 74.5) ≈ 3.6°`
- Cuboid v range: `549.7 (top) - 569.8 (bottom)` ✓ on-image (cy=545), spans ~20 px for a 1.6 m-tall car at 75 m
- Cuboid u range: `~892 - 919` ✓ left of cx=960 (cuboid is left of ego optical axis)

All numbers check out. **Ready to implement `FthetaForwardProjector` with winning combo D hardcoded.**
