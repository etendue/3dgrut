# PIN-MASK-1 Forward-Valid Supervision Mask Validation

Date: 2026-07-15
Branch: `fix/pinhole-forward-valid-mask`
Reviewed code commit after rebase: `ac3e027`
Probe script commit after rebase: `7b71987`
Host: `inceptio` / NCore SDK v4

## Scope

This task adds an opt-in dataset flag:

```yaml
dataset:
  mask_forward_invalid_pixels: false
```

When enabled, only `OpenCVPinholeCameraModel` cameras AND the forward projection `valid_flag` into the existing static ego/valid mask. The default is false. FTheta and OpenCVFisheye paths are strict no-ops.

This task does not claim a visual-quality improvement. The 5-second training comparison is tracked separately as PIN-AB-1.

## Production Path

1. Generate full-resolution inverse rays with `pixels_to_camera_rays()`.
2. Repair non-finite rational-pole rays with `repair_nonfinite_rays()` and mark them invalid.
3. For OpenCVPinhole only and when the flag is enabled, compute `camera_rays_to_pixels(rays).valid_flag` from the repaired cached rays.
4. AND the result into the existing static ego mask.
5. Copy the combined mask to per-frame masks.
6. Train, val, and `make_test()` consume the same mask. Validation indexes the full mask by `camera_pixels_subsampled`, so no second subsampled mask is required.

## b6a9 Standard 9-Camera Probe

Manifest:

```text
/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json
```

Command:

```bash
python scripts/probe_forward_valid_camera_masks.py \
  --manifest <manifest> \
  --camera-ids \
    camera_front_wide_120fov camera_cross_left_120fov \
    camera_cross_right_120fov camera_left_wide_90fov \
    camera_right_wide_90fov camera_back_rear_wide_90fov \
    camera_rear_left_70fov camera_front_standard_55fov \
    camera_front_tele_30fov
```

| Camera | Model | Non-finite repaired | Applied | Forward-valid kept | Kept % |
|---|---|---:|:---:|---:|---:|
| front wide | OpenCVPinhole | 0 | true | 1,309,462 / 2,073,600 | 63.1492% |
| cross left | OpenCVPinhole | 0 | true | 1,311,515 / 2,073,600 | 63.2482% |
| cross right | OpenCVPinhole | 0 | true | 1,303,756 / 2,073,600 | 62.8740% |
| left wide | OpenCVPinhole | 1 | true | 1,338,593 / 2,073,600 | 64.5541% |
| right wide | OpenCVPinhole | 0 | true | 1,333,878 / 2,073,600 | 64.3267% |
| back rear wide | OpenCVPinhole | 0 | true | 1,308,755 / 2,073,600 | 63.1151% |
| rear left | OpenCVPinhole | 0 | true | 1,313,030 / 2,073,600 | 63.3213% |
| front standard | OpenCVPinhole | 0 | true | 2,073,600 / 2,073,600 | 100.0000% |
| front tele | OpenCVPinhole | 0 | true | 2,073,578 / 2,073,600 | 99.9989% |

The `camera_left_wide_90fov` rational pole remains correctly handled: one inverse ray is repaired and remains invalid after the forward-valid AND.

## PAI FTheta Strict No-Op Probe

Manifest was discovered from the live inceptio filesystem:

```text
/home/inceptio/work/data/9ae151dc/pai_9ae151dc-e87b-41a7-8e85-71772f9603d7.json
```

All five tested cameras reported:

```text
model=FThetaCameraModel
nonfinite=0
applied=False
unchanged=True
application_kept_pct=100.0000
```

Tested cameras:

- `camera_front_wide_120fov`
- `camera_rear_tele_30fov`
- `camera_cross_left_120fov`
- `camera_cross_right_120fov`
- `camera_rear_left_70fov`

This proves the application path is skipped, rather than merely producing an all-true FTheta mask.

## Configuration Parity

`mask_forward_invalid_pixels` is passed with default `false` through all three factory paths:

- train `NCoreDataset`;
- val `NCoreDataset`;
- `make_test()` / offline render `NCoreDataset`.

The static AST regression test pins all three occurrences to the same key and false default.

## Automated Gates

- Focused forward-mask + factory + non-finite tests: `21 passed`.
- Full Mac suite: `1034 passed, 2 skipped, 3 existing warnings`.
- `py_compile`: clean for helper, dataset, factory, and probe.
- `git diff --check`: clean.
- Mermaid half-width-parenthesis gate: clean.

## Handoff

PIN-MASK-1 moves to Review after the final full-suite run. PIN-AB-1 must compare:

```text
Arm A: dataset.mask_forward_invalid_pixels=false
Arm B: dataset.mask_forward_invalid_pixels=true
```

using the same reviewed commit, 9-camera R6t recipe, tele weight 2.0, depth-off, `num_workers=10`, seed, iterations, and 5-second train/val windows.

Overall masked metrics have different denominators and must not be compared naively. The A/B report must include fixed radial bins and common forward-valid-domain metrics.
