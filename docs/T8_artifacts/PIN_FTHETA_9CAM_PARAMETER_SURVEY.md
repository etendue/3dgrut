# PIN-FTHETA 9-Camera Parameter Survey

> **Decision policy update (user-approved, 2026-07-18):**
> [`2026-07-18-ftheta-v4-full-domain-retrain-fix.md`](../superpowers/plans/2026-07-18-ftheta-v4-full-domain-retrain-fix.md)
> supersedes the old hard-STOP interpretation below. For the selected seven
> cameras, residual exceedances are accepted quality warnings; structural,
> domain, provenance, monotonicity, and data failures remain hard stops.
> Front-standard/front-tele remain excluded from both training arms.
>
> Results made with `pin-ftheta-numpy-v3-physical-domain`, front-wide
> `max_angle=41.84°` (`0.730310... rad`), or artifact SHA-256 prefix
> `73965c6d...` are invalid for the v4 decision. Preserve historical paths and
> manifests; do not rewrite them.

## Provenance

- Clip: `inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9`
- Manifest SHA-256: `df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31`
- Fitter: `pin-ftheta-numpy-v4-full-calibration-domain-2026-07-18`
- Evaluation: every native-resolution integer pixel, all azimuths; no spatial downsampling.
- Calibration validity: the complete first monotonic/invertible rational branch from the optical axis. The Pinhole renderer's `0.8 < icD < 1.2` runtime gate is deliberately not applied to FTheta fitting or validation.
- Regions fixed before evaluation: center `r<0.5`, periphery `r>=0.9`, with `r` normalized by image half-diagonal.
- Coverage is measured against the full calibration branch and the native image raster. A roughly 63% wide-camera result is a hard failure, not an expected limitation.

## Historical Quality-Warning Thresholds

These thresholds remain reported. A red symbol in the raw table means a
quality warning, not a hard runtime failure, for an active seven-camera row.

| Metric | Warning threshold |
|---|---:|
| `nonradial_floor_mean_deg` | < 0.01 |
| `forward_poly_max_px` | < 1.5 |
| `mean_deg` | < 0.02 |
| `p95_deg` | < 0.04 |
| `p99_deg` | < 0.08 |
| `max_deg` | < 0.15 |
| `outer_p99_deg` | < 0.1 |

## Per-Camera Result

| Camera | p1 | p2 | nonradial mean/max deg | angular mean/p50/p95/p99/max deg | pixel mean/p50/p95/p99/max px | forward max px | calibration-domain/retained comparison | comparison invalid pixels | Quality threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `camera_front_wide_120fov` | 4.692e-05 | 8.771e-06 | 0.00424/0.04207 | 0.01143/0.01041/0.02553/0.05222/0.11902 | 0.18128/0.16853/0.38934/0.77338/1.91144 | 1.2625 | 1.000000/0.999929 | 148 | 🟢 |
| `camera_cross_left_120fov` | 1.707e-04 | -2.189e-05 | 0.01460/0.14246 | 0.01748/0.01126/0.05667/0.08097/0.23817 | 0.28384/0.18852/0.88699/1.25895/3.79494 | 1.0348 | 1.000000/0.999933 | 138 | 🔴 |
| `camera_cross_right_120fov` | 5.826e-05 | 1.882e-05 | 0.00561/0.05937 | 0.01412/0.01256/0.03146/0.06806/0.13718 | 0.22315/0.20291/0.47545/1.01140/2.24375 | 1.4717 | 1.000000/0.999936 | 133 | 🟢 |
| `camera_left_wide_90fov` | -1.810e-05 | 6.179e-05 | 0.00960/0.27817 | 0.01924/0.01367/0.06340/0.08789/0.56393 | 0.29446/0.23084/0.92021/1.12182/2.79984 | 5.1183 | 0.984722/1.000000 | 31681 | 🔴 |
| `camera_right_wide_90fov` | 6.033e-05 | 3.319e-05 | 0.00752/0.23359 | 0.02176/0.01717/0.05530/0.10050/0.53298 | 0.33548/0.28877/0.80309/1.29040/2.40932 | 5.5201 | 0.975162/1.000000 | 51504 | 🔴 |
| `camera_back_rear_wide_90fov` | -1.111e-05 | -4.177e-05 | 0.00682/0.05657 | 0.01178/0.00765/0.04568/0.05018/0.18129 | 0.18201/0.12384/0.67836/0.74408/2.81709 | 0.9629 | 1.000000/0.999942 | 120 | 🔴 |
| `camera_rear_left_70fov` | 7.309e-05 | -2.412e-05 | 0.00691/0.07519 | 0.01251/0.00961/0.03742/0.05138/0.18675 | 0.19907/0.15698/0.56904/0.77155/2.97166 | 1.1664 | 1.000000/0.999951 | 101 | 🔴 |
| `camera_front_standard_55fov` | -1.580e-03 | -1.501e-04 | 0.02049/0.10276 | 0.02076/0.01645/0.05510/0.07919/0.11405 | 0.66041/0.53747/1.69082/2.37338/3.30814 | 0.1596 | 1.000000/0.999985 | 32 | 🔴 |
| `camera_front_tele_30fov` | 7.565e-05 | 4.847e-04 | 0.00229/0.00687 | 0.00464/0.00371/0.00980/0.02448/0.04760 | 0.29995/0.23830/0.63053/1.60994/5.97851 | 6.5084 | 1.000000/0.999755 | 508 | 🔴 |

Accepted warning inventory for the selected seven cameras:

| Camera | Exceeded warning metrics |
|---|---|
| front-wide | none |
| cross-left | non-radial mean, p95, p99, max, outer p99 |
| cross-right | none |
| left-wide | forward max, p95, p99, max, outer p99 |
| right-wide | forward max, mean, p95, p99, max, outer p99 |
| back-rear-wide | p95, max |
| rear-left | max |

## Accepted v4 Coverage Baselines

| Camera | FTheta domain | OpenCV calibration domain | Pixels outside FTheta `max_angle` |
|---|---:|---:|---:|
| front-wide | 99.9929% | 100.0000% | 148 |
| cross-left | 99.9933% | 100.0000% | 138 |
| cross-right | 99.9936% | 100.0000% | 133 |
| left-wide | 98.7290% | 98.4722% | 26,355 |
| right-wide | 97.8640% | 97.5162% | 44,292 |
| back-rear-wide | 99.9942% | 100.0000% | 120 |
| rear-left | 99.9951% | 100.0000% | 101 |

The legacy serialized name `physical_domain_retention` is a calibration-domain
comparison in v4. It must not be described as the Pinhole renderer's
`0.8 < icD < 1.2` runtime domain.

## Center and Periphery

| Camera | center mean/p50/p95/p99/max deg | peripheral mean/p50/p95/p99/max deg | center mean/p50/p95/p99/max px | peripheral mean/p50/p95/p99/max px |
|---|---:|---:|---:|---:|
| `camera_front_wide_120fov` | 0.00779/0.00812/0.01348/0.01457/0.02820 | 0.04005/0.03727/0.07003/0.07304/0.11902 | 0.12768/0.13286/0.22208/0.23952/0.46907 | 0.60806/0.56723/1.03492/1.07964/1.91144 |
| `camera_cross_left_120fov` | 0.00761/0.00726/0.01438/0.01710/0.02248 | 0.07471/0.07093/0.11383/0.16798/0.23817 | 0.12516/0.11944/0.23572/0.27587/0.37477 | 1.18864/1.12873/1.74188/2.64607/3.79494 |
| `camera_cross_right_120fov` | 0.00934/0.00980/0.01626/0.01723/0.03420 | 0.05432/0.05342/0.09888/0.10243/0.13718 | 0.15265/0.15976/0.26699/0.27819/0.56769 | 0.82111/0.80488/1.46275/1.51467/2.24375 |
| `camera_left_wide_90fov` | 0.01003/0.01047/0.01754/0.01851/0.03664 | 0.07627/0.06928/0.22529/0.38395/0.56393 | 0.17342/0.18056/0.30445/0.32292/0.65044 | 0.80506/0.76067/1.76945/2.47923/2.79984 |
| `camera_right_wide_90fov` | 0.01303/0.01358/0.02319/0.02921/0.04409 | 0.09612/0.06195/0.30235/0.40162/0.53298 | 0.22419/0.23451/0.39585/0.48951/0.78115 | 0.80031/0.63786/1.89127/2.22251/2.40932 |
| `camera_back_rear_wide_90fov` | 0.00573/0.00583/0.01067/0.01121/0.01987 | 0.03567/0.03437/0.07684/0.12666/0.18129 | 0.09415/0.09551/0.17527/0.18429/0.33060 | 0.53068/0.50470/1.14649/1.92702/2.81709 |
| `camera_rear_left_70fov` | 0.00695/0.00718/0.01266/0.01375/0.02462 | 0.03606/0.03058/0.06564/0.11367/0.18675 | 0.11430/0.11771/0.20804/0.22644/0.41015 | 0.57539/0.48634/0.97838/1.77112/2.97166 |
| `camera_front_standard_55fov` | 0.00870/0.00748/0.02228/0.02812/0.03157 | 0.06584/0.06464/0.09962/0.10842/0.11405 | 0.28568/0.24698/0.73022/0.91740/1.02751 | 1.97838/1.93901/2.92806/3.16176/3.30814 |
| `camera_front_tele_30fov` | 0.00262/0.00267/0.00501/0.00664/0.00918 | 0.02069/0.02070/0.03363/0.03455/0.04760 | 0.16833/0.17115/0.32225/0.42668/0.58792 | 1.38118/1.36732/2.26770/2.33514/5.97851 |

## Inverse Round-Trip and Invalid Coverage

| Camera | OpenCV round-trip p50/p95/p99/max px | OpenCV valid/invalid | comparison valid/invalid | outer samples |
|---|---:|---:|---:|---:|
| `camera_front_wide_120fov` | 4.620e-14/0.000e+00/2.274e-13/3.595e-13/7.626e-13 | 2073600/0 | 2073452/148 | 286892 |
| `camera_cross_left_120fov` | 4.612e-14/0.000e+00/2.274e-13/3.595e-13/9.166e-13 | 2073600/0 | 2073462/138 | 282879 |
| `camera_cross_right_120fov` | 4.534e-14/0.000e+00/2.274e-13/3.595e-13/7.280e-13 | 2073600/0 | 2073467/133 | 292738 |
| `camera_left_wide_90fov` | 5.395e-14/0.000e+00/2.542e-13/4.857e-13/7.625e-10 | 2041919/31681 | 2041919/31681 | 187679 |
| `camera_right_wide_90fov` | 1.711e-13/0.000e+00/2.542e-13/5.084e-13/9.874e-08 | 2022096/51504 | 2022096/51504 | 173045 |
| `camera_back_rear_wide_90fov` | 4.638e-14/0.000e+00/2.274e-13/3.595e-13/8.198e-13 | 2073600/0 | 2073480/120 | 286895 |
| `camera_rear_left_70fov` | 4.452e-14/0.000e+00/2.274e-13/3.595e-13/8.039e-13 | 2073600/0 | 2073499/101 | 282730 |
| `camera_front_standard_55fov` | 3.854e-14/0.000e+00/2.274e-13/3.595e-13/1.017e-12 | 2073600/0 | 2073568/32 | 0 |
| `camera_front_tele_30fov` | 5.531e-14/0.000e+00/2.542e-13/5.684e-13/1.160e-11 | 2073600/0 | 2073092/508 | 0 |

## Tele Regression Diagnosis

The previously observed `8059 px` tele forward residual was a numerical artifact from fitting the pixel-radius Vandermonde in raw units (the `r^5` column is ill-conditioned). It is not a valid physical-branch calibration error. With both primary and fallback least-squares paths normalized, and with later rational roots excluded, the deterministic tele result is `6.5084 px`. This is far smaller than 8059 px but still exceeds the predeclared `<1.5 px` gate, so tele remains a real representation blocker.

## Decision

**PROCEED WITH WARNINGS for the selected seven-camera v4 experiment**, provided
all hard runtime invariants pass. The residual exceedances above remain visible
but do not independently block training.

This does not approve front-standard/front-tele or nine-camera training, and it
does not validate old v3 smoke/full/native/Viser evidence. New runs require
immutable versioned v4 artifacts and distinct output roots.
