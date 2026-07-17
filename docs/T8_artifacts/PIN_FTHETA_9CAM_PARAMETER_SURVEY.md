# PIN-FTHETA 9-Camera Parameter Survey

## Provenance

- Clip: `inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9`
- Manifest SHA-256: `df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31`
- Fitter: `pin-ftheta-numpy-v3-physical-domain-2026-07-17`
- Evaluation: every native-resolution integer pixel, all azimuths; no spatial downsampling.
- OpenCV validity: NCore `0.8 < icD < 1.2` and only the first monotonic/invertible branch from the optical axis. Later low-residual roots are invalid.
- Regions fixed before evaluation: center `r<0.5`, periphery `r>=0.9`, with `r` normalized by image half-diagonal.
- Coverage is reported against that physical OpenCV domain. The roughly 63% wide-camera coverage is expected and is not compared with an idealized 100% image domain.

## Declared Gate

| Metric | Strict threshold |
|---|---:|
| `nonradial_floor_mean_deg` | < 0.01 |
| `forward_poly_max_px` | < 1.5 |
| `mean_deg` | < 0.02 |
| `p95_deg` | < 0.04 |
| `p99_deg` | < 0.08 |
| `max_deg` | < 0.15 |
| `outer_p99_deg` | < 0.1 |

## Per-Camera Result

| Camera | p1 | p2 | nonradial mean/max deg | angular mean/p50/p95/p99/max deg | pixel mean/p50/p95/p99/max px | forward max px | physical/retained coverage | invalid pixels | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| `camera_front_wide_120fov` | 4.692e-05 | 8.771e-06 | 0.00185/0.00869 | 0.00186/0.00148/0.00485/0.00584/0.00748 | 0.03061/0.02453/0.07868/0.09435/0.11936 | 0.0021 | 0.631471/0.995991 | 769432 | 🟢 |
| `camera_cross_left_120fov` | 1.707e-04 | -2.189e-05 | 0.00598/0.02276 | 0.00601/0.00502/0.01517/0.01979/0.02597 | 0.09853/0.08375/0.24432/0.31636/0.41098 | 0.0039 | 0.632474/0.997097 | 765908 | 🟢 |
| `camera_cross_right_120fov` | 5.826e-05 | 1.882e-05 | 0.00221/0.01007 | 0.00232/0.00197/0.00570/0.00781/0.00958 | 0.03785/0.03278/0.09090/0.12320/0.14994 | 0.0049 | 0.628745/0.997212 | 773470 | 🟢 |
| `camera_left_wide_90fov` | -1.810e-05 | 6.179e-05 | 0.00379/0.01195 | 0.00320/0.00191/0.01130/0.01524/0.01785 | 0.05470/0.03393/0.18619/0.24577/0.28450 | 0.0012 | 0.645493/0.999440 | 735855 | 🟢 |
| `camera_right_wide_90fov` | 6.033e-05 | 3.319e-05 | 0.00433/0.01706 | 0.00387/0.00298/0.00898/0.01057/0.01238 | 0.06722/0.05231/0.15139/0.17533/0.20358 | 0.0010 | 0.643184/0.999933 | 739983 | 🟢 |
| `camera_back_rear_wide_90fov` | -1.111e-05 | -4.177e-05 | 0.00223/0.00701 | 0.00327/0.00242/0.00952/0.01175/0.01318 | 0.05302/0.04006/0.15100/0.18528/0.20712 | 0.0032 | 0.631152/0.996340 | 769633 | 🟢 |
| `camera_rear_left_70fov` | 7.309e-05 | -2.412e-05 | 0.00273/0.01193 | 0.00290/0.00197/0.00872/0.01184/0.01543 | 0.04736/0.03287/0.14054/0.18883/0.24373 | 0.0035 | 0.633203/0.995725 | 766204 | 🟢 |
| `camera_front_standard_55fov` | -1.580e-03 | -1.501e-04 | 0.02049/0.10276 | 0.02076/0.01645/0.05510/0.07919/0.11405 | 0.66041/0.53747/1.69082/2.37338/3.30814 | 0.1596 | 1.000000/0.999985 | 32 | 🔴 |
| `camera_front_tele_30fov` | 7.565e-05 | 4.847e-04 | 0.00229/0.00687 | 0.00464/0.00371/0.00980/0.02448/0.04760 | 0.29995/0.23830/0.63053/1.60994/5.97851 | 6.5084 | 1.000000/0.999755 | 508 | 🔴 |

## Center and Periphery

| Camera | center mean/p50/p95/p99/max deg | peripheral mean/p50/p95/p99/max deg | center mean/p50/p95/p99/max px | peripheral mean/p50/p95/p99/max px |
|---|---:|---:|---:|---:|
| `camera_front_wide_120fov` | 0.00137/0.00097/0.00406/0.00506/0.00565 | N/A/N/A/N/A/N/A/N/A | 0.02266/0.01613/0.06622/0.08179/0.09088 | N/A/N/A/N/A/N/A/N/A |
| `camera_cross_left_120fov` | 0.00422/0.00343/0.01100/0.01398/0.01585 | N/A/N/A/N/A/N/A/N/A | 0.06933/0.05722/0.17888/0.22609/0.25558 | N/A/N/A/N/A/N/A/N/A |
| `camera_cross_right_120fov` | 0.00161/0.00138/0.00391/0.00456/0.00504 | N/A/N/A/N/A/N/A/N/A | 0.02643/0.02286/0.06316/0.07340/0.08084 | N/A/N/A/N/A/N/A/N/A |
| `camera_left_wide_90fov` | 0.00213/0.00171/0.00568/0.00760/0.00871 | N/A/N/A/N/A/N/A/N/A | 0.03725/0.03013/0.09740/0.12813/0.14546 | N/A/N/A/N/A/N/A/N/A |
| `camera_right_wide_90fov` | 0.00318/0.00275/0.00777/0.00935/0.01036 | N/A/N/A/N/A/N/A/N/A | 0.05556/0.04845/0.13294/0.15748/0.17287 | N/A/N/A/N/A/N/A/N/A |
| `camera_back_rear_wide_90fov` | 0.00238/0.00197/0.00582/0.00678/0.00742 | N/A/N/A/N/A/N/A/N/A | 0.03896/0.03260/0.09396/0.10923/0.11934 | N/A/N/A/N/A/N/A/N/A |
| `camera_rear_left_70fov` | 0.00200/0.00130/0.00649/0.00813/0.00918 | N/A/N/A/N/A/N/A/N/A | 0.03283/0.02165/0.10555/0.13138/0.14790 | N/A/N/A/N/A/N/A/N/A |
| `camera_front_standard_55fov` | 0.00870/0.00748/0.02228/0.02812/0.03157 | 0.06584/0.06464/0.09962/0.10842/0.11405 | 0.28568/0.24698/0.73022/0.91740/1.02751 | 1.97838/1.93901/2.92806/3.16176/3.30814 |
| `camera_front_tele_30fov` | 0.00262/0.00267/0.00501/0.00664/0.00918 | 0.02069/0.02070/0.03363/0.03455/0.04760 | 0.16833/0.17115/0.32225/0.42668/0.58792 | 1.38118/1.36732/2.26770/2.33514/5.97851 |

## Inverse Round-Trip and Invalid Coverage

| Camera | OpenCV round-trip p50/p95/p99/max px | OpenCV valid/invalid | comparison valid/invalid | outer samples |
|---|---:|---:|---:|---:|
| `camera_front_wide_120fov` | 2.800e-14/0.000e+00/1.608e-13/2.542e-13/5.684e-13 | 1309418/764182 | 1304168/769432 | 0 |
| `camera_cross_left_120fov` | 2.801e-14/0.000e+00/1.608e-13/2.542e-13/6.431e-13 | 1311499/762101 | 1307692/765908 | 0 |
| `camera_cross_right_120fov` | 2.719e-14/0.000e+00/1.608e-13/2.542e-13/6.043e-13 | 1303765/769835 | 1300130/773470 | 0 |
| `camera_left_wide_90fov` | 2.073e-14/0.000e+00/1.137e-13/2.344e-13/5.363e-13 | 1338494/735106 | 1337745/735855 | 0 |
| `camera_right_wide_90fov` | 2.194e-14/0.000e+00/1.137e-13/2.542e-13/5.684e-13 | 1333706/739894 | 1333617/739983 | 0 |
| `camera_back_rear_wide_90fov` | 2.835e-14/0.000e+00/1.608e-13/2.542e-13/5.713e-13 | 1308757/764843 | 1303967/769633 | 0 |
| `camera_rear_left_70fov` | 2.649e-14/0.000e+00/1.271e-13/2.542e-13/6.431e-13 | 1313009/760591 | 1307396/766204 | 0 |
| `camera_front_standard_55fov` | 3.854e-14/0.000e+00/2.274e-13/3.595e-13/1.017e-12 | 2073600/0 | 2073568/32 | 0 |
| `camera_front_tele_30fov` | 5.531e-14/0.000e+00/2.542e-13/5.684e-13/1.160e-11 | 2073600/0 | 2073092/508 | 0 |

## Tele Regression Diagnosis

The previously observed `8059 px` tele forward residual was a numerical artifact from fitting the pixel-radius Vandermonde in raw units (the `r^5` column is ill-conditioned). It is not a valid physical-branch calibration error. With both primary and fallback least-squares paths normalized, and with later rational roots excluded, the deterministic tele result is `6.5084 px`. This is far smaller than 8059 px but still exceeds the predeclared `<1.5 px` gate, so tele remains a real representation blocker.

## Decision

**STOP.** Failed cameras: `camera_front_standard_55fov`, `camera_front_tele_30fov`.

A STOP result blocks GPU training until the failed representation gate is explicitly resolved.
