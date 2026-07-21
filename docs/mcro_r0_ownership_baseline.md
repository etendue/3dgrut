# MCRO R0 ownership baseline

**Status:** frozen read-only R0 baseline.  No training was run and neither
checkpoint was modified.

## Protocol

Both frozen checkpoints were evaluated on the 24 held-out
`camera_front_wide_120fov` frames.  Each checkpoint was rendered three times
with exactly one enabled layer: `background`, `road`, and `sky_envmap`.
The evaluator saves per-frame Gaussian alpha, semantic road support, and sky
contribution; metrics use the eroded road interior (`erosion_px=1`).

Artifacts on inceptio are rooted at
`/home/inceptio/work/output/mcro_r0_ownership/`; the machine-readable reports
are `1cam/ownership.json` and `6cam/ownership.json`.

## R0 distribution

| checkpoint | bg alpha on road mean | road alpha P10 | road alpha P50 | road alpha mean | sky energy on road |
|---|---:|---:|---:|---:|---:|
| 1-cam | 0.2694 | 0.2271 | 0.5554 | 0.5725 | 0.0000 |
| 6-cam | 0.9905 | 0.3717 | 0.8897 | 0.7800 | 0.0000 |

The 6-cam checkpoint assigns almost opaque background coverage to the road
interior in this isolated render, while the 1-cam baseline does not.  The
road-only alpha distribution is also different, but is not by itself a
quality ranking: the Task 7 full-frame 1-cam re-render parity issue remains
open.  This R0 is an ownership reference, not a replacement for that parity
gate.

## Frozen B6 guards

`configs/eval/mcro_ownership_guards.json` is the source of truth.  The road
P10 guard is **0.37**, rather than the pre-R0 placeholder 0.8: observed
isolated-road alpha is materially below 0.8 even for the 6-cam baseline.
Future ownership arms must reduce background-on-road alpha by at least 50%,
keep road P10 at or above 0.37, keep sky energy on road at or below 0.001,
and satisfy the existing full-image and road-crop quality guards.  These
thresholds are frozen after R0 and must not be relaxed for a later arm.

## Remaining visual diagnostic attachment

The layer artifacts needed for full/background/road/sky four-panel samples
are available in the R0 directories above.  Novel-view assessment remains
subject to the existing 1-cam full-render parity gate, so it must be reported
as a diagnostic visual (not a 1-cam-vs-6-cam quality conclusion).
