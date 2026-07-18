# PIN-FTHETA v3 Invalidated Evidence Inventory

> **Read-only recovery date:** 2026-07-18
> **Disposition:** Preserve as historical evidence. Do not use any entry below
> to support the FTheta v4 Arm F decision.

Every recovered FTheta run below is bound to the v3 parameter artifact
`73965c6d10693b7742c7afb538169990a94d2457890738a7a4f66d13fcd0a450`.
That artifact imported the Pinhole `0.8 < icD < 1.2` runtime trust gate into
the OpenCV-to-FTheta conversion and limited front-wide to approximately
`41.84°` (`0.730310... rad`). Consequently, the smoke, full-training, native
render, and any unbound interactive-viewer observations below are invalid for
the v4 full-domain conclusion. This inventory records their identity without
rewriting any remote manifest, checkpoint, render, or parameter artifact.

## Common inputs

| Item | Exact path/value | SHA-256 |
|---|---|---|
| v3 FTheta artifact | `/home/inceptio/repo/3dgrut2-wt/ftheta-7cam-smoke/scripts/pin_ftheta_b6a9_7cam_params.json` | `73965c6d10693b7742c7afb538169990a94d2457890738a7a4f66d13fcd0a450` |
| Base config | `/home/inceptio/repo/3dgrut2-wt/ftheta-7cam-smoke/configs/apps/ncore_3dgut_mcmc_multilayer_inceptio_7cam.yaml` | `4f5718158a1883fe806f21916303205aa8c784f78d8014498c42541d2bed76ad` |
| Dataset manifest | `/home/inceptio/work/data/inc_b6a9ed61_20s/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9.json` | `df2021203cfe318cfa8da3462e38c5b7fbf6bf3963d3a8149d145f98f6036e31` |

## v3 5-second / 5k smoke

- Run root: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475`
- Run manifest: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/run_manifest.json`
- Run-manifest SHA-256: `4e6c980560c8deab46f2ec21d3b9d3763c8cb7ff886f48f177d4630983478894`
- Schema/status: `2` / `complete`
- Created/completed UTC: `2026-07-17T06:03:22.164561+00:00` / `2026-07-17T06:29:23.673540+00:00`
- Git commit: `778a68dbe1ec89eabfadf7cc161a5e9e1bc58aaa` (`fix: validate wrapped FTheta override logs`)
- Normalized scientific-config SHA-256: `d123655f5602ab784dd4864466f7d79170bac05464085ca09c2b28ad16b02589`
- Representation comparison flag: `only_representation_path_differs=true`
- Driver SHA-256: `f9a51d645b4704c84019aaa84769f14e0e53a008339251fb2b39e4e708a44a6f`
- Validator SHA-256: `4ac5cebe2a9e7d6d38fee90d191b6dc1c3641f43aee84384f9b2fdbe0e0bf431`

These smoke driver/validator source hashes identify the blobs at the recorded
run commit; they are not hashes of a current mutable worktree path.

### Arm P

- Checkpoint: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/train_outputs/pin_ftheta_7cam_armP_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_140616/ckpt_last.pt`
- Checkpoint SHA-256: `f5fc7e024d5fdb31ec2e35d2eab87955cd19bb13bedc25c3744685dfbc863ea7`
- Global step: `5000`
- Resolved `parsed.yaml` SHA-256: `7a528e3f69aff3da823a51a39cdc9e8d54ddff7f40e611b2e7df07c5944d0a58`
- Metrics: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/eval_outputs/pin_ftheta_7cam_armP_5s_5k/pin_ftheta_7cam_armP_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_141521/metrics.json`
- Metrics SHA-256: `3f3801303262c7e49ac5ebe4b82fdc4dd3b7007b92511e526f377623c065dd68`
- Native renders: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/eval_outputs/pin_ftheta_7cam_armP_5s_5k/pin_ftheta_7cam_armP_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_141521/ours_5000/renders`
- Render count/tree SHA-256: `44` / `8fb504e83b0671180839bedec549daa92128fb4302ed999ecf8ae099ee6fcd77`
- Native GT: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/eval_outputs/pin_ftheta_7cam_armP_5s_5k/pin_ftheta_7cam_armP_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_141521/ours_5000/gt`
- GT count/tree SHA-256: `44` / `a0b9758eb5151dae7f102bf97df9e579ab30392380e5cebc2e57c3ce1e21be6d`

### Arm F

- Checkpoint: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/train_outputs/pin_ftheta_7cam_armF_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_141911/ckpt_last.pt`
- Checkpoint SHA-256: `f8f6db0fcfc979bc6fcd54aee8d5ddfbd41d6f4f023599f2716836f438c91b09`
- Global step: `5000`
- Resolved `parsed.yaml` SHA-256: `e67963eebc62aff243768e6661f9c1df49b4c313ac2c2451e02e3225b4412d27`
- Metrics: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/eval_outputs/pin_ftheta_7cam_armF_5s_5k/pin_ftheta_7cam_armF_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_142727/metrics.json`
- Metrics SHA-256: `83865679942a66414565e4ef6681b02a1037fbc9819d634d604e6bb1b5184501`
- Native renders: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/eval_outputs/pin_ftheta_7cam_armF_5s_5k/pin_ftheta_7cam_armF_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_142727/ours_5000/renders`
- Render count/tree SHA-256: `44` / `de2bebf3cc613ca4ce631236eb8b6680b0d3dff7c7eb8bba4dcf117b475772a9`
- Native GT: `/home/inceptio/work/output/pin_ftheta_smoke_runs/20260717T060319Z_1784268199582296060_pid2536459_r31475/eval_outputs/pin_ftheta_7cam_armF_5s_5k/pin_ftheta_7cam_armF_5s_5k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_142727/ours_5000/gt`
- GT count/tree SHA-256: `44` / `a0b9758eb5151dae7f102bf97df9e579ab30392380e5cebc2e57c3ce1e21be6d`

The schema-v2 smoke manifest contains no `native_render` or
`native_render_inventory` binding. The four smoke tree hashes above were
recomputed during this read-only recovery using the same algorithm as
`scripts/pin_ftheta_full_ab_validation.py::_tree_sha256()`; they are not
formal fields in the original run manifest.

## v3 full 20-second / 30k A/B

- Run root: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458`
- Run manifest: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/run_manifest.json`
- Run-manifest SHA-256: `fe9d52c2220de08f17c4152073bfc715a961f6f215aba54b9407204e5ec347b8`
- Schema/status/profile: `3` / `complete` / `pin_ftheta_7cam_full_20s_30k_v1`
- Created/completed UTC: `2026-07-17T08:31:43.696325+00:00` / `2026-07-17T10:42:40.563552+00:00`
- Git commit: `2e1c4901392b659b8c14eb0c7c8707b1a8eaa597` (`fix: accept wrapped FTheta checkpoint basenames`)
- Normalized scientific-config SHA-256: `3b027665a8a9beaf4ffa4923609f7cefa776bf73de85549f0cf5a5012d3a577d`
- Representation comparison flag: `only_representation_path_differs=true`
- Driver SHA-256: `eac69e8af57fdd60de200a5405e0b8b72e644c59c2f8f39eb360cbc9a0793300`
- Full validator SHA-256: `5aca07c4212dc54ef18cbd32a54529b89a9e2ae3551c80bc165c8366221fc123`

### Arm P

- Checkpoint: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/train_outputs/pin_ftheta_7cam_armP_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_163735/ckpt_last.pt`
- Checkpoint SHA-256: `73b6c5cb4d0272559115c7cb7dca97d5ce6e97f6a16d8de5775850835c02a0d0`
- Global step: `30000`
- Resolved `parsed.yaml` SHA-256: `3836cb198bdf3aaf4d119dcd316303babbfa6bdcac791295d0a80687ab64420f`
- Metrics: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/eval_outputs/pin_ftheta_7cam_armP_full_30k/pin_ftheta_7cam_armP_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_173113/metrics.json`
- Metrics SHA-256: `da05c0d2f3d7eef0331ab84e53ef4c2b64abee131a3dcd5198dabfe77dc52aea`
- Native renders: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/eval_outputs/pin_ftheta_7cam_armP_full_30k/pin_ftheta_7cam_armP_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_173113/ours_30000/renders`
- Render count/tree SHA-256: `168` / `613fb688a4b56c6ece56ebab1daace1b56401c399d3c80dadb70c868510a162b`
- Native GT: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/eval_outputs/pin_ftheta_7cam_armP_full_30k/pin_ftheta_7cam_armP_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_173113/ours_30000/gt`
- GT count/tree SHA-256: `168` / `2d7bbe011188693cf91984f1f3d8bbe3fa4acef48fd19e118e78f2d11dbd73ef`
- Native inventory: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/arms/P/native_render_inventory.json`
- Native-inventory SHA-256: `c952938333d1ac7709edaadfe93c4565ff2b188e026c5e238ca2c4a74ae5bdc2`

### Arm F

- Checkpoint: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/train_outputs/pin_ftheta_7cam_armF_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_174223/ckpt_last.pt`
- Checkpoint SHA-256: `43ad60529e89e2c3045a5f8c27abdf10d424f3cd15c94a2479ef921a972a5918`
- Global step: `30000`
- Resolved `parsed.yaml` SHA-256: `ff7715446503f35074fbae1871eda79f678b1a5e163e68cf860f47ab083d9b67`
- Metrics: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/eval_outputs/pin_ftheta_7cam_armF_full_30k/pin_ftheta_7cam_armF_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_183625/metrics.json`
- Metrics SHA-256: `c731a0dd2e15b4458f148a094aabaa95a54b20612716b8a498434f62e62ecc9f`
- Native renders: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/eval_outputs/pin_ftheta_7cam_armF_full_30k/pin_ftheta_7cam_armF_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_183625/ours_30000/renders`
- Render count/tree SHA-256: `168` / `13199931d9c7ce3ac113a26461f7a68b4d26dc99a7ea7fb1f64ed6e5619cb3a6`
- Native GT: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/eval_outputs/pin_ftheta_7cam_armF_full_30k/pin_ftheta_7cam_armF_full_30k/inceptio_b6a9ed61-8952-4b0c-90d8-fd2893e849e9-1707_183625/ours_30000/gt`
- GT count/tree SHA-256: `168` / `2d7bbe011188693cf91984f1f3d8bbe3fa4acef48fd19e118e78f2d11dbd73ef`
- Native inventory: `/home/inceptio/work/output/pin_ftheta_phase4_full_ab_2e1c490_relaunch1/20260717T083141Z_1784277101115348359_pid2592731_r17458/arms/F/native_render_inventory.json`
- Native-inventory SHA-256: `274399bc44c3acd718974ed10422a51321d6d978693ee9db29a9167f9e992eef`

The full Arm P/F GT filename sets, contents, counts, and tree hashes match
exactly. The four full render/GT tree hashes were recomputed during recovery
and agree with the schema-v3 manifest's formal inventory bindings.

## Viser / viewer evidence status

There is **no separately hashed or manifest-bound Viser evidence** in either
recovered run. Neither manifest has a `viser`, `viewer`, `screenshot`, or
browser-evidence field, and neither run root has a matching `*viser*`,
`*viewer*`, `*screenshot*`, or `*.html` file.

This statement does **not** claim that no temporary interactive Viser session
was launched. It only establishes that any such temporary session or screenshot
was not bound to these manifests and therefore is not reproducible evidence.

## Audit notes

- All checkpoint, metrics, native-inventory, artifact, config, manifest, and
  source hashes above were rechecked read-only on `inceptio` on 2026-07-18.
- Tree hashes use the relative-name length, relative name, and per-file SHA-256
  algorithm implemented by
  `scripts/pin_ftheta_full_ab_validation.py::_tree_sha256()`.
- Nothing in this inventory authorizes metadata-swapping, parameter-file
  replacement, or reuse of a v3 Arm F checkpoint in the v4 experiment.
