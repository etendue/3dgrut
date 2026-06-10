---
name: viser-gui-4d
description: >
  Launch the 3dgrut interactive 4D web viewer (threedgrut_playground/viser_gui_4d.py)
  on a trained checkpoint so the user can visually inspect a reconstruction in a browser.
  Use this WHENEVER the user wants to eyeball / open / view / 看一下 / 可视化 a trained
  ckpt, visually compare two checkpoints (e.g. poseopt vs baseline, Fourier vs DC),
  sanity-check dynamic-actor motion / jitter / appearance, or says things like "用
  viser 打开", "open in the 4d viewer", "let me look at the model", "show me this run".
  Trigger even when the user names a ckpt/experiment without saying "viser" — if they
  want to SEE a 3dgrut result interactively, this is the skill. Handles host selection
  (inceptio / a800), the proven nohup launch pattern, server-up verification, the
  browser URL + SSH-tunnel fallback, running several ckpts on different ports, and
  cleanup. NOT for headless metric eval (that's render.py → metrics.json) and NOT for
  training (see the vast-train / standard training flow).
---

# Run viser_gui_4d on a checkpoint

The 4D viewer is a **viser web server** that loads a ckpt and renders it interactively
(time slider scrubs the sequence; dynamic actors move; free camera). It runs on a GPU
host (inceptio / a800), and the user connects from a browser. Your job is to launch it
reliably, confirm it's up, and hand the user a working URL — then clean up after.

## 1. Locate the checkpoint + its data manifest

A run dir looks like `…/output/<experiment>/pai_<clip>-<stamp>/`. Inside:
- final ckpt: `ours_30000/ckpt_30000.pt` (preferred) or `ckpt_last.pt`.
- The viewer also needs `--dataset_path <manifest>.json` (the NCore `pai_<clip>.json`)
  for multi-camera poses / the 4D metadata. The ckpt embeds a `viz_4d` block, but pass
  the manifest so the camera dropdown / Follow-Camera work.

Find them on the host:
```bash
ssh <host> 'find <out_dir>/<experiment>* -name "ckpt_30000.pt" 2>/dev/null'
```

## 2. Pick the host + activate conda (every ssh)

Run the viewer on the host where the ckpt + GPU live. Non-interactive ssh does NOT
inherit conda — always export the env PATH first (see CLAUDE.md):
- **inceptio**: `export PATH=/home/inceptio/miniforge3/envs/3dgrut2/bin:$PATH`,
  data under `/home/inceptio/work/...`.
- **a800-x2**: `export PATH=/root/miniforge3/envs/3dgrut/bin:$PATH`,
  data under `/root/work/yusun/ncore-nurec/...`.

**Renderer backend by GPU (CRITICAL — wrong choice = segfault).** The viewer's
`--renderer` defaults to `3dgrt` (OptiX ray tracing, **needs RT cores**). Pick by host GPU:
- **A100 / A800** (datacenter, NO RT cores) → **MUST pass `--renderer 3dgut`** (tile
  rasterization, no OptiX). The default `3dgrt` **segfaults** here (OptiX
  `libplayground_cc` dlopen cores: log shows `Loading extension module libplayground_cc...`
  then `dumped core`). 3dgut renders ego / cuboid / LiDAR + the 4D timeline fine.
- **RTX (4090) / inceptio / Hopper (H100/H800)** (have RT cores) → default `3dgrt` is fine
  (OptiX works); pass it explicitly or omit.

VRAM: each viewer instance loads ~7.5 GB; two fit on a 24 GB card.

## 3. Launch (proven inline-nohup pattern)

ssh to these hosts is occasionally flaky (random `exit 255`). Use the **simple inline
nohup + `echo PID`** form — it's the most robust. Do NOT use `setsid bash <script> &
disown`; that half-dies under connection blips. Add `timeout <secs>` as a safety
auto-kill so a forgotten viewer doesn't hold the GPU forever (the user can ask to
extend / relaunch).

```bash
ssh <host> "export PATH=<env-bin>:\$PATH && export CUDA_VISIBLE_DEVICES=0 \
  && cd <repo> && rm -f /tmp/viser_<tag>.log \
  && nohup timeout 3600 python threedgrut_playground/viser_gui_4d.py \
       --gs_object <ckpt.pt> --dataset_path <manifest.json> --port 8090 \
       --renderer 3dgut \
       > /tmp/viser_<tag>.log 2>&1 & echo PID \$!"
```

> `--renderer 3dgut` shown above is for **A100/A800** (no RT cores). On **RTX/4090/H100**
> drop the flag or use `--renderer 3dgrt` (OptiX, the default). See §2 — getting this
> wrong segfaults the viewer at OptiX dlopen.

**Several ckpts at once** (visual A/B): launch one per **distinct port** (8090, 8091, …),
each with its own `/tmp/viser_<tag>.log`. They share GPU0 fine (mind the ~7.5 GB each).
Launch each as its OWN ssh command — do NOT chain two launches in one ssh line; a 255
blip on the first kills the second (observed). Verify each separately.

## 4. Verify the server is actually up

Don't trust the launch echo alone (the connection may drop after it). Poll the log:

```bash
ssh <host> 'grep -iE "listening|Traceback|Error" /tmp/viser_<tag>.log'
```

Success looks like `loaded schema_v2 (N tracks …)` then a viser box
`viser (listening *:8090)`. Also confirm the port is bound:
```bash
ssh <host> 'ss -ltn | grep -c :8090'   # → 1 when up
```

## 5. Give the user the URL

The viewer binds to all interfaces, so direct IP usually works from a same-network Mac:
- **http://<host-ip>:<port>** (inceptio = `10.8.31.113`; check `~/.ssh/config` HostName for others).

**If direct IP is blocked** (firewall), offer the SSH-tunnel fallback:
```bash
ssh -L 8090:localhost:8090 <host>     # then open http://localhost:8090
```
For several instances, tunnel each port (`-L 8090:localhost:8090 -L 8091:localhost:8091`).

## 6. Cleanup (when the user is done)

```bash
ssh <host> 'pkill -9 -f "viser_gui_4d.py"'
ssh <host> 'echo "ports:$(ss -ltn|grep -cE ":8090|:8091") GPU:$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"'
```
Confirm `ports:0` + GPU back to idle (~28 MiB). Note: `pgrep -fc viser_gui_4d.py` may
report `1` even when down — that's the grep command **self-matching** its own argv, not
a live viewer. Trust the port + GPU numbers over pgrep counts.

## Gotchas (learned the hard way)

- **`--renderer` backend must match the GPU (A800 segfault, 2026-06-10).** Default
  `3dgrt` uses OptiX (needs RT cores). On **A100/A800** the OptiX extension dlopen
  **segfaults** (`Loading extension module libplayground_cc...` → `dumped core`) — you
  MUST pass **`--renderer 3dgut`** (tile rasterization). RTX/4090/H100 have RT cores so
  the default is fine. The code documents this at `viser_gui_4d.py:1532` (`--renderer`
  choices) + the `--no_gaussian_render` help, but it's easy to miss — always set it
  explicitly per host.
- **sky_backend must match the ckpt.** Ckpts trained with `trainer.sky_backend=mlp`
  (the A800/inceptio default, nvdiffrast unavailable) load a `SkyEnvmapMLP`; the viewer
  auto-detects from the ckpt. A mismatch → state_dict shape error on first render. If
  you see sky-weight shape errors, it's this, not the ckpt being bad.
- **ssh `exit 255`** is transient — retry; verify state in a fresh ssh rather than
  trusting a cut-off output.
- **Local `~` expands on the Mac**, not the remote. In ssh command strings use absolute
  remote paths (`/home/inceptio/...`), or the viewer will assert on a `/Users/...` path.
- **Fourier-albedo / poseopt ckpts render correctly** — time-varying albedo and dynamic
  poses both resolve their frame `t` from the same `timestamp_us → _resolve_pose_idx`
  path the viewer already drives, so no special handling is needed.
