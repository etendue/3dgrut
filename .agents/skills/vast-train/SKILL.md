---
name: vast-train
description: >
  Provision a rented GPU host on vast.ai and run a 3dgrut training job (smoke / KPI /
  A/B) on it. Use this WHENEVER the local GPU boxes (a800-x2, inceptio) are busy,
  contended, or out of memory and the user still wants to run training NOW — or when
  the user explicitly says "spin up a vast box", "rent a GPU", "跑到 vast 上", "起一个
  vast 实例", "use vast.ai for this run". Trigger when a training launch is blocked on
  local capacity and renting cloud GPU is the unblock. Covers the full lifecycle:
  create instance → fix the ssh-config polling bug → install env → transfer the NCore
  clip → launch training (nohup) → monitor → DESTROY the instance to stop billing.
  NOT for local runs on a800/inceptio (launch those directly), and NOT for viewing
  results (see viser-gui-4d).
---

# Run 3dgrut training on a vast.ai rented GPU

This is the documented fallback when a800-x2 / inceptio are unavailable. The full,
authoritative runbook lives in **[AGENTS.md](../../../AGENTS.md) → "Vast.ai 远程执行环境"**
— read it for exact commands and history. This skill is the workflow spine + the
pitfalls that waste the most time, so you execute it in the right order and don't
re-learn the traps.

**Cost discipline:** an RTX 4090 bills ~$0.534/hr even idle. A 5k smoke A/B ≈ $0.45.
**Always destroy the instance the moment the job is done** (last step). Never leave it
idle.

**Secrets:** the vast API key is already baked into `scripts/t8_12_fix_vast_create.sh`
(and the create command also accepts `--api-key $VAST_API_KEY`). Do NOT paste the key
into new files, logs, or commits — reference the script / env var.

## Tools
- `vastai` CLI: `/Users/etendue/repo/ncore/.venv/bin/vastai` (v1.0.3).
- HF token (if pulling data from HF): `~/.cache/huggingface/token`.

## Stage 1 — Create the instance + write ssh config

```bash
LABEL=<task>_smoke DISK_GB=100 MAX_DPH=0.80 bash scripts/t8_12_fix_vast_create.sh
```
Image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`. Prefer high-bandwidth hosts
(California / Norway, inet > 1 Gbps). **Avoid France `mid=67891`** (image pull hangs).

⚠️ **Known bug:** the script's status polling uses an old field-name match and hangs at
"timed out waiting" even though the instance IS created. Don't wait on it — pull the
real `ssh_host` / `ssh_port` from `vastai show instance <id> --raw` and write the
`~/.ssh/config` alias `vast-rtx4090` yourself (Python snippet in AGENTS.md).

## Stage 2 — Install the environment

The vast pytorch container has **no system `python3`** (only conda python) — don't run
`python3` to write files; use shell/awk or `scp` a Mac-written script over.

```bash
ssh vast-rtx4090 'apt-get install -y -qq git python3.11-venv rsync \
    libxcb1 libxext6 libxrender1 libsm6 libice6 libgl1 libglib2.0-0 \
  && cd /root && git clone https://github.com/etendue/3dgrut.git \
  && cd 3dgrut && git checkout <branch> && git submodule update --init --recursive \
  && bash install_env_uv.sh'
```
The `libxcb1 …` system libs are **mandatory** (opencv-python ImportError on `libxcb.so.1`
without them). `install_env_uv.sh` buffers upstream output with `tail -30` so it looks
"stuck" while actually compiling — monitor real progress via `ps -ef | grep nvcc` or
`du -sh .venv` (~10–15 min; slangc + kaolin install last).

## Stage 3 — Get the NCore clip onto the box

Prefer **reverse-rsync from a800** (it has the full clips, 3–5 MB/s, ~25 min for a
7.2 GB clip) over re-downloading from HF. Generate an a800 ssh keypair, add its pubkey
to vast `authorized_keys`, write a800's ssh config → `vast-rtx4090`, then:
```bash
ssh a800-x2 'rsync -avz --info=progress2 \
  /root/work/yusun/ncore-nurec/data/ncore/clips/<clip>/ \
  vast-rtx4090:/root/data/ncore/clips/<clip>/'
```
(Run as a background/long task; full keypair + config steps in AGENTS.md.)

## Stage 4 — Launch training

scp a launcher script and run it under nohup — do NOT use an inline ssh heredoc (the
container bashrc may have `set -e`, which aborts the heredoc when a `pkill` matches
nothing):
```bash
scp -q /tmp/launch.sh vast-rtx4090:/tmp/launch.sh
ssh vast-rtx4090 'bash /tmp/launch.sh'   # launcher does: cd /root/3dgrut; nohup … & echo PID
```
Use the standard multilayer command (see AGENTS.md "训练配置约定"): `--config-name
apps/ncore_3dgut_mcmc_multilayer[_poseopt]`, `trainer.sky_backend=mlp`, from-scratch.

**Hydra override rule (critical):** `+key` only ADDS a new key (errors if it exists);
`++key` OVERRIDES whether or not it exists (universal, safe); bare `key=` overrides an
existing top-level key. **All `layers.overrides.<layer>.<field>` must use `++`.**

**Memory:** a 4090 has 24 GB VRAM + limited system RAM. If you hit a silent OOM (process
killed ~iter 5000 with no Traceback), it's the DataLoader workers — lower `num_workers`
(see AGENTS.md "num_workers 必须按系统内存调"). Keep `use_lidar_depth` as the run needs;
turning it off lightens the data pipeline if comparability allows.

## Stage 5 — Monitor (don't drown the log)

When tailing eval, do NOT grep `PSNR` — render.py prints per-frame `Frame N, PSNR: X`
for thousands of frames and rate-limit suppression kills the monitor. Grep only key
nodes: `RUN [0-9]:|⭐ Test Metrics|^=== |Traceback|FAILED|OOM|🎊 Training Statistics`.
Confirm both the `🎊 Training Statistics` and `⭐ Test Metrics` tables appear, then read
`<out_dir>/.../metrics.json` for the KPI fields.

## Stage 6 — DESTROY the instance (do not skip)

```bash
echo y | /Users/etendue/repo/ncore/.venv/bin/vastai destroy instance <ID> --api-key $VAST_API_KEY
```
Verify it's gone (`vastai show instances`). Billing stops only on destroy.

## After the run — sync results back
Per AGENTS.md discipline: write the real per-class PSNR / LPIPS / commit hash / it·s
into `v3_plan_revised.md` § 6 Done Log + `v2_architecture.md` § 7. An A800/vast out-task
is only ✅ with metrics.json numbers + commit hash, not just "exit 0 + ckpt written".
