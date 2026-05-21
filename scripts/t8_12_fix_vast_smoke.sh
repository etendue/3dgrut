#!/usr/bin/env bash
# T8.12-FIX vast.ai viser launch + ssh tunnel setup.
#
# Phase 3 (after Phase 1 create + Phase 2 install). Launches viser_gui_4d
# in 3 configurations sequentially for visual comparison:
#   A. baseline:           --initial_fov_deg 90 (pinhole, matches T8.12 旧版)
#   B. Phase A.2 control:  --initial_fov_deg 90 (already in baseline; this is the A.2 fix)
#   C. Phase A.5 fisheye:  --camera_type Fisheye --camera_fov_deg 120 (engine._raygen_fisheye)
#
# Operator opens browser at http://localhost:8080 between launches to take
# screenshots, kills viser server with Ctrl-C, runs next variant.
#
# Assumes:
#   - ssh alias 'vast-rtx4090' is configured (from t8_12_fix_vast_create.sh)
#   - /root/3dgrut on remote is current (rsync done)
#   - /root/ckpt_with_viz_4d_v2.pt on remote is current
#   - /root/3dgrut/.venv has install_env_uv.sh products
#   - viser==1.0.27 installed (this script will install if missing)
set -euo pipefail

HOST_ALIAS="${HOST_ALIAS:-vast-rtx4090}"
REMOTE_REPO="${REMOTE_REPO:-/root/3dgrut}"
REMOTE_CKPT="${REMOTE_CKPT:-/root/ckpt_with_viz_4d_v2.pt}"
PORT="${PORT:-8080}"
LOCAL_PORT="${LOCAL_PORT:-8080}"
ARTIFACTS="${ARTIFACTS:-$HOME/4dviz_artifacts/t8_12_fix}"

mkdir -p "$ARTIFACTS"

echo "==> [1/4] Ensure viser==1.0.27 on remote"
ssh -T "$HOST_ALIAS" "cd $REMOTE_REPO && source .venv/bin/activate && \
    python -c 'import viser; print(\"viser\", viser.__version__)' 2>/dev/null || \
    pip install viser==1.0.27" 2>&1 | tail -5

echo ""
echo "==> [2/4] Kill any stale viser servers on remote"
ssh -T "$HOST_ALIAS" "pkill -f viser_gui_4d.py 2>/dev/null || true; \
    fuser -k ${PORT}/tcp 2>/dev/null || true; sleep 1; echo done" | tail -3

echo ""
echo "==> [3/4] Setup ssh tunnel (kill stale local 8080 first)"
lsof -ti tcp:"$LOCAL_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
nohup ssh -N -T -o ExitOnForwardFailure=yes \
    -o ControlMaster=no -o ControlPath=none \
    -L "$LOCAL_PORT:localhost:$PORT" "$HOST_ALIAS" \
    > /tmp/ssh_tunnel.log 2>&1 &
TUNNEL_PID=$!
echo "    tunnel PID=$TUNNEL_PID local=:$LOCAL_PORT → remote=:$PORT"
sleep 3
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "ERR: tunnel died. Log:"
    cat /tmp/ssh_tunnel.log
    exit 1
fi

echo ""
echo "==> [4/4] LAUNCH GUIDE — open http://localhost:$LOCAL_PORT in browser"
echo ""
cat <<EOF
================== VARIANT A: Baseline (Pinhole, fov 90°) ==================
ssh -T $HOST_ALIAS "cd $REMOTE_REPO && source .venv/bin/activate && \\
    python threedgrut_playground/viser_gui_4d.py \\
        --gs_object $REMOTE_CKPT \\
        --initial_fov_deg 90 \\
        --port $PORT 2>&1 | tee /tmp/viser_A.log"

→ Open http://localhost:$LOCAL_PORT, save screenshot → $ARTIFACTS/A_pinhole_fov90.png
→ Ctrl-C to kill, then run VARIANT B.

================== VARIANT B: Pinhole fov 60° (try narrower) ==================
ssh -T $HOST_ALIAS "cd $REMOTE_REPO && source .venv/bin/activate && \\
    python threedgrut_playground/viser_gui_4d.py \\
        --gs_object $REMOTE_CKPT \\
        --initial_fov_deg 60 \\
        --port $PORT 2>&1 | tee /tmp/viser_B.log"

→ Screenshot → $ARTIFACTS/B_pinhole_fov60.png

================== VARIANT C: Fisheye fov 120° (engine raygen) ==================
ssh -T $HOST_ALIAS "cd $REMOTE_REPO && source .venv/bin/activate && \\
    python threedgrut_playground/viser_gui_4d.py \\
        --gs_object $REMOTE_CKPT \\
        --camera_type Fisheye \\
        --camera_fov_deg 120 \\
        --port $PORT 2>&1 | tee /tmp/viser_C.log"

→ Screenshot → $ARTIFACTS/C_fisheye_fov120.png
→ Compare with ~/4dviz_artifacts/render_first_v4camspace.png (pinhole + c2w fix)

================== TEARDOWN ==================
kill $TUNNEL_PID                            # ssh tunnel
ssh -T $HOST_ALIAS "pkill -f viser_gui_4d.py"
/Users/etendue/repo/ncore/.venv/bin/vastai destroy instance \$(cat /tmp/t8_12_fix_instance.txt) \\
    --api-key \$VAST_API_KEY
EOF
echo ""
echo "Tunnel PID stored for teardown: $TUNNEL_PID"
echo "$TUNNEL_PID" > /tmp/t8_12_fix_tunnel.pid
