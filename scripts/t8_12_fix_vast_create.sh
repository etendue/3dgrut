#!/usr/bin/env bash
# T8.12-FIX vast.ai RTX 4090 instance creation + ssh setup.
#
# Phase 1 of the T8.12-FIX validation pipeline: just spin up the instance
# and wire ssh. Env install (Phase 2) and viser launch (Phase 3) live in
# separate scripts so each phase is restartable.
#
# References:
#   - docs/T8.12_handover_day1.md § 5.2 完整重建
#   - v2_plan.md Done Log § ⚠️ T8.12 / 🟡 T8.12-FIX
#
# Usage:
#   bash scripts/t8_12_fix_vast_create.sh
#
# Outputs:
#   - Creates a new vast.ai instance (cheapest RTX 4090, ~$0.5-0.8/hr)
#   - Attaches ed25519 pubkey
#   - Writes ~/.ssh/config Host vast-rtx4090 alias (idempotent replace)
#   - Prints instance ID + ssh host:port + cost summary
#   - Waits until ssh handshake succeeds (max ~5 min)
#
# Cost guard:
#   - Caps at $0.80/hr (override with MAX_DPH=<float>)
#   - Prints estimate before creating
set -euo pipefail

VASTAI="${VASTAI:-/Users/etendue/repo/ncore/.venv/bin/vastai}"
API_KEY="${VAST_API_KEY:-a6d4a47d11507fec636572f4ba555a1cb2395864eac29f33035eb1bcd5712f0d}"
PUBKEY_PATH="${PUBKEY_PATH:-$HOME/.ssh/id_ed25519.pub}"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
HOST_ALIAS="${HOST_ALIAS:-vast-rtx4090}"
LABEL="${LABEL:-eason_3dgrut_t8_12_fix}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel}"
DISK_GB="${DISK_GB:-80}"
MAX_DPH="${MAX_DPH:-0.80}"

[ -x "$VASTAI" ] || { echo "ERR: vastai CLI not found at $VASTAI" >&2; exit 1; }
[ -f "$PUBKEY_PATH" ] || { echo "ERR: ssh pubkey not at $PUBKEY_PATH" >&2; exit 1; }

echo "==> [1/6] Search cheapest RTX 4090 offer (max \$$MAX_DPH/hr)"
OFFER_JSON=$("$VASTAI" search offers \
    "gpu_name=RTX_4090 num_gpus=1 cpu_cores>=16 cpu_ram>=64 disk_space>=$DISK_GB reliability>0.95 rentable=true" \
    --storage "$DISK_GB" --raw --api-key "$API_KEY")
OFFER_LINE=$(python3 -c "
import json, sys
data = json.loads('''$OFFER_JSON''')
data = [d for d in data if d.get('dph_total', 999) <= $MAX_DPH]
data.sort(key=lambda x: x.get('dph_total', 999))
if not data:
    sys.exit('no offers under cap')
d = data[0]
print(f\"{d['id']} {d['dph_total']:.3f} {d.get('geolocation','?')} {d.get('inet_down','?')}/{d.get('inet_up','?')} Mbps rel={d.get('reliability2','?'):.3f}\")
")
OFFER_ID=$(echo "$OFFER_LINE" | awk '{print $1}')
OFFER_DPH=$(echo "$OFFER_LINE" | awk '{print $2}')
echo "    pick: $OFFER_LINE"

echo "==> [2/6] Create instance (label=$LABEL, image=$IMAGE, disk=${DISK_GB}GB)"
CREATE_JSON=$("$VASTAI" create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --disk "$DISK_GB" --label "$LABEL" \
    --ssh --direct --api-key "$API_KEY" --raw)
INSTANCE_ID=$(python3 -c "import json; print(json.loads('''$CREATE_JSON''')['new_contract'])")
echo "    new instance: $INSTANCE_ID (~\$$OFFER_DPH/hr)"

echo "==> [3/6] Attach ed25519 pubkey to instance"
PUBKEY=$(cat "$PUBKEY_PATH")
"$VASTAI" attach ssh "$INSTANCE_ID" "$PUBKEY" --api-key "$API_KEY" >/dev/null
echo "    pubkey attached"

echo "==> [4/6] Wait for ssh_host/port + status=running (max 5 min)"
SSH_HOST=""; SSH_PORT=""; STATUS=""
for i in $(seq 1 60); do
    SHOW=$("$VASTAI" show instance "$INSTANCE_ID" --raw --api-key "$API_KEY" 2>/dev/null || true)
    SSH_HOST=$(python3 -c "import json,sys; d=json.loads('''$SHOW''') if '''$SHOW''' else {}; print(d.get('ssh_host','') or '')" 2>/dev/null || true)
    SSH_PORT=$(python3 -c "import json,sys; d=json.loads('''$SHOW''') if '''$SHOW''' else {}; p=d.get('ssh_port'); print(p if p else '')" 2>/dev/null || true)
    STATUS=$(python3 -c "import json,sys; d=json.loads('''$SHOW''') if '''$SHOW''' else {}; print(d.get('actual_status','') or '')" 2>/dev/null || true)
    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] && [ "$STATUS" = "running" ]; then
        echo "    ready: $SSH_HOST:$SSH_PORT (status=$STATUS)"
        break
    fi
    printf "    waiting [%02d/60] host=%s port=%s status=%s\r" "$i" "${SSH_HOST:-?}" "${SSH_PORT:-?}" "${STATUS:-?}"
    sleep 5
done
echo
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] || [ "$STATUS" != "running" ]; then
    echo "ERR: timed out waiting for instance to become ready"
    echo "    Manual check: $VASTAI show instance $INSTANCE_ID --api-key \$API_KEY"
    exit 1
fi

echo "==> [5/6] Update ~/.ssh/config alias '$HOST_ALIAS'"
TMP_CONFIG=$(mktemp)
# Strip any existing block for HOST_ALIAS, then append new one.
python3 - <<PY > "$TMP_CONFIG"
import re
path = "$SSH_CONFIG"
alias = "$HOST_ALIAS"
try:
    with open(path) as f:
        txt = f.read()
except FileNotFoundError:
    txt = ""
# Remove existing block starting with "Host <alias>" until next "Host " or EOF.
pat = re.compile(rf"(^|\n)Host\s+{re.escape(alias)}\b.*?(?=\nHost\s|\Z)", re.S)
txt = pat.sub("", txt)
txt = txt.rstrip() + "\n\n"
txt += f"""Host {alias}
    HostName $SSH_HOST
    Port $SSH_PORT
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
"""
print(txt, end="")
PY
mv "$TMP_CONFIG" "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"
echo "    wrote Host $HOST_ALIAS → $SSH_HOST:$SSH_PORT"

echo "==> [6/6] Verify ssh handshake (max 3 min, sshd may still be starting)"
SSH_OK=0
for i in $(seq 1 36); do
    if ssh -T -o ConnectTimeout=5 -o BatchMode=yes "$HOST_ALIAS" \
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null; then
        SSH_OK=1
        break
    fi
    printf "    handshake retry [%02d/36]\r" "$i"
    sleep 5
done
echo
if [ "$SSH_OK" -ne 1 ]; then
    echo "WARN: ssh handshake failed; instance up but sshd not responding."
    echo "    Try manually in 30s: ssh -T $HOST_ALIAS 'nvidia-smi'"
    echo "    Or check: $VASTAI show instance $INSTANCE_ID --api-key \$API_KEY"
    exit 2
fi

echo ""
echo "================== T8.12-FIX vast.ai instance READY =================="
echo "  Instance ID:  $INSTANCE_ID"
echo "  Cost:         \$$OFFER_DPH/hr"
echo "  ssh alias:    $HOST_ALIAS"
echo "  ssh_host:     $SSH_HOST:$SSH_PORT"
echo ""
echo "Next steps (separate scripts):"
echo "  bash scripts/t8_12_fix_vast_install.sh    # ~10-15 min env install"
echo "  bash scripts/t8_12_fix_vast_smoke.sh      # launch viser + tunnel"
echo ""
echo "Destroy when done:"
echo "  $VASTAI destroy instance $INSTANCE_ID --api-key \$API_KEY"
echo "======================================================================"
echo "$INSTANCE_ID" > /tmp/t8_12_fix_instance_id.txt
echo "(instance ID saved to /tmp/t8_12_fix_instance_id.txt for downstream scripts)"
