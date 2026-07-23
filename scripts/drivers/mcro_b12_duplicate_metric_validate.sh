#!/usr/bin/env bash
# Validate the B12 duplicate metric against the frozen R6 and Z-filter arms.

set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 R6_CKPT DROP_ZLT_CKPT DROP_ZGT_CKPT OUT_ROOT" >&2
  exit 2
fi

R6_CKPT=$1
DROP_ZLT_CKPT=$2
DROP_ZGT_CKPT=$3
OUT_ROOT=$4
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

mkdir -p "$OUT_ROOT"
for spec in \
  "r6:$R6_CKPT" \
  "drop_zlt0:$DROP_ZLT_CKPT" \
  "drop_zgt0:$DROP_ZGT_CKPT"
do
  name=${spec%%:*}
  checkpoint=${spec#*:}
  test -f "$checkpoint"
  bash "$REPO_DIR/scripts/drivers/mcro_depth_ownership_diag.sh" \
    "$checkpoint" "$OUT_ROOT/$name"
done

python - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for name in ("r6", "drop_zlt0", "drop_zgt0"):
    rows[name] = json.loads((root / name / "ownership.json").read_text())["summary"]

baseline = rows["r6"]["bg_road_duplicate_alpha_mean"]
for values in rows.values():
    values["duplicate_reduction_vs_r6"] = 1.0 - (
        values["bg_road_duplicate_alpha_mean"] / baseline
    )

report = {"arms": rows}
(root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY

echo "MCRO_B12_TASK1_DONE report=$OUT_ROOT/comparison.json"
