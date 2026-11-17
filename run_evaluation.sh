#!/bin/bash
# SKINET-M2F: evaluate a trained run against the annotations (stages 05-09).
#
# Runs inference on val and test, tunes per-class score thresholds on val only,
# applies them to test, and scores instance-level precision / recall / F1 and
# Dice per class. Reproduces the protocol behind the reported numbers.
#
# Usage:
#   ./run_evaluation.sh RUN_NAME [--weights path/to/model.pth]
#   ./run_evaluation.sh released --weights models/skinet_m2f_structures.pth

set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${1:?usage: $0 RUN_NAME [--weights file]}"; shift
W=()
[ "$1" = "--weights" ] && { W=(--weights "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"); shift 2; }
export M2F_WORK_DIR="${M2F_WORK_DIR:-$HERE/work}"
export PYTHONPATH="$HERE/skinet_m2f${M2F_ROOT:+:$M2F_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE/skinet_m2f"
WD="$M2F_WORK_DIR"; TAG=_base050

for SPLIT in val test; do
    python -u s05_infer.py --run "$RUN" --split $SPLIT "${W[@]}"
    python -u s06_stitch.py --run "$RUN" --split $SPLIT
done
python -u s07_postproc.py --run "$RUN" --split val --tag $TAG
python -u s09_tune.py --run "$RUN" --tag $TAG --low 0.50
THR="$WD/s09_tune/$RUN/tune$TAG/thresholds.json"
python -u s07_postproc.py --run "$RUN" --split test --tag $TAG --thresholds "$THR"
python -u s08_eval.py --run "$RUN" --split test --tag $TAG
echo "results: $WD/s08_eval/$RUN/test$TAG/"
