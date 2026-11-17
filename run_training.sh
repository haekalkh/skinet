#!/bin/bash
# SKINET-M2F: train a structure model on your own annotated data (stages 01-04).
#
# Data layout: images and their ASAP XML annotations in one folder (or two,
# see SKINET_XML_DIR), one XML per image with the same file stem. Class names
# and the cortex polygons are read from the XML; edit STRUCTURE_CLASSES and
# LABEL_ALIASES in skinet_m2f/config.py for a new label set.
#
# Usage:
#   SKINET_RAW_DIR=/path/to/data ./run_training.sh [RUN_NAME] [extra s04 args]
#
# Examples:
#   ./run_training.sh my_run
#   ./run_training.sh my_run_100k --max-iter 100000 --eval-period 5000 \
#       --checkpoint-period 5000 --lr-schedule cosine --rot90        # final recipe
#   ./run_training.sh my_finetune --weights models/skinet_m2f_structures.pth  # fine-tune
#
# Training from scratch needs the COCO-pretrained Mask2Former Swin-B weights in
# models/pretrained/mask2former_swin_base.pkl (see models/README.md).

set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${1:-my_run}"; shift || true
export M2F_WORK_DIR="${M2F_WORK_DIR:-$HERE/work}"
export PYTHONPATH="$HERE/skinet_m2f${M2F_ROOT:+:$M2F_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE/skinet_m2f"

echo "=== stage 01: audit raw data in ${SKINET_RAW_DIR:-$HERE/data/raw}"
python -u s01_audit.py
echo "=== stage 02: slide split, then regroup by patient"
python -u s02_split.py
python -u s02b_split_patient.py --dry-run
echo "Check the patient grouping printed above (identifier convention is local)."
python -u s02b_split_patient.py
echo "=== stage 03: tiles + COCO annotations"
python -u s03_tiles.py
echo "=== stage 04: training run $RUN"
python -u s04_train.py --run "$RUN" "$@"
echo "model: $M2F_WORK_DIR/s04_train/$RUN/model_best.pth"
echo "next : ./run_evaluation.sh $RUN"
