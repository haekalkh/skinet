#!/bin/bash
# SKINET-M2F: from raw kidney-biopsy images to a morphometry report.
#
# No annotation is needed. For each image the cortex is predicted with the
# region model, the ten structures are segmented inside it with the structure
# model, the predictions are post-processed, and stage 10 writes counts, areas
# and densities. An overlay is rendered for visual review.
#
# Usage:
#   ./run_inference.sh [options] image1.tif [image2.tif ...]
#
# Options:
#   --case NAME        treat all images as one biopsy, so densities are computed
#                      over the whole case instead of per image
#   --out DIR          where the report and overlays go (default: reports/)
#   --work DIR         intermediate files (default: work/)
#   --no-score-resolve keep the inherited priority rule for the tubular pair
#                      (this is what the thesis numbers were computed with)
#   --fp16             mixed-precision inference (faster, same predictions)
#   --scale S          overlay scale (default 0.3)
#
# Needs: a CUDA GPU (CPU works but is very slow), the environment from
# docs/INSTALL.md, and the two models from ./download_models.sh.

set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CASE=""; OUT="$HERE/reports"; WORK="${M2F_WORK_DIR:-$HERE/work}"
SCORE_RESOLVE=1; FP16=""; SCALE=0.3; RUN=released; TAG=_prod
IMAGES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --case) CASE="$2"; shift 2;;
        --out) OUT="$2"; shift 2;;
        --work) WORK="$2"; shift 2;;
        --no-score-resolve) SCORE_RESOLVE=0; shift;;
        --fp16) FP16="--fp16"; shift;;
        --scale) SCALE="$2"; shift 2;;
        -h|--help) sed -n 2,26p "$0"; exit 0;;
        *) IMAGES+=("$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"); shift;;
    esac
done
[ ${#IMAGES[@]} -eq 0 ] && { echo "usage: $0 [options] image1 [image2 ...]"; exit 2; }

WEIGHTS="${SKINET_WEIGHTS:-$HERE/models/skinet_m2f_structures.pth}"
export SKINET_CORTEX_WEIGHTS="${SKINET_CORTEX_WEIGHTS:-$HERE/models/skinet_m2f_cortex.pth}"
THRESHOLDS="${SKINET_THRESHOLDS:-$HERE/configs/thresholds/patient_100k.json}"
for f in "$WEIGHTS" "$SKINET_CORTEX_WEIGHTS" "$THRESHOLDS"; do
    [ -f "$f" ] || { echo "missing: $f  (run ./download_models.sh)"; exit 3; }
done

export M2F_WORK_DIR="$WORK"
M2F_ROOT="${M2F_ROOT:-$HERE/third_party/Mask2Former}"
export PYTHONPATH="$HERE/skinet_m2f:$M2F_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c "import mask2former" 2>/dev/null || {
    echo "mask2former is not importable. Install it (docs/INSTALL.md) or set M2F_ROOT"
    echo "to the folder of the Mask2Former repository."; exit 4; }
cd "$HERE/skinet_m2f"

echo "=== stage 05: cortex, then structures within it"
python -u s05_infer.py --run "$RUN" --weights "$WEIGHTS" --images "${IMAGES[@]}" \
    --predict-cortex --out-dir-tag "$TAG" $FP16

echo "=== stage 06: reassembly"
python -u s06_stitch.py --run "$RUN" --split test \
    --in-dir "$WORK/s05_infer/$RUN/test$TAG/preds" \
    --out-dir "$WORK/s06_stitch/$RUN/test$TAG"

echo "=== stage 07: post-processing"
S7=""
[ "$SCORE_RESOLVE" = "1" ] && S7="--score-resolve tubule_sain tubule_atrophique"
python -u s07_postproc.py --run "$RUN" --split test --tag "$TAG" \
    --in-dir "$WORK/s06_stitch/$RUN/test$TAG/stitched" \
    --thresholds "$THRESHOLDS" $S7

echo "=== stage 10: morphometry"
CASE_ARG=()
if [ -n "$CASE" ]; then
    CASE_ARG=(--case-of)
    for s in "${IMAGES[@]}"; do b=$(basename "$s"); CASE_ARG+=("${b%.*}=$CASE"); done
fi
python -u s10_morphometry.py --run "$RUN" --split test --tag "$TAG" "${CASE_ARG[@]}"

echo "=== overlays"
mkdir -p "$OUT"
for s in "${IMAGES[@]}"; do
    python -u render_from_stage07.py --run "$RUN" --split test --tag "$TAG" \
        --image "$s" --scale "$SCALE" --legible --out-dir "$OUT"
done
cp "$WORK/s10_morphometry/$RUN/test$TAG/morphometry.csv" "$OUT/" 2>/dev/null || true

echo ""
echo "report  : $OUT/morphometry.csv"
echo "overlays: $OUT"
echo "The report gives counts, areas and densities. It is not a graded IFTA"
echo "score: interstitial fibrosis is not segmented, and the atrophic share"
echo "covers tubular area only."
