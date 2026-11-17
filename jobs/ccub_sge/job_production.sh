#!/bin/bash
#$ -N skinet_prod
#$ -cwd
#$ -o /work/lead/mk657917/M2F-ANATOMY/logs/skinet_prod_$JOB_ID.log
#$ -e /work/lead/mk657917/M2F-ANATOMY/logs/skinet_prod_$JOB_ID.err
#
# From a raw slide to a morphometry report.
#
# The evaluation path ends at stage 08, which scores predictions against
# annotations. A slide arriving from the scanner has none, so this path predicts
# the cortex with the region model, segments the structures inside it, and ends
# at the quantities a report contains: glomerular density per square millimetre
# of cortex and the atrophic share of tubular area.
#
# This cluster's GPU starter does not forward positional arguments, so every
# setting arrives through -v.
#
#   one slide:
#     qsub -q gpu -v SLIDE=/work/.../HI15011717B1.tif,CASE=case01 job_production.sh
#
#   several slides of one biopsy, so that densities are computed over the whole
#   case rather than over each slide separately -- quote the list:
#     qsub -q gpu -v SLIDE="/path/a.tif /path/b.tif",CASE=case01 job_production.sh
#
#   keeping the inherited priority rule instead of resolving by score:
#     qsub -q gpu -v SLIDE=...,SCORE_RESOLVE=0 job_production.sh
#
# Only stage 05 needs the GPU; the rest is CPU and runs in the same job because
# splitting it would mean shipping intermediate files between queues.

set -eo pipefail

SLIDE=${SLIDE:-}
CASE=${CASE:-}
RUN=${RUN:-patient_100k}
TAG=${TAG:-_prod}
WORK_DIR=${WORK_DIR:-/work/lead/mk657917/M2F-ANATOMY/work_patient}
THRESHOLDS=${THRESHOLDS:-}
SCORE_RESOLVE=${SCORE_RESOLVE:-1}
SCALE=${SCALE:-0.3}
OUT=${OUT:-}

if [ -z "$SLIDE" ]; then
    echo "ERROR: SLIDE is required"
    echo "submit with:  qsub -q gpu -v SLIDE=/path/to/slide.tif,CASE=case01 job_production.sh"
    exit 2
fi

echo "===== START production $RUN ====="; date; hostname
nvidia-smi

source /etc/profile
module load gcc/11.2.0
module load cuda/11.8
source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh
conda activate skinet

ANATOMY=/work/lead/mk657917/M2F-ANATOMY
SKINET_OLD=/work/lead/mk657917/skinet
PIPELINE=${PIPELINE:-$ANATOMY/merdeka}

# Stock Mask2Former, not the fp16 fork: the fork's narrowed precision guard is
# a measured change but has not been validated for a clinical report.
M2F_ROOT=${M2F_ROOT:-$SKINET_OLD/models/Mask2Former}
export M2F_WORK_DIR="$WORK_DIR"
echo "work dir: $M2F_WORK_DIR"

export PYTHONPATH=$PIPELINE:$M2F_ROOT

if [ ! -d "$M2F_ROOT/mask2former" ]; then
    echo "ERROR: no mask2former package under $M2F_ROOT"
    exit 3
fi
export LD_LIBRARY_PATH=/work/lead/mk657917/conda/envs/skinet/lib/python3.10/site-packages/torch/lib:/soft/c7/cuda/11.8/lib64:/soft/c7/gcc/11.2.0/lib64:$LD_LIBRARY_PATH

python -c "
import inspect, os, mask2former
p = os.path.realpath(inspect.getfile(mask2former))
print('mask2former from:', p)
if os.sep + 'Mask2Former_patched' + os.sep in p:
    print('STOP: this is the old patched fork')
    raise SystemExit(4)
"

cd "$PIPELINE"

[ -z "$THRESHOLDS" ] && THRESHOLDS="$WORK_DIR/s09_tune/$RUN/tune_base050/thresholds.json"
[ -z "$OUT" ] && OUT="$WORK_DIR/reports"
if [ ! -f "$THRESHOLDS" ]; then
    echo "ERROR: no thresholds at $THRESHOLDS"
    echo "pass another with -v THRESHOLDS=/path/to/thresholds.json"
    exit 5
fi

echo ""
echo "=============================================================="
echo "PRODUCTION $RUN   tag=$TAG"
echo "slides: $SLIDE"
echo "case  : ${CASE:-one per slide}"
echo "output: $OUT"
echo "=============================================================="

# ---------------------------------------------------------------- stage 05
# The cortex is predicted, not read: a fresh slide has no annotation. The region
# model works on the slide resized so its long side is 2048, which is the
# magnification it was trained at; a single tile then covers the whole biopsy.
echo ""
echo "--- stage 05: cortex, then structures within it"
python -u s05_infer.py --run "$RUN" --images $SLIDE \
    --predict-cortex --out-dir-tag "$TAG"

echo ""
echo "--- stage 06: reassembly"
python -u s06_stitch.py --run "$RUN" --split test \
    --in-dir "$WORK_DIR/s05_infer/$RUN/test$TAG/preds" \
    --out-dir "$WORK_DIR/s06_stitch/$RUN/test$TAG"

# ---------------------------------------------------------------- stage 07
# Score resolution replaces the inherited priority rule for the tubular pair.
# That rule discarded every atrophic prediction overlapping a non-atrophic one,
# whatever the model thought, and it is the single largest source of the
# systematic understatement of atrophy.
echo ""
echo "--- stage 07: post-processing"
S7EXTRA=""
[ "$SCORE_RESOLVE" = "1" ] && S7EXTRA="--score-resolve tubule_sain tubule_atrophique"
python -u s07_postproc.py --run "$RUN" --split test --tag "$TAG" \
    --in-dir "$WORK_DIR/s06_stitch/$RUN/test$TAG/stitched" \
    --thresholds "$THRESHOLDS" $S7EXTRA

echo ""
echo "--- stage 10: morphometry"
CASE_ARG=""
if [ -n "$CASE" ]; then
    for s in $SLIDE; do
        b=$(basename "$s"); b="${b%.*}"
        CASE_ARG="$CASE_ARG $b=$CASE"
    done
    CASE_ARG="--case-of$CASE_ARG"
fi
python -u s10_morphometry.py --run "$RUN" --split test --tag "$TAG" $CASE_ARG

echo ""
echo "--- overlay for review"
mkdir -p "$OUT"
for s in $SLIDE; do
    python -u render_from_stage07.py --run "$RUN" --split test --tag "$TAG" \
        --image "$s" --scale "$SCALE" --legible --out-dir "$OUT"
done
cp "$WORK_DIR/s10_morphometry/$RUN/test$TAG/morphometry.csv" "$OUT/" 2>/dev/null || true

echo ""
echo "===== END production $RUN ====="; date
echo "report  : $OUT/morphometry.csv"
echo "overlays: $OUT"
echo ""
echo "The report gives counts, areas and densities. It does not give a graded"
echo "IFTA score: interstitial fibrosis is not segmented by this pipeline, and"
echo "the atrophic percentage covers tubular area only."