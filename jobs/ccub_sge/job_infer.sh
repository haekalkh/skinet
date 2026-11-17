#!/bin/bash
#$ -N m2f_infer
#$ -cwd
#$ -o /work/lead/mk657917/M2F-ANATOMY/logs/m2f_infer_$JOB_ID.log
#$ -e /work/lead/mk657917/M2F-ANATOMY/logs/m2f_infer_$JOB_ID.err
#
# Stages 05 and 06: run the model over whole ROIs, then map the tile
# predictions into whole-ROI coordinates. Stage 06 needs no GPU but costs
# minutes, so it rides along rather than being submitted separately.
#
# This cluster's GPU starter does not forward positional arguments, so
# everything arrives through -v.
#
#   two slides first, to see it work end to end:
#     qsub -q gpu -v RUN=baseline_seed42,SPLIT=test,LIMIT=2 job_infer.sh
#
#   the full split:
#     qsub -q gpu -v RUN=baseline_seed42,SPLIT=test job_infer.sh
#
#   half precision, into its own folder so the fp32 run survives:
#     qsub -q gpu -v RUN=baseline_seed42,SPLIT=test,FP16=1,TAG=_fp16 job_infer.sh
#
#   cortex masking, the way the legacy system restricts MAIN:
#     qsub -q gpu -v RUN=baseline_seed42,SPLIT=test,MASK_CORTEX=1,TAG=_mask job_infer.sh
#
#   the Mask R-CNN arm (--arch must match how the run was trained):
#     qsub -q gpu -v RUN=mrcnn_seed42,ARCH=maskrcnn,SPLIT=test job_infer.sh
#
#   a different checkpoint:
#     qsub -q gpu -v RUN=long_seed42,SPLIT=test,WEIGHTS=/path/to/model_best.pth job_infer.sh
#
#   skip the stitch (stage 06 only, no GPU needed — run it directly instead):
#     qsub -q gpu -v RUN=baseline_seed42,SPLIT=test,STITCH=0 job_infer.sh
#
#   a whole slide, which carries no annotation, so the cortex is predicted by
#   the region model instead of read from XML:
#     qsub -q gpu -v RUN=patient_100k,IMAGES=/work/.../HI15011717B1.tif,TAG=_wsi job_infer.sh
#
#   several slides at once (quote the list, it is split on spaces):
#     qsub -q gpu -v RUN=patient_100k,IMAGES="/path/a.tif /path/b.tif",TAG=_wsi job_infer.sh

set -eo pipefail

RUN=${RUN:-}
SPLIT=${SPLIT:-test}
LIMIT=${LIMIT:-0}
WEIGHTS=${WEIGHTS:-}
SCORE_FLOOR=${SCORE_FLOOR:-}
MIN_TILE_CORTEX=${MIN_TILE_CORTEX:-}
STITCH=${STITCH:-1}
ARCH=${ARCH:-mask2former}
FP16=${FP16:-0}
MASK_CORTEX=${MASK_CORTEX:-0}
IMAGES=${IMAGES:-}
PREDICT_CORTEX=${PREDICT_CORTEX:-0}
TAG=${TAG:-}

if [ -z "$RUN" ]; then
    echo "ERROR: RUN is required (the training run whose weights to load)"
    echo "submit with:  qsub -q gpu -v RUN=baseline_seed42,SPLIT=test job_infer.sh"
    exit 2
fi

echo "===== START infer $RUN / ${IMAGES:-$SPLIT} ====="; date; hostname
nvidia-smi

source /etc/profile
module load gcc/11.2.0
module load cuda/11.8
source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh
conda activate skinet

ANATOMY=/work/lead/mk657917/M2F-ANATOMY
SKINET_OLD=/work/lead/mk657917/skinet
PIPELINE=${PIPELINE:-$ANATOMY/pipeline}

# Stock Mask2Former, not the patched fork under $ANATOMY.
M2F_ROOT=${M2F_ROOT:-$SKINET_OLD/models/Mask2Former}
[ -n "${WORK_DIR:-}" ] && export M2F_WORK_DIR="$WORK_DIR"

export PYTHONPATH=$PIPELINE:$M2F_ROOT
export LD_LIBRARY_PATH=/work/lead/mk657917/conda/envs/skinet/lib/python3.10/site-packages/torch/lib:/soft/c7/cuda/11.8/lib64:/soft/c7/gcc/11.2.0/lib64:$LD_LIBRARY_PATH

if [ ! -d "$M2F_ROOT/mask2former" ]; then
    echo "ERROR: no mask2former package under $M2F_ROOT"
    exit 3
fi

python -c "
import inspect, os, mask2former
p = os.path.realpath(inspect.getfile(mask2former))
print('mask2former from:', p)
# The original guard rejected anything under \$ANATOMY, to stop the heavily
# patched old fork being used by accident. Deliberate forks now live there too,
# so only that one is refused and every other path is reported and allowed.
if os.sep + 'Mask2Former_patched' + os.sep in p:
    print('STOP: this is the old patched fork')
    raise SystemExit(4)
"

cd "$PIPELINE"

EXTRA=""
[ "$LIMIT" != "0" ]        && EXTRA="$EXTRA --limit $LIMIT"
[ -n "$WEIGHTS" ]          && EXTRA="$EXTRA --weights $WEIGHTS"
[ -n "$SCORE_FLOOR" ]      && EXTRA="$EXTRA --score-floor $SCORE_FLOOR"
[ -n "$MIN_TILE_CORTEX" ]  && EXTRA="$EXTRA --min-tile-cortex $MIN_TILE_CORTEX"
EXTRA="$EXTRA --arch $ARCH"
[ "$FP16" = "1" ] && EXTRA="$EXTRA --fp16"
[ "$MASK_CORTEX" = "1" ] && EXTRA="$EXTRA --mask-cortex"
[ -n "$TAG" ]    && EXTRA="$EXTRA --out-dir-tag $TAG"
# unquoted on purpose: several paths arrive as one space-separated string
[ -n "$IMAGES" ] && EXTRA="$EXTRA --images $IMAGES"
[ "$PREDICT_CORTEX" = "1" ] && EXTRA="$EXTRA --predict-cortex"

if [ -n "$IMAGES" ] && [ -z "$TAG" ]; then
    echo "ERROR: give a TAG when running on explicit images, so the output"
    echo "does not overwrite the split it shares a folder name with."
    exit 2
fi

echo ""
echo "=============================================================="
echo "stage 05 - inference | run $RUN  split $SPLIT"
echo "=============================================================="
python -u s05_infer.py --run "$RUN" --split "$SPLIT" $EXTRA

if [ "$STITCH" = "1" ]; then
    echo ""
    echo "=============================================================="
    echo "stage 06 - stitch"
    echo "=============================================================="
    if [ -n "$TAG" ]; then
        # only override the paths when a tag is in play; otherwise let the
        # script derive them from config, which follows M2F_WORK_DIR
        BASE=${M2F_WORK_DIR:-/work/lead/mk657917/M2F-ANATOMY/work}
        python -u s06_stitch.py --run "$RUN" --split "$SPLIT" \
            --in-dir "$BASE/s05_infer/$RUN/${SPLIT}${TAG}/preds" \
            --out-dir "$BASE/s06_stitch/$RUN/${SPLIT}${TAG}"
    else
        python -u s06_stitch.py --run "$RUN" --split "$SPLIT"
    fi
fi

echo ""
echo "===== END infer $RUN / $SPLIT ====="; date