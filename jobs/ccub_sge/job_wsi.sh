#!/bin/bash
#$ -N m2f_wsi
#$ -cwd
#$ -o /work/lead/mk657917/M2F-ANATOMY/logs/m2f_wsi_$JOB_ID.log
#$ -e /work/lead/mk657917/M2F-ANATOMY/logs/m2f_wsi_$JOB_ID.err
#
# Paint a model's predictions across a whole slide. Needs a GPU.
#
#   the three baseline profiles in one job:
#     qsub -q gpu job_profile.sh
#
#   a small region first, to check it reads and paints correctly:
#     qsub -q gpu -v IMAGE=/work/.../HI15011717B1.tif,CROP="10000 6000 6000 6000" job_wsi.sh
#
#   one family per figure reads much better than ten classes at once:
#     qsub -q gpu -v IMAGE=...,ONLY="nsg nsg_complet nsg_partiel pac" job_wsi.sh
#     qsub -q gpu -v IMAGE=...,ONLY="vaisseau veine intima media" job_wsi.sh
#
#   with cortex gating, the way the production system restricts the main
#   model. The cortex stage runs at its own magnification first:
#     qsub -q gpu -v IMAGE=...,CORTEX=1 job_wsi.sh
#
#   sharper, for a figure a reader will zoom into:
#     qsub -q gpu -v IMAGE=...,CORTEX=1,OUT_SCALE=0.25,LINE_WIDTH=2 job_wsi.sh
#
#   then the whole slide:
#     qsub -q gpu -v IMAGE=/work/.../HI15011717B1.tif job_wsi.sh
#
# Note the GPU it lands on: timings are not comparable across V100 and A100, so
# every configuration in one table should come from the same card.

set -eo pipefail

RUN=${RUN:-patient_100k}
IMAGE=${IMAGE:-}
ARCH=${ARCH:-mask2former}
CROP=${CROP:-}
OUT_SCALE=${OUT_SCALE:-0.125}
CLASS_SET=${CLASS_SET:-main}
ONLY=${ONLY:-}
CORTEX=${CORTEX:-0}
DRAW_CORTEX=${DRAW_CORTEX:-1}
CORTEX_MIN=${CORTEX_MIN:-}
LINE_WIDTH=${LINE_WIDTH:-}
ALPHA=${ALPHA:-}
SCORE=${SCORE:-0.50}
FP16=${FP16:-1}

echo "===== START wsi $RUN ====="; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv

source /etc/profile
module load gcc/11.2.0
module load cuda/11.8
source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh
conda activate skinet

ANATOMY=/work/lead/mk657917/M2F-ANATOMY
SKINET_OLD=/work/lead/mk657917/skinet
PIPELINE=${PIPELINE:-$ANATOMY/pipeline}
M2F_ROOT=${M2F_ROOT:-$SKINET_OLD/models/Mask2Former}

[ -n "${WORK_DIR:-}" ] && export M2F_WORK_DIR="$WORK_DIR"

export PYTHONPATH=$PIPELINE:$M2F_ROOT
export LD_LIBRARY_PATH=/work/lead/mk657917/conda/envs/skinet/lib/python3.10/site-packages/torch/lib:/soft/c7/cuda/11.8/lib64:/soft/c7/gcc/11.2.0/lib64:$LD_LIBRARY_PATH

cd "$PIPELINE"

if [ -z "$IMAGE" ]; then
    echo "ERROR: IMAGE is required"
    exit 2
fi

EXTRA=""
[ -n "$CROP" ]  && EXTRA="$EXTRA --crop $CROP"
[ "$FP16" = "1" ] && EXTRA="$EXTRA --fp16"
[ -n "$ONLY" ]    && EXTRA="$EXTRA --only $ONLY"
[ "$CORTEX" = "1" ]      && EXTRA="$EXTRA --cortex"
[ "$DRAW_CORTEX" = "1" ] && [ "$CORTEX" = "1" ] && EXTRA="$EXTRA --draw-cortex"
[ -n "$CORTEX_MIN" ]     && EXTRA="$EXTRA --cortex-min $CORTEX_MIN"
[ -n "$LINE_WIDTH" ]     && EXTRA="$EXTRA --line-width $LINE_WIDTH"
[ -n "$ALPHA" ]          && EXTRA="$EXTRA --alpha $ALPHA"

python -u render_wsi_overlay.py \
    --run "$RUN" --arch "$ARCH" --image "$IMAGE" \
    --class-set "$CLASS_SET" --out-scale "$OUT_SCALE" --score "$SCORE" $EXTRA

echo ""
echo "===== END wsi ====="; date