#!/bin/bash
#$ -N m2f_profile
#$ -cwd
#$ -o /work/lead/mk657917/M2F-ANATOMY/logs/m2f_profile_$JOB_ID.log
#$ -e /work/lead/mk657917/M2F-ANATOMY/logs/m2f_profile_$JOB_ID.err
#
# Where the cost actually lives. Needs a GPU: on CPU the timings mean nothing.
#
#   the three baseline profiles in one job:
#     qsub -q gpu job_profile.sh
#
#   the compute-bound vs bandwidth-bound analysis, both precisions:
#     qsub -q gpu -v RUN=baseline_seed42,FLOPS=1,PROBE=1 job_profile.sh
#     qsub -q gpu -v RUN=baseline_seed42,FLOPS=1,PROBE=1,FP16=1 job_profile.sh
#
#   one configuration on its own:
#     qsub -q gpu -v RUN=q50_seed42,NUM_QUERIES=50 job_profile.sh
#     qsub -q gpu -v RUN=mrcnn_seed42,ARCH=maskrcnn job_profile.sh
#
# Note the GPU it lands on: timings are not comparable across V100 and A100, so
# every configuration in one table should come from the same card.

set -eo pipefail

RUN=${RUN:-}
ARCH=${ARCH:-mask2former}
NUM_QUERIES=${NUM_QUERIES:-}
TILE_SIZE=${TILE_SIZE:-}
FLOPS=${FLOPS:-0}
PROBE=${PROBE:-0}
FP16=${FP16:-0}

echo "===== START profile ====="; date; hostname
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

EXTRA=""
[ -n "$NUM_QUERIES" ] && EXTRA="$EXTRA --num-queries $NUM_QUERIES"
[ -n "$TILE_SIZE" ]   && EXTRA="$EXTRA --tile-size $TILE_SIZE"
[ "$FLOPS" = "1" ]    && EXTRA="$EXTRA --flops"
[ "$PROBE" = "1" ]    && EXTRA="$EXTRA --probe-dtype"
[ "$FP16" = "1" ]     && EXTRA="$EXTRA --fp16"

if [ -n "$RUN" ]; then
    python -u profile_cost.py --run "$RUN" --arch "$ARCH" $EXTRA
else
    # default sweep: the two levers that need no retraining, plus the
    # Mask R-CNN arm for scale
    python -u profile_cost.py --run baseline_seed42
    echo ""
    python -u profile_cost.py --run baseline_seed42 --fp16
    echo ""
    python -u profile_cost.py --run mrcnn_seed42 --arch maskrcnn
fi

echo ""
echo "===== END profile ====="; date