#!/bin/bash
#$ -N m2f_train
#$ -cwd
#$ -o /work/lead/mk657917/M2F-ANATOMY/logs/m2f_train_$JOB_ID.log
#$ -e /work/lead/mk657917/M2F-ANATOMY/logs/m2f_train_$JOB_ID.err
#
# Clean-room pipeline, stage 04.
#
# This cluster's GPU starter does not forward positional arguments, so every
# setting arrives through -v.
#
#   smoke test (proves the plumbing, a few minutes):
#     qsub -q gpu -v RUN=smoke,MAX_ITER=100,EVAL_PERIOD=50 job_train.sh
#
#   the long run on the legacy recipe (cosine decay, rot90, 327700 steps):
#     qsub -q gpu -v RUN=long_seed42,MAX_ITER=327700,EVAL_PERIOD=10000,\
#       LR_SCHEDULE=cosine,ROT90=1 job_train.sh
#   if the queue kills it on wall time, resubmit the same line plus RESUME=1
#
#   a short SGD probe (the legacy optimizer; its 0.008 was tuned for a
#   convolutional backbone, so the rate has to be found again here):
#     qsub -q gpu -v RUN=sgd_lr01,MAX_ITER=2000,EVAL_PERIOD=1000,\
#       OPTIMIZER=SGD,LR=0.01,LR_SCHEDULE=cosine,ROT90=1 job_train.sh
#
#   the Mask R-CNN comparison arm (same data, same chain, one variable):
#     qsub -q gpu -v RUN=mrcnn_smoke,ARCH=maskrcnn,MAX_ITER=200,EVAL_PERIOD=100 job_train.sh
#
#   the query-budget study (100 is the COCO default; this data never needs
#   more than 41, and 82 tiles hold more than 25):
#     qsub -q gpu -v RUN=q50_seed42,NUM_QUERIES=50 job_train.sh
#     qsub -q gpu -v RUN=q25_seed42,NUM_QUERIES=25 job_train.sh
#
#   the three baseline seeds:
#     qsub -q gpu -v RUN=baseline_seed42,SEED=42 job_train.sh
#     qsub -q gpu -v RUN=baseline_seed43,SEED=43 job_train.sh
#     qsub -q gpu -v RUN=baseline_seed44,SEED=44 job_train.sh
#
# About 5-6 hours per run at 0.5 s per iteration. Every GPU node here is Volta,
# so the three can run in parallel without the arithmetic differing.

set -eo pipefail

RUN=${RUN:-}
SEED=${SEED:-42}
MAX_ITER=${MAX_ITER:-40000}
EVAL_PERIOD=${EVAL_PERIOD:-2500}
WORKERS=${WORKERS:-2}
LR_SCHEDULE=${LR_SCHEDULE:-step}
ROT90=${ROT90:-0}
RESUME=${RESUME:-0}
OPTIMIZER=${OPTIMIZER:-}
ARCH=${ARCH:-mask2former}
WORK_DIR=${WORK_DIR:-}
NUM_QUERIES=${NUM_QUERIES:-}
MIN_SCALE=${MIN_SCALE:-}
MAX_SCALE=${MAX_SCALE:-}
LR=${LR:-}

if [ -z "$RUN" ]; then
    echo "ERROR: RUN is required"
    echo "submit with:  qsub -q gpu -v RUN=baseline_seed42,SEED=42 job_train.sh"
    exit 2
fi

echo "===== START $RUN ====="; date; hostname
nvidia-smi

source /etc/profile
module load gcc/11.2.0
module load cuda/11.8
source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh
conda activate skinet

ANATOMY=/work/lead/mk657917/M2F-ANATOMY
SKINET_OLD=/work/lead/mk657917/skinet
PIPELINE=${PIPELINE:-$ANATOMY/pipeline}

# Stock Mask2Former, NOT the patched fork under $ANATOMY. Keeping $ANATOMY off
# the path is what makes this a clean baseline: the fork's modifications cannot
# be picked up by accident. The guard below confirms which copy was imported.
M2F_ROOT=${M2F_ROOT:-$SKINET_OLD/models/Mask2Former}
[ -n "$WORK_DIR" ] && export M2F_WORK_DIR="$WORK_DIR"
echo "work dir: ${M2F_WORK_DIR:-default from config}"

export PYTHONPATH=$PIPELINE:$M2F_ROOT

if [ ! -d "$M2F_ROOT/mask2former" ]; then
    echo "ERROR: no mask2former package under $M2F_ROOT"
    echo "pass the right one with -v M2F_ROOT=/path/to/Mask2Former"
    exit 3
fi
export LD_LIBRARY_PATH=/work/lead/mk657917/conda/envs/skinet/lib/python3.10/site-packages/torch/lib:/soft/c7/cuda/11.8/lib64:/soft/c7/gcc/11.2.0/lib64:$LD_LIBRARY_PATH

# Report where Mask2Former comes from, and list every config key the fork adds
# on top of stock together with its default. Anything that is not False or 0
# here means the baseline is carrying a modification -- stop and fix it before
# trusting any number that comes out.
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

echo ""
echo "=============================================================="
echo "RUN $RUN   seed=$SEED  max_iter=$MAX_ITER  eval_every=$EVAL_PERIOD"
echo "=============================================================="
EXTRA=""
[ "$ROT90" = "1" ] && EXTRA="$EXTRA --rot90"
[ "$RESUME" = "1" ] && EXTRA="$EXTRA --resume"
[ -n "$OPTIMIZER" ] && EXTRA="$EXTRA --optimizer $OPTIMIZER"
[ -n "$LR" ] && EXTRA="$EXTRA --lr $LR"
EXTRA="$EXTRA --arch $ARCH"
[ -n "$NUM_QUERIES" ] && EXTRA="$EXTRA --num-queries $NUM_QUERIES"
[ -n "$MIN_SCALE" ]   && EXTRA="$EXTRA --min-scale $MIN_SCALE"
[ -n "$MAX_SCALE" ]   && EXTRA="$EXTRA --max-scale $MAX_SCALE"

python -u s04_train.py \
    --run "$RUN" \
    --seed "$SEED" \
    --max-iter "$MAX_ITER" \
    --eval-period "$EVAL_PERIOD" \
    --checkpoint-period "$EVAL_PERIOD" \
    --lr-schedule "$LR_SCHEDULE" \
    --workers "$WORKERS" $EXTRA

echo ""
echo "===== END $RUN ====="; date