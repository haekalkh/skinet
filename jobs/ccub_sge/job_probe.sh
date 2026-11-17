#!/bin/bash
#$ -N m2f_probe
#$ -cwd
#$ -o /work/lead/mk657917/M2F-ANATOMY/logs/m2f_probe_$JOB_ID.log
#$ -e /work/lead/mk657917/M2F-ANATOMY/logs/m2f_probe_$JOB_ID.err
#
# What dtype the deformable-attention kernel actually receives. Needs a GPU:
# the CUDA kernel is the whole point.
#
# By default it runs both precisions, since fp32 is the reference the fp16
# line has to be read against.
#
#   qsub -q gpu job_probe.sh
#   qsub -q gpu -v RUN=long_seed42 job_probe.sh

set -eo pipefail

RUN=${RUN:-baseline_seed42}

echo "===== START probe $RUN ====="; date; hostname
nvidia-smi --query-gpu=name --format=csv,noheader

source /etc/profile
module load gcc/11.2.0
module load cuda/11.8
source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh
conda activate skinet

ANATOMY=/work/lead/mk657917/M2F-ANATOMY
SKINET_OLD=/work/lead/mk657917/skinet
PIPELINE=${PIPELINE:-$ANATOMY/pipeline}
M2F_ROOT=${M2F_ROOT:-$SKINET_OLD/models/Mask2Former}

export PYTHONPATH=$PIPELINE:$M2F_ROOT
export LD_LIBRARY_PATH=/work/lead/mk657917/conda/envs/skinet/lib/python3.10/site-packages/torch/lib:/soft/c7/cuda/11.8/lib64:/soft/c7/gcc/11.2.0/lib64:$LD_LIBRARY_PATH

cd "$PIPELINE"

echo ""
python -u probe_msdeform.py --run "$RUN"
echo ""
python -u probe_msdeform.py --run "$RUN" --fp16

echo ""
echo "===== END probe $RUN ====="; date