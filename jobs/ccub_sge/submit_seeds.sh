#!/bin/bash
# Submit three seeds of one variant.
#
# Run the smoke test first and let it finish:
#   qsub -q gpu -v RUN=smoke,MAX_ITER=100,EVAL_PERIOD=50 job_train.sh
#
# Then:
#   ./submit_seeds.sh              # baseline
#   ./submit_seeds.sh onetomany    # a later variant

set -eo pipefail
VARIANT="${1:-baseline}"

for SEED in 42 43 44; do
    qsub -q gpu -v RUN=${VARIANT}_seed${SEED},SEED=${SEED} job_train.sh
done

echo
echo "submitted. watch with:  qstat -u \$USER"
echo "logs in /work/lead/mk657917/M2F-ANATOMY/logs/"