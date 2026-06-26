#!/bin/bash
# =============================================================================
#  submit_e7_run.sh -- SGE job run by the `skinet` CLI (do not call by hand).
#  The CLI passes -v SKINET_ENV=<effective config> and forwards pipeline args.
#  This script sources that config, sets up the env, and runs e7_run.py.
# =============================================================================
#$ -N skinet
#$ -j y

set -uo pipefail

# ---- config resolved by the CLI --------------------------------------------
if [ -n "${SKINET_ENV:-}" ] && [ -f "${SKINET_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${SKINET_ENV}"
else
  echo "ERROR: SKINET_ENV not set or missing; submit via the 'skinet' CLI"; exit 1
fi

SDIR="${SKINET_SCRIPTS_DIR:?SKINET_SCRIPTS_DIR not set}"
CONDA_SH="${SKINET_CONDA_SH:-/work/lead/mk657917/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${SKINET_CONDA_ENV:-skinet}"
M2F_LIB="${SKINET_M2F_LIB:-}"
SAM_LIB="${SKINET_SAM_LIB:-}"
CUDA_MODULES="${SKINET_MODULES:-gcc/11.2.0 cuda/11.8}"

echo "=============================================================="
echo " SKINET e7 pipeline"
echo " host    : $(hostname)"
echo " job     : ${JOB_ID:-<interactive>}"
echo " time    : $(date)"
echo " scripts : $SDIR"
echo " args    : $*"
echo "=============================================================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  || echo "(nvidia-smi unavailable)"

# ---- environment ------------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH" 2>/dev/null \
  || source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /work/lead/mk657917/conda/etc/profile.d/conda.sh 2>/dev/null \
  || { echo "ERROR: cannot find conda.sh (set SKINET_CONDA_SH)"; exit 1; }
conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate conda env '$CONDA_ENV'"; exit 1; }
# shellcheck disable=SC2086
module load $CUDA_MODULES 2>/dev/null || echo "WARN: GPU modules not loaded ($CUDA_MODULES)"

# PYTHONPATH: Mask2Former + SAM-Med2D (both code deps), skipping empties
PP="${PYTHONPATH:-}"
[ -n "$M2F_LIB" ] && PP="$M2F_LIB:$PP"
[ -n "$SAM_LIB" ] && PP="$SAM_LIB:$PP"
export PYTHONPATH="$PP"

# ---- preflight on the node --------------------------------------------------
[ -f "$SDIR/e7_run.py" ]   || { echo "ERROR: e7_run.py not found in $SDIR"; exit 1; }
[ -f "$SDIR/e7_config.py" ]|| { echo "ERROR: e7_config.py not found in $SDIR"; exit 1; }
[ -f "$SDIR/tissue_filter.py" ] || echo "WARN: tissue_filter.py missing in $SDIR"

# ---- run --------------------------------------------------------------------
if [ "${SKINET_EVAL_ONLY:-0}" = "1" ]; then
  echo "--- eval-only: skipping e7_run, computing metrics on existing output ---"
  status=0
else
  echo "--- e7_run.py $* ---"
  python "$SDIR/e7_run.py" --scripts-dir "$SDIR" "$@"
  status=$?
fi

# ---- metrics (only when we ran a labelled split: test/all, not a bare image) -
# The CLI exports SKINET_EVAL=1 and SKINET_EVAL_SPLIT / SKINET_EVAL_OUTDIR when
# evaluation is possible (ground truth exists). e7_run output schema matches
# Stage-14's expected input, so we point --in-dir straight at it.
if [ "$status" -eq 0 ] && [ "${SKINET_EVAL:-0}" = "1" ]; then
  EVAL_SPLIT="${SKINET_EVAL_SPLIT:-test}"
  PRED_DIR="${SKINET_EVAL_OUTDIR:?}"
  EVAL_OUT="${PRED_DIR}/metrics"
  mkdir -p "$EVAL_OUT"
  echo ""
  echo "=============================================================="
  echo " METRICS (Stage 14 + confusion) on $PRED_DIR"
  echo "=============================================================="

  EVAL_PY="$(ls "$SDIR"/*eval_main_e7.py 2>/dev/null | head -1)"
  CONF_PY="$(ls "$SDIR"/make_confusion_e7.py 2>/dev/null | head -1)"
  SPLIT_CSV_ARG=""
  [ -n "${SKINET_SPLIT_CSV:-}" ] && SPLIT_CSV_ARG="--split-csv $SKINET_SPLIT_CSV"

  if [ -n "$EVAL_PY" ]; then
    echo "--- eval: $(basename "$EVAL_PY") ---"
    # shellcheck disable=SC2086
    python "$EVAL_PY" --split "$EVAL_SPLIT" --in-dir "$PRED_DIR" \
        --out-dir "$EVAL_OUT" $SPLIT_CSV_ARG \
      && echo "    metrics JSON/CSV -> $EVAL_OUT" \
      || echo "    WARN: eval step failed (segmentation output is still valid)"
  else
    echo "WARN: no *eval_main_e7.py in $SDIR -- skipping metrics"
  fi

  if [ -n "$CONF_PY" ]; then
    echo "--- confusion: $(basename "$CONF_PY") ---"
    # shellcheck disable=SC2086
    python "$CONF_PY" --split "$EVAL_SPLIT" --pred-dir "$PRED_DIR" \
        --out "$EVAL_OUT/confusion.csv" --scripts-dir "$SDIR" $SPLIT_CSV_ARG \
      && echo "    confusion CSV -> $EVAL_OUT/confusion.csv" \
      || echo "    WARN: confusion step failed (segmentation output is still valid)"
  fi

  # brief on-screen summary so you see numbers without opening files
  python - "$EVAL_OUT" << 'PYSUM' || true
import json, sys, glob, os
d = sys.argv[1]
js = sorted(glob.glob(os.path.join(d, "*eval*.json")))
if js:
    r = json.load(open(js[0]))
    ma = r.get("macro_all", {})
    if ma:
        print("  macro-all  F1={:.3f}  inst-Dice={:.3f}  sem-Dice={:.3f}".format(
            ma.get("f1", 0), ma.get("mean_instance_dice", 0), ma.get("semantic_dice", 0)))
    mh = r.get("macro_hard", {})
    if mh:
        print("  macro-hard F1={:.3f}  inst-Dice={:.3f}".format(
            mh.get("f1", 0), mh.get("mean_instance_dice", 0)))
PYSUM
fi

echo "=============================================================="
if [ $status -eq 0 ]; then
  echo " DONE  $(date)   (see the 'wrote -> ...' line above for outputs)"
else
  echo " FAILED (exit $status)  $(date)"
fi
echo "=============================================================="
exit $status
