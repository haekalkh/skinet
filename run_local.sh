#!/bin/bash
# =============================================================================
#  run_local.sh -- run the SKINET pipeline. Two modes:
#
#    bash run_local.sh <image.jpg>           run HERE (needs a compatible GPU)
#    bash run_local.sh <image.jpg> --qsub    submit to a GPU node (CCuB cluster)
#    bash run_local.sh <image.jpg> --sam     add SAM-Med2D refinement
#
#  On the CCuB login node use --qsub: the login node's GPU (K80) is too old, so
#  the job must run on a V100 compute node. On your own machine with a modern
#  GPU, omit --qsub to run directly in the foreground.
#
#  CONFIG: edit the paths below, or export them before calling.
# =============================================================================
set -uo pipefail

# ---- configuration (override by exporting these before running) -------------
SKINET_ROOT="${SKINET_ROOT:-/work/lead/mk657917/SKINET}"
SKINET_OLD_ROOT="${SKINET_OLD_ROOT:-/work/lead/mk657917/skinet}"
CONDA_SH="${SKINET_CONDA_SH:-/work/lead/mk657917/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${SKINET_CONDA_ENV:-skinet}"
M2F_LIB="${SKINET_M2F_LIB:-$SKINET_OLD_ROOT/models/Mask2Former}"
SAM_LIB="${SKINET_SAM_LIB:-$SKINET_OLD_ROOT/models/SAM-Med2D}"
CUDA_MODULES="${SKINET_MODULES:-gcc/11.2.0 cuda/11.8}"
WEIGHTS="${SKINET_WEIGHTS:-$SKINET_ROOT/outputs/e7_clinical_pipeline/main_m2f_swin_b/checkpoints/model_best.pth}"
SPLIT_CSV="${SKINET_SPLIT_CSV:-$SKINET_ROOT/outputs/preprocessing/e7_clinical_pipeline/02_e7_source_split.csv}"
OUTDIR="${SKINET_OUTDIR:-$HOME/skinet-runs/outputs/test}"

# scripts dir = the scripts/ folder next to this file
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDIR="${SKINET_SCRIPTS_DIR:-$HERE/scripts}"
[ -f "$SDIR/e7_run.py" ] || { echo "ERROR: cannot find e7_run.py (set SKINET_SCRIPTS_DIR)"; exit 1; }

if [ "$#" -lt 1 ]; then
  echo "usage: bash run_local.sh <image> [flags]        # run here (needs local GPU)"
  echo "   or: bash run_local.sh <image> --qsub [flags] # submit to a GPU node (CCuB)"
  exit 1
fi

# ---- mode select: --qsub submits THIS script to a GPU compute node -----------
# On CCuB the login node has an old K80 (unsupported); --qsub lands the job on a
# V100. Without --qsub the pipeline runs right here (needs a compatible GPU).
if [ "${1:-}" = "--qsub" ] || [ "${2:-}" = "--qsub" ] || printf '%s\n' "$@" | grep -qx -- "--qsub"; then
  command -v qsub >/dev/null 2>&1 || { echo "ERROR: --qsub given but qsub not found (not on a cluster?)"; exit 1; }
  # strip --qsub from the args; keep the rest verbatim
  ARGS=(); for a in "$@"; do [ "$a" = "--qsub" ] || ARGS+=("$a"); done
  LOGDIR="${SKINET_LOGDIR:-$HOME/skinet-runs/logs}"; mkdir -p "$LOGDIR"
  QUEUE="${SKINET_QUEUE:-gpu}"
  echo "Submitting to '$QUEUE' (the job will run on a GPU compute node)..."
  qsub -q "$QUEUE" -N skinet -j y -o "$LOGDIR/skinet_\$JOB_ID.log" \
       -v "SKINET_ROOT=$SKINET_ROOT,SKINET_OLD_ROOT=$SKINET_OLD_ROOT" \
       "$HERE/run_local.sh" "${ARGS[@]}"
  echo "Use 'qstat' to see the job; logs go to $LOGDIR/"
  exit 0
fi

# If the first arg is a file, treat it as --image with auto --source.
EXTRA=()
if [ -f "$1" ]; then
  IMG="$1"; shift
  SRC="$(basename "${IMG%.*}")"
  EXTRA=(--image "$IMG" --source "$SRC" --overlay)
fi

echo "=============================================================="
echo " SKINET local run (no qsub)"
echo " host    : $(hostname)"
echo " scripts : $SDIR"
echo " image   : ${IMG:-<from --source/--all>}"
echo " outdir  : $OUTDIR"
echo "=============================================================="

# ---- environment ------------------------------------------------------------
# shellcheck disable=SC1090
source "$CONDA_SH" 2>/dev/null \
  || source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || { echo "ERROR: cannot find conda.sh (set SKINET_CONDA_SH)"; exit 1; }
conda activate "$CONDA_ENV" || { echo "ERROR: cannot activate '$CONDA_ENV'"; exit 1; }
# shellcheck disable=SC2086
module load $CUDA_MODULES 2>/dev/null || echo "WARN: GPU modules not loaded ($CUDA_MODULES)"

PP="${PYTHONPATH:-}"
[ -n "$M2F_LIB" ] && PP="$M2F_LIB:$PP"
[ -n "$SAM_LIB" ] && PP="$SAM_LIB:$PP"
export PYTHONPATH="$PP"
export SKINET_ROOT SKINET_OLD_ROOT

mkdir -p "$OUTDIR"

# ---- early GPU check: fail fast with a clear message (not mid-pipeline) -----
python - << 'PYGPU'
import sys
try:
    import torch
except Exception as e:
    print(f"WARN: could not import torch ({e})"); sys.exit(0)
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("=" * 60)
    print("WARNING: no GPU visible to PyTorch.")
    print("Run this on a GPU compute node / server, NOT the login node.")
    print("The pipeline will be extremely slow or fail without a GPU.")
    print("=" * 60)
PYGPU

# ---- run e7_run.py directly (foreground) ------------------------------------
set -x
python "$SDIR/e7_run.py" \
  --scripts-dir "$SDIR" \
  --weights "$WEIGHTS" \
  --split-csv "$SPLIT_CSV" \
  --out-dir "$OUTDIR" \
  "${EXTRA[@]}" "$@"
status=$?
set +x

echo "=============================================================="
if [ $status -eq 0 ]; then
  echo " DONE  -> results in $OUTDIR"
else
  echo " FAILED (exit $status)"
fi
echo "=============================================================="
exit $status
