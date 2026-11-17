#!/bin/bash
# One-command installer for SKINET-M2F.
#
#   ./install.sh                 # GPU (CUDA 11.x driver), env name "skinet"
#   ./install.sh --cpu           # CPU only (inference works, but slowly)
#   ./install.sh --name myenv    # another conda env name
#   ./install.sh --skip-env      # install into the currently active env
#
# Steps: conda env (Python 3.10 + requirements) -> PyTorch 1.13.1 ->
# detectron2 -> Mask2Former in third_party/ with its MSDeformAttn CUDA op ->
# import check. These are the versions the released models were trained with.

set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME=skinet; CPU=0; SKIP_ENV=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) CPU=1; shift;;
        --name) ENV_NAME="$2"; shift 2;;
        --skip-env) SKIP_ENV=1; shift;;
        -h|--help) sed -n 2,13p "$0"; exit 0;;
        *) echo "unknown option $1"; exit 2;;
    esac
done
step() { echo; echo "=== $*"; }

if [ "$SKIP_ENV" = "0" ]; then
    command -v conda >/dev/null || { echo "conda not found: install Miniconda first"; exit 3; }
    step "1/5 conda env '$ENV_NAME'"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "env exists, updating"
        conda env update -n "$ENV_NAME" -f "$HERE/environment.yml"
    else
        (cd "$HERE" && conda env create -n "$ENV_NAME" -f environment.yml)
    fi
    conda activate "$ENV_NAME"
else
    step "1/5 using the active environment ($(python -c 'import sys;print(sys.prefix)'))"
    pip install -r "$HERE/requirements.txt"
fi

step "2/5 PyTorch 1.13.1"
if [ "$CPU" = "1" ]; then
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
        --extra-index-url https://download.pytorch.org/whl/cu116
fi

step "3/5 detectron2"
python -c "import detectron2" 2>/dev/null && echo "already installed" || \
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

step "4/5 Mask2Former -> third_party/Mask2Former"
M2F="$HERE/third_party/Mask2Former"
if [ ! -d "$M2F" ]; then
    mkdir -p "$HERE/third_party"
    git clone https://github.com/facebookresearch/Mask2Former.git "$M2F"
fi
pip install -r "$M2F/requirements.txt"
if [ "$CPU" = "1" ]; then
    echo "CPU install: skipping the MSDeformAttn CUDA op (PyTorch fallback is used)"
elif ! command -v nvcc >/dev/null; then
    echo "WARNING: nvcc not found, the MSDeformAttn CUDA op is NOT compiled."
    echo "Load a CUDA toolkit (11.6-11.8) and rerun ./install.sh, or inference will be slow."
else
    (cd "$M2F/mask2former/modeling/pixel_decoder/ops" && sh make.sh)
fi

step "5/5 check"
PYTHONPATH="$M2F:$HERE/skinet_m2f" python - <<'PY'
import torch, detectron2, mask2former
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("detectron2", detectron2.__version__, "| mask2former OK")
try:
    import MultiScaleDeformableAttention  # noqa
    print("MSDeformAttn CUDA op: compiled")
except Exception:
    print("MSDeformAttn CUDA op: not compiled (PyTorch fallback)")
import config as C
print("pipeline config OK, repo:", C.REPO_ROOT)
PY

echo
echo "Done. Next:"
echo "  conda activate $ENV_NAME"
echo "  ./download_models.sh"
echo "  ./run_inference.sh /path/to/image.tif"
