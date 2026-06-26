# Installation

Two layers: (1) the lightweight `skinet` CLI, and (2) the compute environment
that the submitted job activates on the cluster.

## 1. CLI

```bash
pip install --user .
# if `skinet` isn't found, add the user bin to PATH:
echo '' >> ~/.bashrc
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Install **without** a conda env active so the command lands in your own space.

## 2. Compute environment (conda)

```bash
conda env create -f environment.yml
conda activate skinet
```

### detectron2

Install from source matching your PyTorch / CUDA build:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Mask2Former

Clone the Mask2Former repo and add it to `PYTHONPATH` (the job script does this
from `SKINET_M2F_LIB`):

```bash
git clone https://github.com/facebookresearch/Mask2Former.git
# build its custom ops per its README (MSDeformAttn)
```

### SAM-Med2D (provides `segment_anything`)

```bash
git clone https://github.com/OpenGVLab/SAM-Med2D.git
# add to PYTHONPATH via SKINET_SAM_LIB; download sam_med2d_b.pth separately
```

## 3. Configure

```bash
cp skinet/skinet.env.example skinet.env
# edit paths: scripts dir is auto (packaged) once vendored; set conda, libs,
# weights, split CSV. Then:
skinet doctor
```

`doctor` checks every path (scripts, conda, weights, split CSV, M2F/SAM libs,
Mask2Former base config) and reports anything unreadable before you submit.

## Notes

- Weights, configs, and data are referenced by absolute path (set in
  `skinet.env` / `SKINET_ROOT`); they are not part of the repo.
- GPU with ≥ 16 GB is recommended; the refiner adds ~3.4 GB over the detector.
