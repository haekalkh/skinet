# Installation

The released models were trained and run with the versions below (CCuB cluster,
NVIDIA V100S / A100). Newer versions generally work; these are the known-good ones.

| Package | Version |
|---|---|
| Python | 3.10.18 |
| PyTorch / torchvision | 1.13.1+cu116 / 0.14.1+cu116 |
| CUDA toolkit (to compile MSDeformAttn) | 11.8, gcc 11.2 |
| detectron2 | 0.6 |
| numpy | 1.26.4 |
| opencv-python | 4.13 |
| fvcore / iopath | 0.1.5.post20221221 / 0.1.9 |
| pycocotools, scikit-learn, timm, scipy, Pillow | recent |

## Steps

```bash
conda create -n skinet python=3.10 -y
conda activate skinet

# 1. PyTorch matched to your CUDA driver
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# 2. detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 3. Mask2Former and its deformable-attention CUDA op
git clone https://github.com/facebookresearch/Mask2Former.git
pip install -r Mask2Former/requirements.txt
cd Mask2Former/mask2former/modeling/pixel_decoder/ops && sh make.sh && cd -
export M2F_ROOT=$PWD/Mask2Former          # the run_*.sh scripts add it to PYTHONPATH

# 4. Pipeline requirements
pip install -r requirements.txt
```

Check:

```bash
PYTHONPATH=$M2F_ROOT python -c "import detectron2, mask2former; print('ok')"
```

`MSDeformAttn` has no half-precision kernel. `skinet_m2f/patch_msdeform_autocast.py`
narrows the autocast guard in a Mask2Former checkout so the rest of the pixel
decoder can run in FP16 (about 12% faster, identical predictions); it is optional.

CPU-only inference is possible (the CUDA op falls back to a PyTorch
implementation) but takes minutes per image.
