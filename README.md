# SKINET — Kidney-Biopsy Segmentation

Automated instance segmentation of Masson's trichrome kidney-biopsy whole-slide
images: a Mask2Former (Swin-B) detector with an optional SAM-Med2D refiner.

## Run it

You need a machine with an **NVIDIA GPU** (CUDA 11.8). On the CCuB cluster, use a
**GPU compute node**, not the login node.

```bash
# 1. get the code
git clone https://github.com/haekalkh/skinet.git
cd skinet

# 2. make the environment available (CCuB cluster)
source /work/lead/mk657917/miniconda3/etc/profile.d/conda.sh
conda activate skinet

# 3. run on one slide
bash run_local.sh /work/lead/mk657917/SKINET/data/raw/cortex/HI14010577B2.jpg
```

That's it. Replace the path with your own image (`.jpg`, `.png`, `.tif`, `.jp2`).
Add `--sam` for SAM-Med2D refinement.

## Results

Written to `~/skinet-runs/outputs/test/`:

- `<name>_overlay.png` — the slide with predicted masks drawn on top
- `<name>.json` — every instance (class, score, mask)

## Notes

- A full whole-slide image is large and takes ~20-30 min.
- On your own server (not the cluster), build the environment first with
  `conda env create -f environment.yml` (see `docs/installation.md`), then run
  step 3.
- Paths to weights/data are configurable via environment variables — see the
  top of `run_local.sh`.

## Common issues

| message | fix |
|---------|-----|
| `cuDNN ... ARCH_MISMATCH` | wrong GPU — use a compute GPU node, not the login node |
| `cannot find conda.sh` | run step 2 first |
| `Permission denied` on a path | ask the data owner for read access (chmod g+rX) |
| `cannot read <file>` | convert `.svs`/`.ndpi` to `.tif`/`.jp2` first |
