# SKINET-M2F

Instance segmentation of kidney biopsies (Masson's trichrome) with Mask2Former,
from raw image to morphometry report. Ten renal structures are segmented —
non-atrophic and atrophic tubules, arteries with their internal and external
elastic laminae, veins, and glomeruli with their subtypes — inside a cortex mask
predicted by a second model, then post-processed and turned into counts, areas
and densities per mm² of cortex.

This is the pipeline and the trained models behind the M2 thesis
*Transformer-Based Instance Segmentation for Kidney Biopsy Analysis: From
Benchmark Performance to Clinical Readout* (M. H. Khowarizmi, LEAD — CNRS
UMR5022 / Université Bourgogne Europe, with CHU Dijon Bourgogne, 2026).

| | |
|---|---|
| Structure model | Mask2Former, Swin-B, 10 classes — test macro F1 **0.723**, mean Dice **0.917** (23 held-out patients) |
| Cortex model | Mask2Former, Swin-B, 3 classes (cortex, capsule, medulla) — macro F1 0.867, Dice 0.947 |
| Input | RGB exports at about 0.227 µm/pixel (e.g. 3840×2176 regions) |

## 1. Install

See [docs/INSTALL.md](docs/INSTALL.md). In short: Python 3.10, PyTorch with CUDA,
detectron2, and the Mask2Former repository with its MSDeformAttn CUDA op compiled.

## 2. Get the models

```bash
./download_models.sh
```

This puts `skinet_m2f_structures.pth` and `skinet_m2f_cortex.pth` (about 1.3 GB
each) in `models/` and checks their SHA-256. They are hosted as assets of the
[v1.0 release](https://github.com/haekalkh/skinet/releases/tag/v1.0).

## 3. Use it — inference on new images

No annotation is needed.

```bash
./run_inference.sh /path/to/biopsy_A.tif                          # one image
./run_inference.sh --case patient01 /path/to/a.tif /path/to/b.tif # one biopsy, several images
```

Output in `reports/`: `morphometry.csv` (per image or per case: counts, areas,
densities per mm² of cortex, atrophic share of tubular area) and an overlay per
image for visual review. Run `./run_inference.sh --help` for options.

## 4. Train on your own data

Put the images and their ASAP XML annotations (same file stem) in a folder, edit
the class list in `skinet_m2f/config.py` if your labels differ, then:

```bash
SKINET_RAW_DIR=/path/to/data ./run_training.sh my_run          # 40k-iteration recipe
./run_evaluation.sh my_run                                     # thresholds on val, scores on test
```

To fine-tune from the released model instead of COCO, add
`--weights models/skinet_m2f_structures.pth` to `run_training.sh`. The exact
recipe of the released model is in `docs/PARAMETERS.md`.

## 5. Pipeline

| Stage | Script | |
|---|---|---|
| 01 | `s01_audit.py` | inventory images and XML annotations |
| 02 | `s02_split.py`, `s02b_split_patient.py` | train / val / test split, grouped by patient |
| 03 | `s03_tiles.py` | 1024-px tiles, stride 683, cortex-gated, COCO format |
| 04 | `s04_train.py` | train Mask2Former (or Mask R-CNN for comparison) |
| 05 | `s05_infer.py` | tile-wise inference; cortex predicted when no annotation |
| 06 | `s06_stitch.py` | map tile predictions back to image coordinates |
| 07 | `s07_postproc.py` | fusion, border, containment, priority, small-object rules |
| 08 | `s08_eval.py` | instance P / R / F1 and Dice per class |
| 09 | `s09_tune.py` | per-class score thresholds, tuned on validation only |
| 10 | `s10_morphometry.py` | counts, areas, densities |

Every setting lives in `skinet_m2f/config.py`; every path can be overridden by an
environment variable (`SKINET_RAW_DIR`, `M2F_WORK_DIR`, `SKINET_WEIGHTS`,
`SKINET_CORTEX_WEIGHTS`, …). Every stage writes a `run_meta.json` recording the
command and settings that produced its output.

## Repository layout

```
run_inference.sh  run_training.sh  run_evaluation.sh  download_models.sh
skinet_m2f/        pipeline stages and config.py
configs/           Mask2Former config, per-class thresholds of the released model
models/            released weights (downloaded), SHA256SUMS
docs/              INSTALL.md, PARAMETERS.md (every training-to-inference setting)
jobs/ccub_sge/     original SGE job scripts used on the CCuB cluster
analysis/          scripts behind the thesis figures and analyses
records/           run_meta.json of every stage of the reported runs, split, results
```

## Limits worth knowing

- Trained on 292 regions from 153 patients of one centre (CHU Dijon), one stain.
  Other centres, scanners or stains will need checking and probably fine-tuning.
- Precision figures are lower bounds: the reference annotations are incomplete,
  and some "false positives" are real but unannotated structures.
- Atrophy is systematically understated by the inherited priority rule;
  `run_inference.sh` resolves the tubular pair by score by default
  (`--no-score-resolve` restores the rule used for the reported numbers).
- The report is not a graded IFTA score: interstitial fibrosis is not segmented.
- Research use only; not a medical device.

## Contact

Muhammad Haekal Khowarizmi — LEAD, Université Bourgogne Europe / Universitas Gunadarma.
