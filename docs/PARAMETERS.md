# SKINET — Mask2Former pipeline (M2F-ANATOMY): training-to-inference parameters

Muhammad Haekal Khowarizmi — LEAD (CNRS UMR5022 / UBE), M2 internship March–August 2026.

This package documents every setting used to produce the numbers in the thesis
*"Transformer-Based Instance Segmentation for Kidney Biopsy Analysis: From
Benchmark Performance to Clinical Readout"*. Every value below was read from the
code (`code/config.py`, `code/s0*.py`) and from the `run_meta.json` files that
each stage wrote when it finished (`records/provenance/`). Nothing is
reconstructed from memory.

Final model: **`patient_100k`** (Mask2Former, Swin-B, seed 42).
Cluster: CCuB, SGE, Tesla V100S-PCIE-32GB nodes (`webern*`), Python 3.10.18,
CUDA 11.8, gcc 11.2, conda env `skinet` (detectron2 + Mask2Former).
Working tree on the cluster: `/work/lead/mk657917/M2F-ANATOMY/merdeka` (code)
and `/work/lead/mk657917/M2F-ANATOMY/work_patient` (all final outputs).

---

## 0. Pipeline overview

| Stage | Script | What it does |
|---|---|---|
| 01 | `s01_audit.py` | Reads the ASAP XMLs, normalises labels, counts classes |
| 02 | `s02_split.py` → `s02b_split_patient.py` | Slide split, then regrouped **by patient** (removes leakage) |
| 03 | `s03_tiles.py` | Tiles ROIs into a COCO dataset (cortex-gated) |
| 04 | `s04_train.py` (`job_train.sh`) | Trains Mask2Former or Mask R-CNN |
| 05 | `s05_infer.py` (`job_infer.sh`) | Runs the model over every tile of each ROI |
| 06 | `s06_stitch.py` | Maps tile predictions back to ROI coordinates |
| 07 | `s07_postproc.py` | 7-stage post-processing ported from the legacy system |
| 08 | `s08_eval.py` | Instance-level P/R/F1 and Dice per class, per ROI |
| 09 | `s09_tune.py` | Per-class score thresholds, tuned on **val** only |
| 10 | `s10_morphometry.py` | Counts / areas / densities (MPP 0.227 µm/px) |

All defaults live in `code/config.py`; every stage accepts command-line overrides,
and the overrides actually used are recorded in each `run_meta.json`.

---

## 1. Data and split

| Item | Value |
|---|---|
| Source | CHU Dijon, Masson's trichrome, exported ROIs (mostly 3840×2176 or 1920×1088) |
| Annotated ROIs / patients | 292 / 153 |
| Pixel size | 0.227 µm/px (from the scale bar burnt into the exports) |
| Structure classes (order = COCO id − 1) | tubule_sain, tubule_atrophique, vaisseau, nsg, intima, nsg_partiel, media, veine, nsg_complet, pac |
| Excluded | artefact; region classes cortex / capsule / medullaire are used only for gating |
| Split unit | **patient** (`s02b_split_patient.py`, seed 42, 70/15/15) |
| Patients train/val/test | 108 / 22 / 23 |
| ROIs train/val/test | 204 / 44 / 44 |
| Note | The earlier slide-level split put 31 patients in more than one split; it is superseded |

Split file: `records/split/s02_split.csv`.

## 2. Tiling (stage 03)

| Parameter | Value |
|---|---|
| Tile size | 1024 px |
| Overlap ratio | 0.333 → stride 683 px |
| Keep object in a tile if visible fraction of whole object ≥ | 0.10 |
| and visible area ≥ | 100 px |
| Drop tile if cortex coverage < | 0.10 (`MIN_TILE_CORTEX`) |
| Keep empty tiles | train: no · val: yes · test: yes |
| Tiles written train/val/test | 2311 / 560 / 460 |

## 3. Training (stage 04)

### 3.1 Settings set by the pipeline

| Parameter | `patient_40k` (+ s43, s44) | `patient_100k` (final) | `patient_mrcnn` (+ s43, s44) |
|---|---|---|---|
| Architecture | Mask2Former, Swin-B | Mask2Former, Swin-B | Mask R-CNN, ResNet-101-FPN |
| Base config | `skinet/configs/mask2former/main_instance_swin_base.yaml` | same | detectron2 `COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml` |
| Initial weights | `mask2former_swin_base.pkl` (COCO) | same | `model_final_a3ec72.pkl` (COCO) |
| Batch size | 1 | 1 | 1 |
| Base LR | 2.5e-5 | 2.5e-5 | 2.5e-3 |
| Max iterations | 40 000 | 100 000 | 40 000 |
| LR schedule | step, drops at 28 000 and 36 000 | WarmupCosineLR | step, 28 000 / 36 000 |
| Augmentation | LSJ: ResizeScale 0.5–2.0 → FixedSizeCrop 1024 → RandomFlip | same **+ rot90** (0/90/180/270) | same LSJ as M2F, spelled out explicitly |
| Mask format | bitmask | bitmask | polygon |
| Eval / checkpoint period | 2 500 | 5 000 | 2 500 |
| Checkpoint kept | best `segm/AP` on val | best `segm/AP` on val | best `segm/AP` on val |
| Seeds | 42, 43, 44 | 42 | 42, 43, 44 |
| Repeat sampler / class loss weights | off | off | off |
| Best val segm/AP (iteration) | 54.41 (37 499) · 55.83 (40 000) · 54.84 (37 499) | **55.63 (54 999)** | 48.21 (37 499) · 48.68 (34 999) · 49.23 (32 499) |

Query-budget runs: `patient_q50` and `patient_q25` are `patient_40k` with
`--num-queries 50` / `25` (val AP 55.49 and 51.94).

Exact commands (from `records/provenance/s04_train/<run>/run_meta.json`), e.g. the final model:

```
s04_train.py --run patient_100k --seed 42 --max-iter 100000 --eval-period 5000 \
  --checkpoint-period 5000 --lr-schedule cosine --workers 2 --rot90 --arch mask2former
```

### 3.2 Model and optimiser settings inherited from the base YAML

These are not overridden by the pipeline. The cluster copy of each run's
`resolved_config.yaml` is no longer accessible, so the values below were
recovered from two sources. Column **Source** says which one.

- **CKPT**: read directly from the saved checkpoints `patient_100k/model_best.pth`
  and `patient_40k/model_best.pth` (weight shapes, optimiser state, LR-scheduler
  state, AMP grad-scaler). These are certain for the final runs.
- **LOG**: the full configuration dump printed at the start of an earlier
  Mask2Former training run on the same cluster (`e7_m2f_train_6899723.log`), which
  used the same initial weights (`mask2former_swin_base.pkl`) and the same base LR.
  Every value that can also be read from the checkpoints agrees with it.

**Architecture**

| Parameter | Value | Source |
|---|---|---|
| Backbone | Swin-B: embed dim 128, depths 2/2/18/2, heads 4/8/16/32, window 12, MLP ratio 4 | CKPT (heads from LOG) |
| Backbone pre-training size / drop-path | 384 / 0.3 | LOG |
| Pixel decoder | MSDeformAttnPixelDecoder, 6 encoder layers, 8 heads, 4 sampling points, conv/mask dim 256, common stride 4 | CKPT (layers) + LOG |
| Transformer decoder | MultiScaleMaskedTransformerDecoder, 9 layers, hidden 256, 8 heads, FFN 2048, post-norm | CKPT (layers, dims) + LOG |
| Object queries | 100 (50 / 25 in `patient_q50` / `patient_q25`) | CKPT |
| Class outputs | 10 classes + 1 no-object | CKPT |

**Loss and matching**

| Parameter | Value | Source |
|---|---|---|
| Class / mask (BCE) / dice loss weights | 2.0 / 5.0 / 5.0 | LOG |
| No-object weight | 0.1 | LOG |
| Deep supervision | on (loss on every decoder layer) | LOG |
| Point sampling for mask loss | 12 544 points, oversample 3.0, importance ratio 0.75 | LOG |

**Optimiser and schedule**

| Parameter | Value | Source |
|---|---|---|
| Optimiser | AdamW, betas (0.9, 0.999), eps 1e-8 | CKPT |
| Base LR / backbone LR | 2.5e-5 / 2.5e-6 (backbone multiplier 0.1) | CKPT |
| Weight decay | 0.05; 0.0 on normalisation and embedding parameters | CKPT |
| Warm-up | linear, 1 000 iterations, start factor 0.001 | LOG |
| Step schedule (40k runs) | ×0.1 at 28 000 and 36 000 (LR at it. 37 500 = 0.01 × base, confirmed) | CKPT |
| Cosine schedule (100k run) | decays to 0 over 100 000 iterations (LR at it. 55 000 = 0.422 × base, matching the cosine curve) | CKPT |
| Gradient clipping | full-model, max norm 0.01 (L2) | LOG |
| Mixed precision (AMP) | on during training (grad scaler present in checkpoint) | CKPT |
| Pixel normalisation / input format | ImageNet mean/std, BGR | LOG |

**Inference head settings**

| Parameter | Value | Source |
|---|---|---|
| Output mode | instance only (semantic / panoptic off) | LOG |
| Max detections per image | 100 | LOG |


## 4. Inference (stages 05–06)

| Parameter | Value |
|---|---|
| Weights | `work_patient/s04_train/patient_100k/model_best.pth` |
| Tile grid | identical to stage 03: 1024 px, stride 683, last tile snapped to the edge |
| Cortex gate | same as training, tile skipped if cortex < 0.10 (test: 460 of 638 tiles run) |
| Score floor | 0.05 (deliberately low; filtering is done in post-processing) |
| Input masking of non-cortex pixels | off |
| Precision | FP32 for the reported numbers; FP16 + narrowed autocast guard gives identical F1 (`test_fp16fix`) |
| Stitching | pure coordinate mapping, no fusion (16 310 instances in = out) |

## 5. Post-processing (stage 07), in execution order

| # | Step | Setting |
|---|---|---|
| 0 | Confidence | global 0.50, then per-class thresholds from stage 09 (below) |
| 0b | Cortex filter | `cortex_min_inside` 0.0 (disabled; gating is done at tile level) |
| 1 | Same-class fusion (tile overlap) | bbox IoU 0.1, mask IoU 0.1 |
| 2 | Border filter | edge contact ≥ 0.25, classes veine, nsg, intima, media |
| 3 | Orphan filter | bbox 0.3, mask 0.3; nsg must contain nsg_complet or nsg_partiel; vaisseau kept without intima/media |
| 4 | Priority (erasure) | bbox 0.3, mask 0.3; table below |
| 5 | Orphan filter (again) | as step 3 |
| 6 | Cross-class fusion | {nsg_complet, nsg_partiel}, bbox 0.1 / mask 0.1, label from the member seen whole (`whole_view`), score breaks ties |
| 7 | Small-object removal | < 50 px |

Priority rule — "row class erases these":

| Class | Erases |
|---|---|
| tubule_sain | tubule_atrophique, pac, veine, intima, media |
| tubule_atrophique | pac, veine, intima, media |
| nsg, nsg_complet, nsg_partiel | tubule_sain, tubule_atrophique, pac, vaisseau, veine |
| vaisseau | tubule_sain, tubule_atrophique, pac, veine |
| veine | tubule_sain, tubule_atrophique |
| pac, intima, media | — |

Per-class thresholds for `patient_100k` (stage 09: grid 0.50–0.95 step 0.05,
detection IoU 0.3, minimum gain 0.005, tuned on **val**; val macro F1 0.654 → 0.704):

| tubule_sain | tubule_atrophique | vaisseau | nsg | intima | nsg_partiel | media | veine | nsg_complet | pac |
|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.75 | 0.80 | 0.50 | 0.85 | 0.50 | 0.60 | 0.80 | 0.60 | 0.85 |

Funnel on the test set: 16 310 → 7 413 (confidence) → 2 779 (fusion) → 2 750
(border) → 2 732 (orphan) → 2 289 (priority) → 2 287 → 2 261 (final).

Thresholds for the other runs: `records/thresholds/`.

## 6. Evaluation (stage 08)

| Parameter | Value |
|---|---|
| Matching | one-to-one within class, detection IoU 0.30 (strict 0.50 also reported) |
| Metrics | per-class precision / recall / F1, macro F1, mean Dice of matched pairs |
| Uncertainty | 3 training seeds + paired patient bootstrap (`bootstrap_patients.py`, 10 000 draws over the 23 test patients) |

Test result, `patient_100k` (`records/results/s08_eval/patient_100k/test_base050/`):
precision 0.727, recall 0.736, **macro F1 0.723**, strict F1 0.700, **mean Dice 0.917**.

## 7. Cortex model (used for gating)

Mask2Former Swin-B, 3 classes (cortex, capsule, medullaire); ROI resized so the
long side is 2048, tiled at 2048 / stride 1366, score 0.50. Test: AP 70.9,
macro F1 0.867, mean Dice 0.947.

## 8. Where things are in this repository

- Code: `skinet_m2f/` (all defaults in `skinet_m2f/config.py`).
- Mask2Former config: `configs/main_instance_swin_base.yaml`.
- Per-class thresholds of the released model: `configs/thresholds/patient_100k.json`.
- Run records: `records/provenance/<stage>/<run>/run_meta.json`; split in
  `records/split/`; evaluation outputs in `records/results/s08_eval/`.
- The configuration dump used for section 3.2: `records/config_dump/`.
- Released models: `models/README.md`.
