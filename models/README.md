# Models

`../download_models.sh` fetches the two released models from the
[v1.0 release](https://github.com/haekalkh/skinet/releases/tag/v1.0) and checks
them against `SHA256SUMS`.

| File | Model | Training | Selected at |
|---|---|---|---|
| `skinet_m2f_structures.pth` | Mask2Former Swin-B, 10 structure classes (run `patient_100k`) | 100k iterations, cosine LR, rot90, seed 42, patient-level split | iteration 54 999, best val segm/AP 55.63 |
| `skinet_m2f_cortex.pth` | Mask2Former Swin-B, cortex / capsule / medullaire | images resized to long side 2048, tiles 2048 / stride 1366 | iteration 17 499, best val AP 70.19 |

Both are full detectron2 checkpoints (weights plus optimiser and scheduler state),
loadable with `DetectionCheckpointer` or by passing the path as `MODEL.WEIGHTS`.

## Pretrained weights for training from scratch

Not needed for inference. Place them in `models/pretrained/`:

- `mask2former_swin_base.pkl` — COCO instance-segmentation Mask2Former, Swin-B
  (IN21k, 384), from the Mask2Former model zoo.
- `model_final_a3ec72.pkl` — only for the Mask R-CNN comparison arm, from the
  detectron2 model zoo (`mask_rcnn_R_101_FPN_3x`).
