#!/usr/bin/env python
"""
Generate CORTEX region mask (PNG label map) untuk WSI MAIN test memakai
cortex Mask2Former terlatih.

Alur (per WSI):
  1. baca WSI utuh main (.jpg, mis. 2176x3840) dari folder results MRCNN
  2. resize ke long-side TARGET (2048, skala training cortex)
  3. cortex M2F inference (1 gambar = 1 forward; WSI muat dalam satu frame 2048)
  4. bangun label map region (0=bg, 1=cortex, 2=capsule, 3=medullaire)
     -- prioritas saat overlap: cortex > medullaire > capsule
  5. upscale label map ke ukuran WSI asli (NEAREST) -> PNG
  6. simpan {src}.png (label map) + {src}.json (dims, ratio, area cortex)

Output: --out-dir/{src}.png , --out-dir/{src}.json

Contoh:
  python gen_cortex_mask_main.py \
    --weights  OUT_CORTEX/checkpoints/model_best.pth \
    --wsi-dir  outputs/mrcnn_cmp/work/results/test_main_.../  \
    --out-dir  outputs/.../cortex_masks_main/test \
    --sources  ML9360N14                # opsional: 1 source utk uji; kosong=semua

GPU node (MSDeformAttn) + module gcc/cuda. Jalankan via qsub.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

import e7_config as C

# prioritas label saat dua region menutupi piksel yang sama (error model):
# cortex dominan (ROI morphometry), lalu medullaire, lalu capsule.
# nilai label map: cortex=1, capsule=2, medullaire=3 (ikut CORTEX_ID e7)
PRIORITY = ["cortex", "medullaire", "capsule"]  # tinggi -> rendah


def build_predictor(weights, device="cuda"):
    """Load cortex M2F (3 kelas) via config yang sama dgn training."""
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import MetadataCatalog

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(C.BASE_CONFIG))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(C.CORTEX_CLASSES)  # 3
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(str(weights))
    return model, cfg


def infer_one(model, cfg, image_bgr, device="cuda"):
    """Satu forward M2F pada satu gambar. Return list instance:
    {class_id (0-based), score, mask(HxW bool)} pada resolusi image_bgr."""
    h, w = image_bgr.shape[:2]
    # M2F expects BGR uint8 HWC -> tensor CHW
    img = torch.as_tensor(image_bgr.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": img, "height": h, "width": w}]
    use_amp = str(device).startswith("cuda")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)[0]
    inst = outputs["instances"].to("cpu")
    res = []
    for i in range(len(inst)):
        res.append({
            "class_id": int(inst.pred_classes[i]),
            "score": float(inst.scores[i]),
            "mask": inst.pred_masks[i].numpy().astype(bool),
        })
    return res


def build_label_map(instances, H, W, score_thr):
    """Gabung instance region -> label map (0=bg,1=cortex,2=capsule,3=medullaire).
    Prioritas overlap: cortex > medullaire > capsule (tulis prioritas rendah dulu,
    tinggi menimpa)."""
    label = np.zeros((H, W), dtype=np.uint8)
    # id 0-based model -> nama
    id2name = {i: n for i, n in enumerate(C.CORTEX_CLASSES)}
    # tulis dari prioritas RENDAH ke TINGGI agar yang tinggi menimpa
    for name in reversed(PRIORITY):          # capsule, medullaire, cortex
        val = C.CORTEX_ID[name]              # 1/2/3
        for ins in instances:
            if ins["score"] < score_thr:
                continue
            if id2name[ins["class_id"]] != name:
                continue
            label[ins["mask"]] = val
    return label


def process_wsi(model, cfg, wsi_path, out_dir, target_long, score_thr, device):
    src = Path(wsi_path).stem
    img0 = cv2.imread(str(wsi_path))                 # BGR, ukuran asli
    if img0 is None:
        print(f"[skip] gagal baca {wsi_path}")
        return None
    H0, W0 = img0.shape[:2]

    # resize ke long-side target (skala training cortex)
    long0 = max(H0, W0)
    ratio = target_long / float(long0)
    Hs, Ws = round(H0 * ratio), round(W0 * ratio)
    img_s = cv2.resize(img0, (Ws, Hs), interpolation=cv2.INTER_AREA)

    # inference cortex pada gambar resize
    insts = infer_one(model, cfg, img_s, device)

    # label map di resolusi resize
    label_s = build_label_map(insts, Hs, Ws, score_thr)

    # upscale label map -> ukuran WSI asli (NEAREST agar label tetap diskret)
    label0 = cv2.resize(label_s, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # simpan PNG (label map) + JSON (dims + area)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{src}.png"), label0)

    cortex_val = C.CORTEX_ID["cortex"]
    area_cortex_px = int((label0 == cortex_val).sum())
    meta = {
        "source": src,
        "wsi_hw": [H0, W0],
        "infer_hw": [Hs, Ws],
        "resize_ratio": round(ratio, 6),
        "score_thr": score_thr,
        "area_px": {
            name: int((label0 == C.CORTEX_ID[name]).sum())
            for name in C.CORTEX_CLASSES
        },
        "n_instances": len(insts),
    }
    with open(out_dir / f"{src}.json", "w") as f:
        json.dump(meta, f, indent=2)

    pct = 100.0 * area_cortex_px / (H0 * W0)
    print(f"[ok] {src}: {len(insts)} inst -> cortex {pct:.1f}% "
          f"({H0}x{W0}, ratio {ratio:.4f})")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True,
                    help="cortex M2F checkpoint (model_best.pth)")
    ap.add_argument("--wsi-dir", required=True,
                    help="folder berisi WSI .jpg (results MRCNN) atau langsung file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="daftar source key (stem) utk diproses; kosong=semua .jpg")
    ap.add_argument("--target-long", type=int, default=2048)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wsi-glob", default="*/*.jpg",
                    help="pola cari WSI relatif wsi-dir (results: {src}/{src}.jpg)")
    args = ap.parse_args()

    model, cfg = build_predictor(args.weights, args.device)

    wsi_dir = Path(args.wsi_dir)
    # kumpulkan WSI: cari {src}/{src}.jpg tapi buang _Predicted/_clean
    cands = []
    for p in wsi_dir.glob(args.wsi_glob):
        stem = p.stem
        if stem.endswith("_Predicted") or stem.endswith("_clean") \
           or stem.endswith("_cortex") or "_Predicted" in stem:
            continue
        # results: folder {src}/{src}.jpg -> pastikan stem == nama folder
        if p.parent.name != stem:
            continue
        cands.append(p)

    if args.sources:
        want = set(args.sources)
        cands = [p for p in cands if p.stem in want]

    cands = sorted(set(cands))
    print(f"[info] {len(cands)} WSI akan diproses -> {args.out_dir}")
    print("=" * 60)

    summary = []
    for p in cands:
        m = process_wsi(model, cfg, p, args.out_dir, args.target_long,
                        args.score_thr, args.device)
        if m:
            summary.append(m)

    with open(Path(args.out_dir) / "gen_cortex_summary.json", "w") as f:
        json.dump({"n": len(summary), "sources": summary}, f, indent=2)
    print("=" * 60)
    print(f"[done] {len(summary)} mask -> {args.out_dir}")


if __name__ == "__main__":
    main()
