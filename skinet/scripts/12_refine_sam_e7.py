#!/usr/bin/env python
"""
E7 Stage 12 - gated SAM-Med2D + Kidney Adapter refiner.

For each Stage-10 MAIN instance of a refine-eligible class, crop its ROI
(bbox + pad) from the native source field, run SAM-Med2D + the ROI-crop Kidney
Adapter with the M2F box as prompt, clean the mask (largest connected
component + small-hole fill), then DECIDE via a do-no-harm gate whether to keep
the SAM mask or fall back to the original M2F mask. Output uses the same schema
as Stage 10 so Stage 14/15 consume it unchanged.

Why gated + ROI-crop: the per-class oracle screen (Stage 3) showed SAM at
tile-256 hurt thin structures, but ROI-crop (matched retrain) lifts the
intima ceiling above the M2F mask. The gate guarantees SAM never degrades an
instance; the per-class benefit (does it help on M2F boxes?) is then measured
by running Stage 14 on this output vs the Stage-10 baseline.

Eligible classes default to the validated winners (intima + vessel/grade);
freeze the final list from the val Stage-14 diff.

GPU. Input: Stage-10 dir + 02_e7_source_split.csv (native source fields).
Adapter: E7_roicrop_256/checkpoints/best_model.pth.
"""
import argparse, json
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

import e7_config as C
from e7_sam_dataset import PIXEL_MEAN, PIXEL_STD
from segment_anything import sam_model_registry
from kidney_adapter import build_kidney_adapter

OLD_ROOT = Path(str(C.ROOT).replace("SKINET", "skinet"))
DEFAULT_BEST = C.ROOT / "outputs/e7_clinical_pipeline/sam_kidney_adapter/E7_roicrop_256/checkpoints/best_model.pth"
DEFAULT_PRETRAINED = OLD_ROOT / "pretrained" / "sam_med2d_b.pth"
DEFAULT_REFINE = "all"


# ---------- mask helpers ----------
def rle_decode(rle):
    return mask_util.decode({"size": rle["size"],
                             "counts": rle["counts"].encode("ascii")})


def rle_encode(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("ascii")}


def largest_cc(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m.astype(np.uint8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8)


def fill_small_holes(m, max_frac=0.10):
    """Fill interior holes smaller than max_frac of mask area (keeps a large
    legitimate lumen, e.g. the arterial intima ring centre)."""
    m = m.astype(np.uint8)
    a = int(m.sum())
    if a == 0:
        return m
    inv = (1 - m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate(
        [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    out = m.copy()
    for i in range(1, n):
        if i in border:                       # background open to the border
            continue
        if (lab == i).sum() < max_frac * a:   # small interior hole -> fill
            out[lab == i] = 1
    return out


def clean_mask(m):
    return fill_small_holes(largest_cc(m))


def iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def n_components(m):
    n, _ = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    return n - 1                                  # minus background


# ---------- SAM forward on one ROI ----------
@torch.no_grad()
def refine_instance(model, adapter, img_bgr, bbox_xywh, pad, target, device):
    H, W = img_bgr.shape[:2]
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    bw, bh = max(1.0, w), max(1.0, h)
    cx1 = int(max(0, np.floor(x1 - pad * bw))); cy1 = int(max(0, np.floor(y1 - pad * bh)))
    cx2 = int(min(W, np.ceil(x2 + pad * bw)));  cy2 = int(min(H, np.ceil(y2 + pad * bh)))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    crop = img_bgr[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    sx, sy = target / cw, target / ch
    # box in crop frame -> 256 frame
    bx1 = max(0.0, min((x1 - cx1) * sx, target - 1)); by1 = max(0.0, min((y1 - cy1) * sy, target - 1))
    bx2 = max(bx1 + 1.0, min((x2 - cx1) * sx, float(target))); by2 = max(by1 + 1.0, min((y2 - cy1) * sy, float(target)))
    img = cv2.resize(crop, (target, target), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - PIXEL_MEAN) / PIXEL_STD
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    box_t = torch.tensor([[bx1, by1, bx2, by2]], dtype=torch.float32, device=device)

    emb = model.image_encoder(img_t)
    sparse, dense = model.prompt_encoder(points=None, boxes=box_t, masks=None)
    pred, _ = model.mask_decoder(
        image_embeddings=adapter(emb),
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False)
    pred = F.interpolate(pred, size=(ch, cw), mode="bilinear", align_corners=False)
    m_crop = (torch.sigmoid(pred)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    m_crop = clean_mask(m_crop)
    full = np.zeros((H, W), np.uint8)
    full[cy1:cy2, cx1:cx2] = m_crop
    return full


def main():
    ap = argparse.ArgumentParser("E7 Stage 12 gated SAM refiner")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None, help="Stage-10 filtered dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--best", default=str(DEFAULT_BEST))
    ap.add_argument("--checkpoint", default=str(DEFAULT_PRETRAINED))
    ap.add_argument("--model-type", default="vit_b")
    ap.add_argument("--target-size", type=int, default=256)
    ap.add_argument("--roi-pad", type=float, default=0.30)
    ap.add_argument("--refine-classes", default=DEFAULT_REFINE,
                    help="comma-separated class names, or 'all'")
    ap.add_argument("--gate-mode", default="oversegment",
                    choices=["oversegment", "legacy"],
                    help="oversegment: only trim/clean when M2F is over-segmented; "
                         "legacy: symmetric IoU/area band")
    # oversegment-mode gate: only modestly TRIM a single-component bloated mask
    ap.add_argument("--trim-lo", type=float, default=0.60,
                    help="accept only if area_SAM/area_M2F >= this (cap how much SAM may trim)")
    ap.add_argument("--trim-hi", type=float, default=0.95,
                    help="accept only if area_SAM/area_M2F <= this (require a real trim)")
    ap.add_argument("--gate-iou-lo", type=float, default=0.60,
                    help="min IoU(SAM,M2F): SAM must be the same object, only tightened")
    ap.add_argument("--min-area", type=int, default=16,
                    help="reject SAM masks smaller than this (px), sanity")
    # legacy-mode gate
    ap.add_argument("--gate-iou-hi", type=float, default=0.99)
    ap.add_argument("--gate-area-lo", type=float, default=0.5)
    ap.add_argument("--gate-area-hi", type=float, default=2.0)
    ap.add_argument("--adapter-variant", default="B")
    ap.add_argument("--adapter-bottleneck", type=int, default=64)
    ap.add_argument("--adapter-dropout", type=float, default=0.1)
    args = ap.parse_args()
    args.image_size = args.target_size
    args.sam_checkpoint = args.checkpoint
    args.encoder_adapter = False

    if args.refine_classes.strip().lower() == "all":
        refine_set = set(C.MAIN_CLASSES)
    else:
        refine_set = {c.strip() for c in args.refine_classes.split(",") if c.strip()}
    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "postproc" / args.split / "10_filtered")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "postproc" / args.split / "12_refined")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.split_csv)
    src2path = {r["source_key"]: r["image_path"] for _, r in df.iterrows()}

    print("=" * 74)
    print("E7 Stage 12 - gated SAM-Med2D refiner")
    print(f"  split   : {args.split}   in {in_dir}")
    print(f"  refine  : {sorted(refine_set)}")
    if args.gate_mode == "oversegment":
        print(f"  gate    : TRIM-only | accept SAM iff M2F is a single component AND")
        print(f"            SAM trims it to area ratio in [{args.trim_lo},{args.trim_hi}] "
              f"with IoU(SAM,M2F) >= {args.gate_iou_lo}  (no merge/split, no inflate)")
        print(f"            roi_pad {args.roi_pad}  min_area {args.min_area}px")
    else:
        print(f"  gate    : LEGACY IoU[{args.gate_iou_lo},{args.gate_iou_hi}] "
              f"area[{args.gate_area_lo},{args.gate_area_hi}]  roi_pad {args.roi_pad}")
    print(f"  adapter : {args.best}")
    print("=" * 74)

    model = sam_model_registry[args.model_type](args).to(device)
    adapter = build_kidney_adapter(variant=args.adapter_variant, d_model=256,
                                   bottleneck=args.adapter_bottleneck,
                                   dropout=args.adapter_dropout).to(device)
    ck = torch.load(args.best, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    adapter.load_state_dict(ck["adapter"])
    model.eval(); adapter.eval()
    print(f"  loaded adapter epoch {ck.get('epoch','?')}")

    files = [f for f in in_dir.glob("*.json") if not f.name.startswith(("10_", "12_"))]
    if not files:
        raise FileNotFoundError(f"No Stage-10 source JSONs in {in_dir}")
    stat = defaultdict(lambda: {"att": 0, "acc": 0, "frag": 0})
    n_src = 0

    for pf in sorted(files):
        rec = json.load(open(pf))
        src = rec["source_key"]
        path = src2path.get(src)
        img = cv2.imread(str(path)) if path else None
        if img is None:
            print(f"  [skip] {src}: source image not found"); continue
        n_src += 1
        for it in rec["instances"]:
            cls = it["class_name"]
            if cls not in refine_set:
                it["refined"] = False
                continue
            m2f = rle_decode(it["mask_rle"])
            s = stat[cls]; s["att"] += 1
            it["refined"] = False
            sam = refine_instance(model, adapter, img, it["bbox_global"],
                                  args.roi_pad, args.target_size, device)
            if sam is None or sam.sum() == 0:
                continue
            a_s, a_m = float(sam.sum()), float(m2f.sum())
            if a_m == 0 or a_s < args.min_area:
                continue
            j = iou(sam, m2f)
            if args.gate_mode == "oversegment":
                # SAM is a single clean component (largest_cc + hole-fill). Only
                # TRIM a single-component M2F mask that is modestly too big.
                # Reject multi-component M2F (merge/split cases: SAM bridges two
                # objects or drops one) and reject over-trims / inflation.
                ncc = n_components(m2f)
                ratio = a_s / a_m                          # SAM smaller => ratio < 1
                if ncc > 1:                                # M2F is fragmented/merged
                    s["frag"] += 1
                    accept = False
                else:
                    accept = (args.trim_lo <= ratio <= args.trim_hi
                              and j >= args.gate_iou_lo)
            else:  # legacy symmetric band
                r = a_s / a_m
                accept = (args.gate_area_lo <= r <= args.gate_area_hi
                          and args.gate_iou_lo <= j < args.gate_iou_hi)
            if accept:
                ys, xs = np.where(sam)
                x0, y0 = int(xs.min()), int(ys.min())
                it["mask_rle"] = rle_encode(sam)
                it["bbox_global"] = [x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)]
                it["refined"] = True
                s["acc"] += 1
        json.dump(rec, open(out_dir / f"{src}.json", "w"))

    print("-" * 74)
    print(f"{'class':<18}{'att':>6}{'multiCC':>8}{'acc':>6}{'rate':>8}")
    for c in sorted(stat):
        s = stat[c]
        print(f"{c:<18}{s['att']:>6}{s['frag']:>8}{s['acc']:>6}"
              f"{(s['acc'] / s['att'] if s['att'] else 0):>8.1%}")
    summary = {"split": args.split, "gate_mode": args.gate_mode,
               "refine_classes": sorted(refine_set), "num_sources": n_src,
               "per_class": {c: dict(s) for c, s in stat.items()}}
    json.dump(summary, open(out_dir / "12_refine_summary.json", "w"), indent=2)
    print(f"\nSaved refined instances -> {out_dir}")
    print("NEXT: run Stage 14 on this dir and on the Stage-10 dir; compare per-class")
    print("iDice/F1. Keep refinement only for classes with val delta > 0 (freeze list).")


if __name__ == "__main__":
    main()