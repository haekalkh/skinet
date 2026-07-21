#!/usr/bin/env python
"""
Verifikasi cortex M2F vs GT cortex (dari XML ASAP raw).
Rasterize polygon PartOfGroup="cortex" -> mask biner, bandingkan dgn cortex M2F
PNG label map (label 1 = cortex). Hitung IoU + Dice.

Koordinat XML sudah di ruang WSI 2176x3840 (sama dgn PNG) -> tak perlu skala.

  python verify_cortex_vs_gt.py \
    --xml   data/raw/main/ML9360N14.xml \
    --pred  outputs/.../cortex_masks_main/test/ML9360N14.png \
    [--cortex-label 1] [--save-overlay out.jpg --wsi data/raw/main/ML9360N14.jpg]
"""
import argparse
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def parse_cortex_polygons(xml_path, group="cortex"):
    """Ambil semua polygon PartOfGroup=group -> list of Nx2 int array."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polys = []
    for ann in root.iter("Annotation"):
        if (ann.get("PartOfGroup") or "").lower() != group.lower():
            continue
        pts = []
        for c in ann.iter("Coordinate"):
            x = float(c.get("X")); y = float(c.get("Y"))
            pts.append([x, y])
        if len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.float32))
    return polys


def rasterize(polys, H, W):
    """Polygon -> mask biner (uint8 HxW)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(mask, [np.round(p).astype(np.int32)], 1)
    return mask


def iou_dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) > 0 else 0.0
    return float(iou), float(dice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--pred", required=True, help="cortex M2F PNG label map")
    ap.add_argument("--cortex-label", type=int, default=1)
    ap.add_argument("--group", default="cortex")
    ap.add_argument("--save-overlay", default=None)
    ap.add_argument("--wsi", default=None, help="WSI jpg utk overlay")
    args = ap.parse_args()

    pred = cv2.imread(args.pred, cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise SystemExit(f"gagal baca pred {args.pred}")
    H, W = pred.shape
    pred_cortex = (pred == args.cortex_label).astype(np.uint8)

    polys = parse_cortex_polygons(args.xml, args.group)
    gt = rasterize(polys, H, W)

    iou, dice = iou_dice(pred_cortex, gt)
    pa = pred_cortex.sum(); ga = gt.sum(); tot = H * W
    print("=" * 55)
    print(f"WSI            : {H}x{W}")
    print(f"GT polygons    : {len(polys)}")
    print(f"GT cortex area : {ga} px ({100*ga/tot:.1f}%)")
    print(f"M2F cortex area: {pa} px ({100*pa/tot:.1f}%)")
    print(f"IoU            : {iou:.4f}")
    print(f"Dice           : {dice:.4f}")
    print("=" * 55)

    if args.save_overlay and args.wsi:
        wsi = cv2.imread(args.wsi)
        if wsi is not None and wsi.shape[:2] == (H, W):
            ov = wsi.copy()
            # GT hijau outline, pred merah fill semi
            ov[pred_cortex.astype(bool)] = (0, 0, 255)
            cnts, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blend = cv2.addWeighted(wsi, 0.7, ov, 0.3, 0)
            cv2.drawContours(blend, cnts, -1, (0, 255, 0), 3)
            cv2.imwrite(args.save_overlay, blend)
            print(f"overlay (GT hijau outline / M2F merah) -> {args.save_overlay}")


if __name__ == "__main__":
    main()
