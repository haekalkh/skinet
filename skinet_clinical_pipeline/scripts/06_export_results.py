#!/usr/bin/env python
"""
Langkah 5 (Cara B) - EXPORT hasil 10b (cortex-gated) ke format MRCNN per-source.

Meniru struktur output MRCNN:
  {out}/{src}/
    {src}.json               LabelMe (shapes: label + polygon points)
    {src}.xml                ASAP    (Annotation Polygon + Coordinates)
    {src}_Predicted.jpg      overlay struktur di WSI (warna per kelas)

Input : 10b output {src}.json (instances RLE, cortex-gated)
        WSI .jpg (utk overlay + ukuran)

  python export_results.py \
    --gated .../10b_cortex_gated/{src}.json  (atau --gated-dir + --sources) \
    --wsi-dir  .../results/test_main_.../  (cari {src}/{src}.jpg) \
    --out-dir  .../export/test

CPU stage.
"""
import argparse
import json
import os
import glob
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

# palette RGB per kelas (mirror overlay style 26_pngGate)
PALETTE = {
    "tubule_sain": (237, 125, 49), "tubule_atrophique": (91, 60, 196),
    "vaisseau": (76, 175, 80), "nsg": (0, 200, 220), "intima": (233, 30, 99),
    "nsg_partiel": (235, 205, 30), "media": (66, 135, 245), "veine": (120, 120, 120),
    "nsg_complet": (156, 39, 176), "pac": (0, 150, 136),
}


def rle_to_mask(r):
    if isinstance(r["counts"], str):
        r = {"size": r["size"], "counts": r["counts"].encode("ascii")}
    return mask_util.decode(r).astype(np.uint8)


def mask_to_polys(mask, min_pts=3):
    """Kontur eksternal -> list polygon [[x,y],...]."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if len(c) < min_pts:
            continue
        polys.append(c.reshape(-1, 2).tolist())
    return polys


def to_labelme(instances, H, W, src):
    """Bangun dict LabelMe (shapes: label + points)."""
    shapes = []
    per_class_idx = {}
    gi = 0
    for it in instances:
        cls = it["class_name"]
        m = rle_to_mask(it["mask_rle"])
        for poly in mask_to_polys(m):
            ci = per_class_idx.get(cls, 0)
            shapes.append({
                "label": f"{cls} {ci} ({gi})",
                "points": poly,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            })
            per_class_idx[cls] = ci + 1
            gi += 1
    return {
        "version": "4.5.6", "flags": {}, "shapes": shapes,
        "imageHeight": H, "imageWidth": W,
        "imageData": None, "imagePath": f"{src}.jpg",
    }


def to_asap_xml(instances, src):
    """Bangun XML ASAP (Annotation Polygon + Coordinates + Groups per kelas)."""
    lines = ['<?xml version="1.0"?>', '<ASAP_Annotations>', '\t<Annotations>']
    gi = 0
    groups = set()
    for it in instances:
        cls = it["class_name"]
        groups.add(cls)
        m = rle_to_mask(it["mask_rle"])
        for poly in mask_to_polys(m):
            color = PALETTE.get(cls, (255, 255, 0))
            hexc = "#%02X%02X%02X" % color
            lines.append(f'\t\t<Annotation Name="Annotation {gi}" Type="Polygon" '
                         f'PartOfGroup="{cls}" Color="{hexc}">')
            lines.append('\t\t\t<Coordinates>')
            for k, (x, y) in enumerate(poly):
                lines.append(f'\t\t\t\t<Coordinate Order="{k}" X="{x}" Y="{y}" />')
            lines.append('\t\t\t</Coordinates>')
            lines.append('\t\t</Annotation>')
            gi += 1
    lines.append('\t</Annotations>')
    lines.append('\t<AnnotationGroups>')
    for g in sorted(groups):
        color = PALETTE.get(g, (255, 255, 0))
        hexc = "#%02X%02X%02X" % color
        lines.append(f'\t\t<Group Name="{g}" PartOfGroup="None" Color="{hexc}">')
        lines.append('\t\t\t<Attributes />')
        lines.append('\t\t</Group>')
    lines.append('\t</AnnotationGroups>')
    lines.append('</ASAP_Annotations>')
    return "\n".join(lines)


def _bgr(cls):
    c = PALETTE.get(cls, (255, 255, 0))
    return (c[2], c[1], c[0])


def make_overlay_clean(wsi, instances, alpha=0.45):
    """_Predicted_clean.jpg ala MRCNN: MASK berwarna saja (tanpa label/bbox)."""
    ov = wsi.copy()
    for it in instances:
        m = rle_to_mask(it["mask_rle"]).astype(bool)
        ov[m] = _bgr(it["class_name"])
    blend = cv2.addWeighted(wsi, 1 - alpha, ov, alpha, 0)
    # kontur tegas biar batas objek jelas
    for it in instances:
        m = rle_to_mask(it["mask_rle"])
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blend, cnts, -1, _bgr(it["class_name"]), 2)
    return blend


def _draw_dashed_rect(img, x1, y1, x2, y2, color, thick=2, dash=8, gap=6):
    """Bounding box garis putus-putus."""
    def dline(p1, p2):
        (xa, ya), (xb, yb) = p1, p2
        dist = int(np.hypot(xb - xa, yb - ya))
        if dist == 0:
            return
        for i in range(0, dist, dash + gap):
            s = i / dist
            e = min((i + dash) / dist, 1.0)
            xs, ys = int(xa + (xb - xa) * s), int(ya + (yb - ya) * s)
            xe, ye = int(xa + (xb - xa) * e), int(ya + (yb - ya) * e)
            cv2.line(img, (xs, ys), (xe, ye), color, thick)
    dline((x1, y1), (x2, y1)); dline((x2, y1), (x2, y2))
    dline((x2, y2), (x1, y2)); dline((x1, y2), (x1, y1))


def make_overlay_predicted(wsi, instances, alpha=0.45):
    """_Predicted.jpg ala MRCNN: mask + bbox putus-putus + label nama objek."""
    blend = make_overlay_clean(wsi, instances, alpha)
    for it in instances:
        cls = it["class_name"]
        color = _bgr(cls)
        m = rle_to_mask(it["mask_rle"])
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        _draw_dashed_rect(blend, x1, y1, x2, y2, color, thick=2)
        # label + skor -- font besar, teks warna kelas di atas kotak hitam
        label = cls
        if "score" in it:
            label = f"{cls} {it['score']:.2f}"
        fs = 0.9          # font scale (besar)
        ft = 2            # thickness
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        ty = max(y1 - 6, th + 4)
        # kotak hitam latar label (kontras)
        cv2.rectangle(blend, (x1, ty - th - 6), (x1 + tw + 6, ty + base), (0, 0, 0), -1)
        # teks warna kelas
        cv2.putText(blend, label, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft, cv2.LINE_AA)
    return blend


def find_wsi(wsi_dir, src):
    for cand in [os.path.join(wsi_dir, src, f"{src}.jpg"),
                 os.path.join(wsi_dir, f"{src}.jpg")]:
        if os.path.exists(cand):
            return cand
    return None


def export_one(gated_json, wsi_dir, out_dir):
    d = json.load(open(gated_json))
    src = d["source_key"]
    H, W = d["height"], d["width"]
    insts = d["instances"]

    odir = Path(out_dir) / src
    odir.mkdir(parents=True, exist_ok=True)

    # LabelMe
    with open(odir / f"{src}.json", "w") as f:
        json.dump(to_labelme(insts, H, W, src), f)
    # ASAP
    with open(odir / f"{src}.xml", "w") as f:
        f.write(to_asap_xml(insts, src))
    # overlay (dua gaya ala MRCNN)
    wsi_path = find_wsi(wsi_dir, src)
    if wsi_path:
        wsi = cv2.imread(wsi_path)
        if wsi is not None:
            if wsi.shape[:2] != (H, W):
                wsi = cv2.resize(wsi, (W, H))
            # _Predicted_clean.jpg : mask saja
            cv2.imwrite(str(odir / f"{src}_Predicted_clean.jpg"),
                        make_overlay_clean(wsi, insts))
            # _Predicted.jpg : mask + bbox putus-putus + label
            cv2.imwrite(str(odir / f"{src}_Predicted.jpg"),
                        make_overlay_predicted(wsi, insts))
    print(f"[ok] {src}: {len(insts)} inst -> {odir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default=None, help="satu file 10b {src}.json")
    ap.add_argument("--gated-dir", default=None)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    files = []
    if args.gated:
        files = [args.gated]
    else:
        files = sorted(glob.glob(os.path.join(args.gated_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith(("10", "0"))
                 and "summary" not in f and "manifest" not in f]
        if args.sources:
            want = set(args.sources)
            files = [f for f in files
                     if os.path.basename(f).replace(".json", "") in want]

    for f in files:
        export_one(f, args.wsi_dir, args.out_dir)
    print(f"[done] {len(files)} source -> {args.out_dir}")


if __name__ == "__main__":
    main()
