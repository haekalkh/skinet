#!/usr/bin/env python
"""
E7 Stage 11b - cortex overlay (visual sanity-check of the stitched region mask).

Reconstructs the low-magnification WSI frame from the cortex COCO tiles and
overlays the saved Stage-11 label map (cortex/capsule/medullaire) so outliers
(0% cortex, ~100% cortex, class overlap) can be judged: model error vs real
anatomy. Writes one PNG per requested source.

Input : cortex COCO tiles + OUT_CORTEX/cortex_masks/{split}/{src}.png (Stage 11)
Output: OUT_CORTEX/cortex_masks/{split}/overlays/{src}.png
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import e7_config as C

# BGR colors per class id (1=cortex,2=capsule,3=medullaire)
COLORS = {1: (0, 200, 0), 2: (255, 80, 0), 3: (0, 0, 230)}


def reconstruct_lowmag(coco_dir, tiles, W_lm, H_lm):
    canvas = np.full((H_lm, W_lm, 3), 255, np.uint8)
    for im in tiles:
        t = cv2.imread(str(coco_dir / im["file_name"]))
        if t is None:
            continue
        x0, y0, ts = int(im["tile_x"]), int(im["tile_y"]), int(im["tile_size"])
        vh, vw = min(ts, H_lm - y0), min(ts, W_lm - x0)
        if vh <= 0 or vw <= 0:
            continue
        canvas[y0:y0 + vh, x0:x0 + vw] = t[:vh, :vw]
    return canvas


def main():
    ap = argparse.ArgumentParser("E7 Stage 11b cortex overlay")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--coco-dir", default=str(C.COCO_CORTEX_DIR))
    ap.add_argument("--mask-dir", default=None,
                    help="dir with Stage-11 {src}.png label maps")
    ap.add_argument("--sources", default="",
                    help="comma-separated source_keys; empty = first --limit")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    coco_dir = Path(args.coco_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else (C.OUT_CORTEX / "cortex_masks" / args.split)
    out_dir = mask_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.load(open(coco_dir / "annotations" / f"instances_{args.split}.json"))
    by_source = defaultdict(list)
    for im in coco["images"]:
        by_source[im["source_key"]].append(im)

    if args.sources.strip():
        sources = [s for s in args.sources.split(",") if s.strip()]
    else:
        sources = sorted(by_source)[: args.limit]

    legend = "cortex=green  capsule=orange  medullaire=red"
    print(f"overlay {len(sources)} sources ({legend})")
    for src in sources:
        tiles = by_source.get(src)
        if not tiles:
            print(f"  [skip] {src}: no tiles"); continue
        t0 = tiles[0]
        ow, oh, ratio = int(t0["orig_width"]), int(t0["orig_height"]), float(t0["resize_ratio"])
        W_lm, H_lm = round(ow * ratio), round(oh * ratio)

        img = reconstruct_lowmag(coco_dir, tiles, W_lm, H_lm)
        label = cv2.imread(str(mask_dir / f"{src}.png"), cv2.IMREAD_GRAYSCALE)
        if label is None:
            print(f"  [skip] {src}: no label map at {mask_dir}/{src}.png"); continue

        color = np.zeros_like(img)
        for cid, bgr in COLORS.items():
            color[label == cid] = bgr
        m = label > 0
        ov = img.copy()
        ov[m] = (args.alpha * color[m] + (1 - args.alpha) * img[m]).astype(np.uint8)
        cv2.putText(ov, f"{src}  {legend}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{src}.png"), ov)
        cov = float(m.mean())
        print(f"  {src}: low-mag {W_lm}x{H_lm}  mask-cover {cov:.1%}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()