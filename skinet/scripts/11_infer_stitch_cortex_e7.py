#!/usr/bin/env python
"""
E7 Stage 11 (CORTEX) - inference + stitch -> WSI-level region mask + area.

Runs the trained cortex Mask2Former on the cortex COCO tiles, stitches the
per-tile predictions back to the low-magnification source frame as SEMANTIC
region masks (cortex / capsule / medullaire), and reports area.

Because the cortex tiler did RESIZE-THEN-TILE (each tile carries resize_ratio
and the original WSI dims), area at native WSI resolution is recovered exactly
by  area_native = area_lowmag / ratio**2  -- the giant native mask is never
materialized. The cortex area is the morphometry DENOMINATOR (density per mm^2
of cortex, as in Marechal et al.); the cortex mask is also the ROI that gates
MAIN tiling at deployment.

Reuses the cortex COCO tiles (data/coco/.../cortex) so coords/ratio are taken
from the stored tile metadata -- no raw-WSI re-tiling needed here.

Weights: OUT_CORTEX/checkpoints/model_best.pth   (3 classes)
Output : OUT_CORTEX/cortex_masks/{split}/{source}.png  (low-mag label map)
         OUT_CORTEX/cortex_masks/{split}/{source}.json (areas + dims + ratio)
         OUT_CORTEX/cortex_masks/{split}/11_cortex_summary.json
"""
import argparse, json
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

import e7_config as C

# label-map priority when two region classes overlap a pixel (model error):
# cortex dominates (it is the morphometry ROI), then medullaire, then capsule.
LABEL_PRIORITY = ["cortex", "medullaire", "capsule"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def build_inference_cfg(weights: str, test_size: int):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(C.BASE_CONFIG))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(C.CORTEX_CLASSES)
    cfg.MODEL.WEIGHTS = weights
    cfg.INPUT.MIN_SIZE_TEST = test_size
    cfg.INPUT.MAX_SIZE_TEST = test_size
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def paste_union(canvas, tile_mask, x0, y0):
    """Union a tile-local mask into the low-mag canvas at (x0, y0), clipped."""
    H, W = canvas.shape
    th, tw = tile_mask.shape
    vh, vw = min(th, H - y0), min(tw, W - x0)
    if vh <= 0 or vw <= 0:
        return
    np.maximum(canvas[y0:y0 + vh, x0:x0 + vw], tile_mask[:vh, :vw],
               out=canvas[y0:y0 + vh, x0:x0 + vw])


def remove_small(mask, min_area):
    """Drop connected components smaller than min_area (pixels)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def main():
    ap = argparse.ArgumentParser("E7 Stage 11 cortex inference + stitch")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--weights",
                    default=str(C.OUT_CORTEX / "checkpoints" / "model_best.pth"))
    ap.add_argument("--coco-dir", default=str(C.COCO_CORTEX_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--test-size", type=int, default=C.TRAIN["test_size"])
    ap.add_argument("--score-thr", type=float, default=0.50,
                    help="region instances below this score are dropped")
    ap.add_argument("--tissue-thr", type=int, default=220,
                    help="grayscale < thr = tissue; region masks are intersected "
                         "with this foreground (removes background/hole/pad bleed). 0=off")
    ap.add_argument("--min-area-lowmag", type=int, default=0,
                    help="drop region components smaller than this (low-mag px). 0=off")
    ap.add_argument("--low-cortex-frac", type=float, default=0.02,
                    help="flag sources whose cortex covers < this frac of WSI")
    ap.add_argument("--mpp", type=float, default=0.0,
                    help="microns/pixel at NATIVE WSI res (0=report px only)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N sources")
    args = ap.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_CORTEX / "cortex_masks" / args.split)
    ensure_dir(out_dir)

    ann_path = coco_dir / "annotations" / f"instances_{args.split}.json"
    coco = json.load(open(ann_path))
    by_source = defaultdict(list)
    for im in coco["images"]:
        by_source[im["source_key"]].append(im)
    sources = sorted(by_source)
    if args.limit:
        sources = sources[: args.limit]

    print("=" * 74)
    print("E7 Stage 11 - cortex inference + stitch")
    print(f"  split    : {args.split}   sources {len(sources)}")
    print(f"  weights  : {args.weights}")
    print(f"  classes  : {C.CORTEX_CLASSES}")
    print(f"  score_thr: {args.score_thr}   mpp {args.mpp or 'n/a'}")
    print("=" * 74)

    cfg = build_inference_cfg(args.weights, args.test_size)
    predictor = DefaultPredictor(cfg)

    summary = {"split": args.split, "sources": [],
               "class_area_native_px": defaultdict(float)}

    for si, src in enumerate(sources, 1):
        tiles = by_source[src]
        t0 = tiles[0]
        ow, oh = int(t0["orig_width"]), int(t0["orig_height"])
        ratio = float(t0["resize_ratio"])
        W_lm, H_lm = round(ow * ratio), round(oh * ratio)
        canvases = {c: np.zeros((H_lm, W_lm), np.uint8) for c in C.CORTEX_CLASSES}

        n_tiles = 0
        tissue = np.zeros((H_lm, W_lm), np.uint8)
        for im in tiles:
            tile = cv2.imread(str(coco_dir / im["file_name"]))
            if tile is None:
                continue
            n_tiles += 1
            tx, ty = int(im["tile_x"]), int(im["tile_y"])
            if args.tissue_thr > 0:
                gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                paste_union(tissue, (gray < args.tissue_thr).astype(np.uint8), tx, ty)
            out = predictor(tile)["instances"].to("cpu")
            if len(out) == 0:
                continue
            masks = out.pred_masks.numpy()
            scores = out.scores.numpy()
            classes = out.pred_classes.numpy()
            for i in range(len(out)):
                if scores[i] < args.score_thr or masks[i].sum() <= 0:
                    continue
                cname = C.CORTEX_CLASSES[int(classes[i])]
                paste_union(canvases[cname], masks[i].astype(np.uint8), tx, ty)

        # intersect region masks with tissue foreground (drop background/hole/pad)
        if args.tissue_thr > 0:
            for c in C.CORTEX_CLASSES:
                canvases[c] &= tissue
        if args.min_area_lowmag > 0:
            for c in C.CORTEX_CLASSES:
                canvases[c] = remove_small(canvases[c], args.min_area_lowmag)

        # per-class areas (independent unions; anatomically near-exclusive)
        inv_r2 = 1.0 / (ratio * ratio)
        areas = OrderedDict()
        for c in C.CORTEX_CLASSES:
            a_lm = int(canvases[c].sum())
            a_native = a_lm * inv_r2
            entry = {"area_lowmag_px": a_lm,
                     "area_native_px": a_native,
                     "frac_of_wsi": a_native / float(ow * oh)}
            if args.mpp > 0:
                entry["area_mm2"] = a_native * (args.mpp ** 2) / 1e6
            areas[c] = entry
            summary["class_area_native_px"][c] += a_native

        # compact label map (priority-resolved) at low-mag, for overlay/gating
        label = np.zeros((H_lm, W_lm), np.uint8)
        for c in reversed(LABEL_PRIORITY):           # lowest priority first
            label[canvases[c] > 0] = C.CORTEX_ID[c]  # higher priority overwrites
        cv2.imwrite(str(out_dir / f"{src}.png"), label)

        rec = OrderedDict(source_key=src, orig_width=ow, orig_height=oh,
                          resize_ratio=ratio, lowmag_w=W_lm, lowmag_h=H_lm,
                          num_tiles=n_tiles, mpp=args.mpp, areas=areas)
        json.dump(rec, open(out_dir / f"{src}.json", "w"), indent=2)

        cx = areas["cortex"]["frac_of_wsi"]
        flagged = cx < args.low_cortex_frac
        summary["sources"].append(OrderedDict(
            source_key=src, num_tiles=n_tiles,
            cortex_frac_wsi=cx,
            cortex_area_native_px=areas["cortex"]["area_native_px"],
            low_cortex=flagged))
        warn = "  <-- LOW CORTEX, review" if flagged else ""
        print(f"  [{si}/{len(sources)}] {src}: tiles {n_tiles}  "
              f"cortex {cx:.1%} WSI  "
              f"(cap {areas['capsule']['frac_of_wsi']:.1%} "
              f"med {areas['medullaire']['frac_of_wsi']:.1%}){warn}")

    flagged_srcs = [s["source_key"] for s in summary["sources"] if s["low_cortex"]]
    summary["low_cortex_sources"] = flagged_srcs
    summary["class_area_native_px"] = dict(summary["class_area_native_px"])
    json.dump(summary, open(out_dir / "11_cortex_summary.json", "w"), indent=2)
    print("-" * 74)
    print(f"Saved {len(sources)} cortex masks + areas -> {out_dir}")
    if flagged_srcs:
        print(f"LOW-CORTEX sources to review ({len(flagged_srcs)}): {', '.join(flagged_srcs)}")
    print("NOTE: area_native = area_lowmag / ratio^2 (exact). cortex area is the")
    print("morphometry denominator; the label map is the ROI for MAIN gating.")


if __name__ == "__main__":
    main()