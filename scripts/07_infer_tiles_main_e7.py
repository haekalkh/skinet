"""
E7 Stage 7 - Tile-level inference (MAIN).

Config-driven (imports e7_config). Runs the trained M2F MAIN model over the FULL
source images, tiled exhaustively (every tile, including empty ones) at the same
TILE_SIZE / stride used for training. This differs from the COCO test set, which
only contains tiles that had GT annotations.

Per detected instance it stores: category_id, score, tile offset (tile_x/tile_y),
local bbox, and the mask as a tile-local RLE. Stage 8 (stitching) maps these to
global WSI coordinates; fusion/filtering happen in later stages. Nothing is fused
or filtered here beyond a raw score floor (kept low on purpose so the downstream
union/priority/orphan rules have the candidates they need).

Input : 02_e7_source_split.csv  (uses --split, default test)
Weights: OUT_MAIN/checkpoints/model_best.pth
Output: OUT_MAIN/inference/{split}/preds/{source_key}.json  + manifest
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

import e7_config as C


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def compute_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def make_tile_image(image, x0, y0, tile_size):
    h, w = image.shape[:2]
    x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
    tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    tile[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
    return tile


def build_inference_cfg(weights: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(C.BASE_CONFIG))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(C.MAIN_CLASSES)
    cfg.MODEL.WEIGHTS = weights
    cfg.INPUT.MIN_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def encode_mask(mask_bool):
    rle = mask_util.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def infer_source(predictor, image, tile_size, stride, score_thr,
                 progress_every=25, src=""):
    """Tile the full image, run M2F per tile, collect raw instances.

    progress_every: print a progress line every N tiles (with flush=True so the
    SGE batch log updates live). Set 0 to silence. `src` is just a log label.
    Total tile count is known up front (grid = len(xs)*len(ys)), so the progress
    line can show a real percentage.
    """
    h, w = image.shape[:2]
    ys = compute_starts(h, tile_size, stride)
    xs = compute_starts(w, tile_size, stride)
    total = len(ys) * len(xs)
    tag = f"[{src}] " if src else ""
    print(f"    {tag}infer: {total} tiles ({len(xs)}x{len(ys)} grid)", flush=True)

    instances_out = []
    n_tiles = 0
    for y0 in ys:
        for x0 in xs:
            tile = make_tile_image(image, x0, y0, tile_size)
            out = predictor(tile)["instances"].to("cpu")
            n_tiles += 1
            if progress_every and (n_tiles % progress_every == 0 or n_tiles == total):
                print(f"    {tag}tile {n_tiles}/{total} "
                      f"({100 * n_tiles / total:.0f}%)  "
                      f"instances so far: {len(instances_out)}", flush=True)
            if len(out) == 0:
                continue
            masks = out.pred_masks.numpy()
            scores = out.scores.numpy()
            classes = out.pred_classes.numpy()
            boxes = out.pred_boxes.tensor.numpy()
            for i in range(len(out)):
                if scores[i] < score_thr:
                    continue
                m = masks[i]
                if m.sum() <= 0:
                    continue
                x1, y1, x2, y2 = boxes[i]
                instances_out.append({
                    "category_id": int(classes[i]) + 1,  # 1-indexed (COCO ids)
                    "class_name": C.MAIN_CLASSES[int(classes[i])],
                    "score": float(scores[i]),
                    "tile_x": int(x0), "tile_y": int(y0),
                    "bbox_local": [float(x1), float(y1),
                                   float(x2 - x1), float(y2 - y1)],
                    "mask_rle": encode_mask(m),  # tile-local, size [tile,tile]
                })
    return instances_out, n_tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--weights",
                    default=str(C.OUT_MAIN / "checkpoints" / "model_best.pth"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--score-thr", type=float, default=0.30,
                    help="Raw score floor at tile level (keep low; downstream "
                         "fusion/priority/orphan do the real filtering).")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print a progress line every N tiles (0 to silence).")
    ap.add_argument("--limit", type=int, default=0, help="Debug: first N sources.")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}. Training must have produced "
            f"model_best.pth (first appears after the first eval at iter "
            f"{C.TRAIN['eval_period']}).")

    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "inference" / args.split)
    preds_dir = out_dir / "preds"
    ensure_dir(preds_dir)
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))

    print("=" * 80)
    print(f"E7 Stage 7 - Tile inference | split={args.split}")
    print(f"Weights : {weights}")
    print(f"Tile {args.tile_size}  stride {stride}  score_thr {args.score_thr}")
    print("=" * 80)

    cfg = build_inference_cfg(str(weights))
    predictor = DefaultPredictor(cfg)

    df = pd.read_csv(args.split_csv)
    df = df[(df["split"] == args.split) & (df["has_image"] == True)
            & (df["parse_ok"] == True) & (df["has_any_main"] == True)].copy()
    df = df.sort_values("source_key").reset_index(drop=True)
    if args.limit > 0:
        df = df.head(args.limit)

    manifest = OrderedDict(split=args.split, weights=str(weights),
                           tile_size=args.tile_size, stride=stride,
                           score_thr=args.score_thr, num_classes=len(C.MAIN_CLASSES),
                           sources=[])

    for idx, r in df.iterrows():
        src = r["source_key"]
        image = cv2.imread(str(r["image_path"]))
        if image is None:
            print(f"[WARN] cannot read: {r['image_path']}")
            continue
        oh, ow = image.shape[:2]
        print(f"[{idx + 1}/{len(df)}] {src} ({ow}x{oh})", flush=True)
        instances, n_tiles = infer_source(predictor, image, args.tile_size,
                                          stride, args.score_thr,
                                          progress_every=args.progress_every,
                                          src=src)
        rec = OrderedDict(source_key=src, orig_width=int(ow), orig_height=int(oh),
                          tile_size=args.tile_size, stride=stride,
                          num_tiles=n_tiles, num_instances=len(instances),
                          instances=instances)
        with open(preds_dir / f"{src}.json", "w") as f:
            json.dump(rec, f)
        manifest["sources"].append({"source_key": src, "num_tiles": n_tiles,
                                    "num_instances": len(instances),
                                    "orig_hw": [int(oh), int(ow)]})
        print(f"[{idx + 1}/{len(df)}] {src}: {n_tiles} tiles -> "
              f"{len(instances)} raw instances", flush=True)

    manifest["num_sources"] = len(manifest["sources"])
    manifest["total_raw_instances"] = int(sum(s["num_instances"] for s in manifest["sources"]))
    with open(out_dir / "07_inference_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Sources: {manifest['num_sources']}  "
          f"total raw instances: {manifest['total_raw_instances']}")
    print(f"Saved per-source predictions to {preds_dir}")
    print(f"Manifest: {out_dir / '07_inference_manifest.json'}")


if __name__ == "__main__":
    main()