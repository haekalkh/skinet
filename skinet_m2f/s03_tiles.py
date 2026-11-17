"""
Stage 03 - cut the slides into tiles and write COCO annotations.

For every slide in the split, walk a regular overlapping grid, and for each
tile decide (in this order):

  1. cortex gate   -- drop the tile if too little of it lies inside the cortex
                      polygons. The legacy main model did exactly this. Slides
                      with no cortex polygon at all keep every tile, and are
                      counted separately so the exception stays visible.
  2. retention     -- keep an object in the tile only if enough of the WHOLE
                      object is visible there.
  3. empty tiles   -- a tile left with no object is written or skipped
                      according to KEEP_EMPTY_TILES, per split.

Two things this does that the legacy build did not:

  * every annotation records visible_ratio, the fraction of the whole object
    present in that tile, so truncation can be measured later instead of
    guessed at;
  * the whole-object area is rasterised the same way as the visible area.
    Comparing a polygon area against a pixel count made the ratio exceed 1 for
    thin structures, so the retention rule never bit for intima and media.

Outputs (into --out-dir):
  <split>/images/*.jpg
  annotations/instances_<split>.json
  s03_summary.json
  run_meta.json

Usage:
  python s03_tiles.py
  python s03_tiles.py --splits val --limit 3          # quick look first
  python s03_tiles.py --min-tile-cortex 0             # gating off
"""

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np

import config as C

CLASS_ID = dict((c, i + 1) for i, c in enumerate(C.STRUCTURE_CLASSES))


# ------------------------------------------------------------------ input --
def parse_xml(xml_path):
    """Return (structures, cortex_polys). structures: list of (label, points)."""
    structures, cortex = [], []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception as e:
        print("[parse failed] {}: {}".format(xml_path, e))
        return structures, cortex

    for ann in root.iter():
        if ann.tag.lower().split("}")[-1] != "annotation":
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = C.canonical_label(raw)
        pts = []
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.float64)
        if label in CLASS_ID:
            structures.append((label, arr))
        elif label == "cortex":
            cortex.append(arr)
    return structures, cortex


# --------------------------------------------------------------- geometry --
def grid_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def raster_area(points):
    """Pixel count of the filled polygon, in its own bounding box.

    Same rasterisation as the per-tile mask, so the two are comparable.
    """
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    w, h = max(int(x1 - x0) + 1, 1), max(int(y1 - y0) + 1, 1)
    if w * h > 80000000:                      # absurd polygon, fall back
        return float(abs(cv2.contourArea(points.astype(np.float32))))
    m = np.zeros((h, w), dtype=np.uint8)
    shifted = points - np.array([x0, y0], dtype=np.float64)
    cv2.fillPoly(m, [shifted.round().astype(np.int32)], 1)
    return float(m.sum())


def tile_mask(points, x0, y0, tile):
    m = np.zeros((tile, tile), dtype=np.uint8)
    shifted = points - np.array([x0, y0], dtype=np.float64)
    cv2.fillPoly(m, [shifted.round().astype(np.int32)], 1)
    return m


def cut_tile(image, x0, y0, tile):
    h, w = image.shape[:2]
    x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
    out = np.full((tile, tile, 3), 255, dtype=np.uint8)
    out[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
    return out


def cortex_coverage(cortex_polys, x0, y0, tile):
    m = np.zeros((tile, tile), dtype=np.uint8)
    for p in cortex_polys:
        shifted = p - np.array([x0, y0], dtype=np.float64)
        cv2.fillPoly(m, [shifted.round().astype(np.int32)], 1)
    return float(m.sum()) / float(tile * tile)


def mask_to_coco(mask):
    res = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
    contours = res[1] if len(res) == 3 else res[0]
    segm = []
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c.astype(np.float32)) <= 0:
            continue
        segm.append(c.reshape(-1).astype(float).tolist())
    if not segm:
        return None
    ys, xs = np.where(mask > 0)
    x, y = int(xs.min()), int(ys.min())
    return {"segmentation": segm,
            "bbox": [x, y, int(xs.max() - x + 1), int(ys.max() - y + 1)],
            "area": float(mask.sum())}


# ------------------------------------------------------------------ build --
def build_split(split, rows, out_dir, tile, stride, min_ratio, min_area_px,
                min_cortex, keep_empty, limit):
    img_out = out_dir / split / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": CLASS_ID[c], "name": c}
                           for c in C.STRUCTURE_CLASSES]}
    image_id, ann_id = 1, 1

    s = OrderedDict(n_slides=0, n_slides_no_cortex=0, n_tiles_in_grid=0,
                    n_tiles_written=0, n_tiles_with_objects=0,
                    n_empty_written=0, n_empty_skipped=0,
                    n_gated_out_cortex=0, n_objects_written=0,
                    n_objects_below_ratio=0, n_objects_below_area=0)
    per_class = defaultdict(int)
    ratio_hist = defaultdict(int)

    if limit > 0:
        rows = rows[:limit]

    for idx, r in enumerate(rows, 1):
        image = cv2.imread(r["image_path"])
        if image is None:
            print("[skip] cannot read {}".format(r["image_path"]))
            continue
        structures, cortex = parse_xml(Path(r["xml_path"]))
        if not structures:
            continue
        s["n_slides"] += 1

        gate_on = min_cortex > 0.0 and len(cortex) > 0
        if min_cortex > 0.0 and not cortex:
            s["n_slides_no_cortex"] += 1

        oh, ow = image.shape[:2]
        # Precompute each object's bbox and its full rasterised area once.
        prepared = []
        for label, pts in structures:
            bx0, by0 = pts.min(axis=0)
            bx1, by1 = pts.max(axis=0)
            full = raster_area(pts)
            if full <= 0:
                continue
            prepared.append((label, pts, bx0, by0, bx1, by1, full))

        ys = grid_starts(oh, tile, stride)
        xs = grid_starts(ow, tile, stride)
        s["n_tiles_in_grid"] += len(xs) * len(ys)
        print("[{}/{}] {} {}x{} -> {} tiles".format(
            idx, len(rows), r["source_key"], ow, oh, len(xs) * len(ys)))
        sys.stdout.flush()

        for y0 in ys:
            for x0 in xs:
                if gate_on:
                    if cortex_coverage(cortex, x0, y0, tile) < min_cortex:
                        s["n_gated_out_cortex"] += 1
                        continue

                objs = []
                for label, pts, bx0, by0, bx1, by1, full in prepared:
                    if bx1 < x0 or bx0 > x0 + tile or by1 < y0 or by0 > y0 + tile:
                        continue
                    m = tile_mask(pts, x0, y0, tile)
                    vis = float(m.sum())
                    if vis <= 0:
                        continue
                    if vis < min_area_px:
                        s["n_objects_below_area"] += 1
                        continue
                    ratio = vis / full
                    if ratio < min_ratio:
                        s["n_objects_below_ratio"] += 1
                        continue
                    obj = mask_to_coco(m)
                    if obj is not None:
                        objs.append((label, obj, min(ratio, 1.0)))

                if not objs and not keep_empty:
                    s["n_empty_skipped"] += 1
                    continue

                fname = "{}_x{}_y{}.jpg".format(r["source_key"], x0, y0)
                cv2.imwrite(str(img_out / fname), cut_tile(image, x0, y0, tile),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                coco["images"].append({
                    "id": image_id,
                    "file_name": "{}/images/{}".format(split, fname),
                    "width": tile, "height": tile,
                    "source_key": r["source_key"],
                    "tile_x": int(x0), "tile_y": int(y0), "tile_size": tile,
                    "orig_width": int(ow), "orig_height": int(oh),
                })
                for label, obj, ratio in objs:
                    coco["annotations"].append({
                        "id": ann_id, "image_id": image_id,
                        "category_id": CLASS_ID[label],
                        "segmentation": obj["segmentation"],
                        "bbox": obj["bbox"], "area": obj["area"],
                        "iscrowd": 0,
                        "visible_ratio": round(float(ratio), 4),
                    })
                    per_class[label] += 1
                    ratio_hist[min(int(ratio * 10), 9)] += 1
                    ann_id += 1
                    s["n_objects_written"] += 1

                image_id += 1
                s["n_tiles_written"] += 1
                if objs:
                    s["n_tiles_with_objects"] += 1
                else:
                    s["n_empty_written"] += 1

    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    with open(str(ann_dir / "instances_{}.json".format(split)), "w") as f:
        json.dump(coco, f)

    s["objects_per_class"] = OrderedDict(
        (c, int(per_class.get(c, 0))) for c in C.STRUCTURE_CLASSES)
    s["visible_ratio_histogram"] = OrderedDict(
        ("{:.1f}-{:.1f}".format(i / 10.0, (i + 1) / 10.0),
         int(ratio_hist.get(i, 0))) for i in range(10))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv",
                    default=str(Path(C.WORK_DIR) / "s02_split" / "s02_split.csv"))
    ap.add_argument("--out-dir", default=str(Path(C.WORK_DIR) / "s03_tiles"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--min-visible-ratio", type=float, default=C.MIN_VISIBLE_RATIO)
    ap.add_argument("--min-visible-area", type=float, default=C.MIN_VISIBLE_AREA_PX)
    ap.add_argument("--min-tile-cortex", type=float, default=C.MIN_TILE_CORTEX)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N slides of each split")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))

    with open(args.split_csv) as f:
        all_rows = list(csv.DictReader(f))
    if not all_rows:
        sys.exit("split csv is empty: {}".format(args.split_csv))

    print("=" * 70)
    print("stage 03 - tiles")
    print("tile {}  stride {}  min_visible_ratio {}  min_tile_cortex {}".format(
        args.tile_size, stride, args.min_visible_ratio, args.min_tile_cortex))
    print("keep empty tiles: {}".format(C.KEEP_EMPTY_TILES))
    print("=" * 70)

    overall = OrderedDict()
    for split in args.splits:
        rows = [r for r in all_rows if r["split"] == split]
        print("\n--- {} : {} slides ---".format(split, len(rows)))
        overall[split] = build_split(
            split, rows, out_dir, args.tile_size, stride,
            args.min_visible_ratio, args.min_visible_area,
            args.min_tile_cortex,
            C.KEEP_EMPTY_TILES.get(split, False), args.limit)

    with open(str(out_dir / "s03_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print("\n" + "=" * 70)
    print("{:<10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "split", "grid", "written", "w/ obj", "empty", "gated"))
    print("-" * 70)
    for split in args.splits:
        st = overall[split]
        print("{:<10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
            split, st["n_tiles_in_grid"], st["n_tiles_written"],
            st["n_tiles_with_objects"], st["n_empty_written"],
            st["n_gated_out_cortex"]))
    print("-" * 70)
    print("{:<22}{:>9}{:>9}{:>9}".format("class", "train", "val", "test"))
    for c in C.STRUCTURE_CLASSES:
        vals = [overall[s]["objects_per_class"][c] if s in overall else 0
                for s in ["train", "val", "test"]]
        print("{:<22}{:>9}{:>9}{:>9}".format(c, vals[0], vals[1], vals[2]))
    print("=" * 70)
    C.write_run_meta(out_dir, "s03_tiles", args, extra=overall)


if __name__ == "__main__":
    main()