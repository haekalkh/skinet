"""
Feed the annotations through the pipeline to find its ceiling.

Tiling, reassembly and filtering lose something no matter how good the model is.
The only way to see how much is to send a perfect input through and score what
comes out: take the ground truth, tile it exactly as stage 03 does, put the
fragments back together as stage 06 does, and hand the result to stage 07.
Whatever it scores against the same ground truth is the highest this pipeline
can reach.

Writes the s06 stitched format, so stages 07 and 08 run on it unchanged:

  python make_gt_as_pred.py --split test
  python s07_postproc.py --run GT_CEILING --split test --tag _ceiling
  python s08_eval.py     --run GT_CEILING --split test --tag _ceiling

Run stage 07 with --stop-after 1, 2, 3 ... to see which stage costs what. And
compare the per-class counts with the annotations: fewer than ground truth means
fusion is merging neighbours, more means it is failing to rejoin fragments.
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils

import config as C


def grid_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def gt_instances(xml_path, H, W):
    out = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return out
    for ann in root.iter():
        if ann.tag.lower().split("}")[-1] != "annotation":
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = C.canonical_label(raw)
        if label not in C.STRUCTURE_CLASSES:
            continue
        pts = []
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) < 3:
            continue
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(pts).round().astype(np.int32)], 1)
        if m.sum() > 0:
            out.append((label, m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--run", default="GT_CEILING")
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--min-visible-ratio", type=float, default=C.MIN_VISIBLE_RATIO)
    ap.add_argument("--min-visible-area", type=int, default=C.MIN_VISIBLE_AREA_PX)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    work = Path(args.work_dir)
    out_dir = work / "s06_stitch" / args.run / args.split
    st_dir = out_dir / "stitched"
    st_dir.mkdir(parents=True, exist_ok=True)
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))

    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]

    print("=" * 72)
    print("ground truth through the pipeline | split %s  sources %d"
          % (args.split, len(rows)))
    print("tile %d  stride %d  keep fragment >= %.0f%% of the object"
          % (args.tile_size, stride, 100 * args.min_visible_ratio))
    print("=" * 72)

    tot_gt = defaultdict(int)
    tot_frag = defaultdict(int)
    n_dropped = 0

    for i, r in enumerate(rows, 1):
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[skip] %s" % r["source_key"])
            continue
        H, W = img.shape[:2]
        objs = gt_instances(Path(r["xml_path"]), H, W)

        ys = grid_starts(H, args.tile_size, stride)
        xs = grid_starts(W, args.tile_size, stride)

        out = []
        for label, full in objs:
            tot_gt[label] += 1
            full_area = float(full.sum())
            for y0 in ys:
                for x0 in xs:
                    y1 = min(y0 + args.tile_size, H)
                    x1 = min(x0 + args.tile_size, W)
                    piece = full[y0:y1, x0:x1]
                    a = float(piece.sum())
                    if a <= 0:
                        continue
                    # the same retention rule stage 03 applies
                    if a < args.min_visible_area or a / full_area < args.min_visible_ratio:
                        n_dropped += 1
                        continue
                    canvas = np.zeros((H, W), dtype=np.uint8)
                    canvas[y0:y1, x0:x1] = piece
                    rle = maskUtils.encode(np.asfortranarray(canvas))
                    x, y, w, h = [float(v) for v in maskUtils.toBbox(rle)]
                    out.append({
                        "class_name": label,
                        "category_id": C.STRUCTURE_CLASSES.index(label) + 1,
                        "score": 1.0,
                        "whole_view": bool(a >= full_area - 0.5),
                        "bbox": [x, y, w, h],
                        "area": float(maskUtils.area(rle)),
                        "mask_rle": {"size": [int(v) for v in rle["size"]],
                                     "counts": rle["counts"].decode("ascii")},
                    })
                    tot_frag[label] += 1

        with open(str(st_dir / ("%s.json" % r["source_key"])), "w") as f:
            json.dump(OrderedDict(source_key=r["source_key"], height=H, width=W,
                                  n_instances=len(out), instances=out), f)
        print("[%d/%d] %-16s %4d objects -> %4d fragments"
              % (i, len(rows), r["source_key"], len(objs), len(out)))

    print("\n" + "=" * 72)
    print("%-20s%10s%12s%10s" % ("class", "objects", "fragments", "per obj"))
    for c in C.STRUCTURE_CLASSES:
        g, fr = tot_gt.get(c, 0), tot_frag.get(c, 0)
        if g:
            print("%-20s%10d%12d%10.2f" % (c, g, fr, fr / float(g)))
    print("-" * 72)
    print("fragments below the retention rule, dropped: %d" % n_dropped)
    print()
    print("Stage 07 has to put those fragments back together. Whatever stage 08")
    print("then scores is this pipeline's ceiling — no model can beat it.")
    print("=" * 72)
    C.write_run_meta(out_dir, "make_gt_as_pred", args,
                     extra={"objects": dict(tot_gt), "fragments": dict(tot_frag),
                            "dropped_fragments": n_dropped})


if __name__ == "__main__":
    main()