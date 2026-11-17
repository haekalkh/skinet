"""
Stage 06 - map tile predictions into whole-ROI coordinates.

Coordinate reconstruction and nothing else. No fusion, no filtering, no label
arbitration: every instance stage 05 produced comes out the other side, just
placed where it belongs on the full image.

Because tiles overlap, one physical object appears here several times over. That
duplication is expected and is exactly what stage 07 collapses. Keeping the two
apart matters — if stitching also made decisions, there would be no way to tell
a stitching mistake from a post-processing one.

Outputs (into --out-dir):
  stitched/<source_key>.json    instances with whole-ROI RLE and bbox
  s06_summary.json
  run_meta.json

Usage:
  python s06_stitch.py --run baseline_seed42 --split test
"""

import argparse
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils

import config as C


def rle_in(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def rle_out(rle):
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("ascii")}


def place(tile_rle, tile_x, tile_y, tile_size, H, W):
    """Express a tile-local mask as an RLE over the whole image.

    The obvious implementation allocates an H x W array per instance and pastes
    the tile into it. For a region of 3840 x 2176 that is 8 MB and unremarkable;
    for a whole slide of 30720 x 17400 it is 534 MB, once per object, which is
    why whole-slide inference could not use this stage at all.

    The encoded RLE is small either way -- what was large was only the scratch
    array used to build it. So the run lengths are constructed directly instead:
    a column-major sweep over the object's bounding box, padded with the zero
    runs that account for everything outside it. Nothing wider than the bounding
    box is ever allocated, and the result is identical to the encoding of the
    full canvas rather than an approximation of it.
    """
    tile_mask = maskUtils.decode(rle_in(tile_rle))
    vh = min(tile_size, H - tile_y)
    vw = min(tile_size, W - tile_x)
    if vh <= 0 or vw <= 0:
        return None
    tile_mask = tile_mask[:vh, :vw]

    ys, xs = np.nonzero(tile_mask)
    if len(xs) == 0:
        return None

    by0, by1 = int(ys.min()), int(ys.max()) + 1
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    gx0, gy0 = tile_x + bx0, tile_y + by0
    wb, hb = bx1 - bx0, by1 - by0

    # a band as tall as the image but only as wide as the object
    band = np.zeros((H, wb), dtype=np.uint8)
    band[gy0:gy0 + hb, :] = tile_mask[by0:by1, bx0:bx1]
    flat = band.flatten(order="F")

    cuts = np.flatnonzero(np.diff(flat)) + 1
    edges = np.concatenate(([0], cuts, [flat.size]))
    lengths = np.diff(edges).tolist()
    values = flat[edges[:-1]].tolist()

    lead = gx0 * H
    tail = (W - gx0 - wb) * H

    counts = []
    cur_val, cur_len = 0, lead
    for v, n in zip(values, lengths):
        if v == cur_val:
            cur_len += n
        else:
            counts.append(cur_len)
            cur_val, cur_len = v, n
    if cur_val == 0:
        cur_len += tail
        counts.append(cur_len)
    else:
        counts.append(cur_len)
        counts.append(tail)

    return maskUtils.frPyObjects({"counts": counts, "size": [H, W]}, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s05_infer" / args.run / args.split / "preds")
    if args.out_dir is None:
        args.out_dir = str(work / "s06_stitch" / args.run / args.split)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    st_dir = out_dir / "stitched"
    st_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit("no stage 05 predictions in %s" % in_dir)

    print("=" * 70)
    print("stage 06 - stitch | run %s  split %s  sources %d"
          % (args.run, args.split, len(files)))
    print("=" * 70)

    s = OrderedDict(n_sources=0, n_instances_in=0, n_instances_out=0,
                    n_dropped_offcanvas=0)
    per_class = defaultdict(int)

    for i, pf in enumerate(files, 1):
        with open(str(pf)) as f:
            rec = json.load(f)
        H, W, ts = rec["height"], rec["width"], rec["tile_size"]

        out = []
        for inst in rec["instances"]:
            g = place(inst["mask_rle"], inst["tile_x"], inst["tile_y"], ts, H, W)
            if g is None:
                s["n_dropped_offcanvas"] += 1
                continue
            x, y, w, h = [float(v) for v in maskUtils.toBbox(g)]
            out.append({
                "class_name": inst["class_name"],
                "category_id": int(inst["category_id"]),
                "score": float(inst["score"]),
                "whole_view": bool(inst.get("whole_view", False)),
                "bbox": [x, y, w, h],
                "area": float(maskUtils.area(g)),
                "mask_rle": rle_out(g),
            })
            per_class[inst["class_name"]] += 1

        s["n_sources"] += 1
        s["n_instances_in"] += len(rec["instances"])
        s["n_instances_out"] += len(out)

        with open(str(st_dir / ("%s.json" % rec["source_key"])), "w") as f:
            json.dump(OrderedDict(source_key=rec["source_key"],
                                  height=H, width=W,
                                  n_instances=len(out), instances=out), f)
        print("[%d/%d] %s: %d instances" % (i, len(files), rec["source_key"], len(out)))

    s["instances_per_class"] = OrderedDict(
        (c, int(per_class.get(c, 0))) for c in C.STRUCTURE_CLASSES)
    with open(str(out_dir / "s06_summary.json"), "w") as f:
        json.dump(s, f, indent=2)

    print("\n" + "=" * 70)
    print("in %d  out %d  dropped off-canvas %d"
          % (s["n_instances_in"], s["n_instances_out"], s["n_dropped_offcanvas"]))
    print("-" * 70)
    print("These counts still hold every overlap duplicate; stage 07 collapses them.")
    for c in C.STRUCTURE_CLASSES:
        print("   %-20s%7d" % (c, per_class.get(c, 0)))
    print("=" * 70)
    C.write_run_meta(out_dir, "s06_stitch", args, extra=s)


if __name__ == "__main__":
    main()