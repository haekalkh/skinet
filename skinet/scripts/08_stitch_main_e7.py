"""
E7 Stage 8 - Stitching (fuse_results).

Config-driven (imports e7_config). Maps Stage 7 tile-level predictions into
GLOBAL WSI coordinates and concatenates them into one instance list per source.

This is the MRCNN `fuse_results` step: coordinate reconstruction ONLY. No
de-duplication, no fusion, no filtering here -- those are Stage 9 (same/cross
class fusion) and Stage 10 (priority + orphan/hierarchy). Because tiles overlap
(33%), the SAME physical object appears as several global instances here; that
redundancy is expected and is what Stage 9 collapses.

Each global instance: category_id, class_name, score, bbox_global [x,y,w,h],
mask_rle (RLE over the full HxW image). Full-image RLE lets Stage 9 compute
overlaps directly in RLE space (pycocotools) without dense decoding.

Input : OUT_MAIN/inference/{split}/preds/{source}.json   (Stage 7)
Output: OUT_MAIN/postproc/{split}/08_stitched/{source}.json + summary

----------------------------------------------------------------------------
PERFORMANCE NOTE (stitch_source)
The previous implementation allocated a fresh full-WSI canvas (H x W, ~535 MP)
AND made a Fortran-order copy of it FOR EVERY instance. With thousands of raw
instances per slide that was billions of pixels of pure allocation/zeroing,
dominating the wall time (~2.8 h on ML0029C2).

This version allocates ONE reusable Fortran-order buffer, writes only each
instance's small ROI into it, encodes, then clears ONLY that ROI. The encoded
full-image RLE and the derived bbox are byte-for-byte identical to the old code
(same logical 2-D array is encoded), so Stage 9/10/12 outputs are unchanged.
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
from pycocotools import mask as mask_util

import e7_config as C


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rle_from_str(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def rle_to_str(rle):
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("ascii")}


def place_tile_mask_global(tile_rle_str, tile_x, tile_y, tile_size, H, W):
    """Decode a tile-local RLE and paste it onto a full HxW canvas.

    Kept for backward compatibility / external callers. The hot path in
    stitch_source no longer uses this (it reuses a single buffer instead), but
    the behaviour here is unchanged.
    """
    tile_mask = mask_util.decode(rle_from_str(tile_rle_str))  # [ts, ts]
    canvas = np.zeros((H, W), dtype=np.uint8)
    vh = min(tile_size, H - tile_y)
    vw = min(tile_size, W - tile_x)
    if vh <= 0 or vw <= 0:
        return None
    canvas[tile_y:tile_y + vh, tile_x:tile_x + vw] = tile_mask[:vh, :vw]
    if canvas.sum() == 0:
        return None
    return mask_util.encode(np.asfortranarray(canvas))


def stitch_source(rec):
    """Reconstruct global-coordinate instances for one source.

    Output is identical to the per-instance full-canvas version, but a single
    Fortran-order buffer is allocated once and reused: each instance writes only
    its ROI, gets encoded, then that ROI is cleared. This removes the two
    full-WSI (~535 MP) allocations/copies that previously ran per instance.
    """
    H, W = rec["orig_height"], rec["orig_width"]
    ts = rec["tile_size"]
    out = []

    # ONE buffer, allocated once, in Fortran order so mask_util.encode needs no
    # copy. Reused across all instances; only the written ROI is ever touched.
    canvas = np.zeros((H, W), dtype=np.uint8, order="F")

    for inst in rec["instances"]:
        tx, ty = inst["tile_x"], inst["tile_y"]
        vh = min(ts, H - ty)
        vw = min(ts, W - tx)
        if vh <= 0 or vw <= 0:
            continue

        tile_mask = mask_util.decode(rle_from_str(inst["mask_rle"]))  # [ts, ts]
        roi = tile_mask[:vh, :vw]
        if roi.sum() == 0:
            continue

        # write ROI -> encode full-image RLE -> derive bbox
        canvas[ty:ty + vh, tx:tx + vw] = roi
        g_rle = mask_util.encode(canvas)
        x, y, w, h = [float(v) for v in mask_util.toBbox(g_rle)]
        out.append({
            "category_id": int(inst["category_id"]),
            "class_name": inst["class_name"],
            "score": float(inst["score"]),
            "bbox_global": [x, y, w, h],
            "mask_rle": rle_to_str(g_rle),
        })

        # clear ONLY this instance's ROI so the buffer is clean for the next one
        canvas[ty:ty + vh, tx:tx + vw] = 0

    return out, H, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "inference" / args.split / "preds")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "postproc" / args.split / "08_stitched")
    ensure_dir(out_dir)

    pred_files = sorted(in_dir.glob("*.json"))
    if not pred_files:
        raise FileNotFoundError(f"No Stage 7 predictions in {in_dir}")

    print("=" * 80)
    print(f"E7 Stage 8 - Stitching | split={args.split}  sources={len(pred_files)}")
    print("=" * 80)

    summary = OrderedDict(split=args.split, num_sources=0,
                          per_class_global=defaultdict(int),
                          total_global_instances=0, sources=[])

    for pf in pred_files:
        with open(pf) as f:
            rec = json.load(f)
        instances, H, W = stitch_source(rec)
        per_class = defaultdict(int)
        for it in instances:
            per_class[it["class_name"]] += 1
            summary["per_class_global"][it["class_name"]] += 1
        out = OrderedDict(source_key=rec["source_key"], height=H, width=W,
                          num_global_instances=len(instances),
                          per_class=dict(per_class), instances=instances)
        with open(out_dir / f"{rec['source_key']}.json", "w") as f:
            json.dump(out, f)
        summary["sources"].append({"source_key": rec["source_key"],
                                   "num_global_instances": len(instances)})
        summary["total_global_instances"] += len(instances)
        summary["num_sources"] += 1
        print(f"  {rec['source_key']}: {len(instances)} global instances "
              f"(pre-fusion)")

    summary["per_class_global"] = {c: int(summary["per_class_global"].get(c, 0))
                                   for c in C.MAIN_CLASSES}
    with open(out_dir / "08_stitch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Total pre-fusion global instances: {summary['total_global_instances']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
