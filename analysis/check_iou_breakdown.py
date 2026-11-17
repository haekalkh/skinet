"""
For each class: is the model failing to FIND the object, or failing to DRAW it?

COCO's headline AP averages ten IoU thresholds from 0.50 to 0.95, so a class
whose masks are slightly off scores the same as one that is never detected. The
two are separated by looking at the thresholds individually.

  AP50 high, AP75 collapsed  -> the object is found, the boundary is wrong
  AP50 also low              -> the object is not found at all

This matters most for thin structures. On a ring five pixels wide, a two-pixel
boundary error already drops IoU below 0.5, while the same error is invisible on
a large blob.

Reads the predictions the trainer already wrote. No GPU, no inference.

Usage:
  python check_iou_breakdown.py --run baseline_seed42
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import config as C


def per_class(ev, k, iou_idx=None):
    """Mean precision for class k, at one IoU threshold or averaged over all."""
    p = ev.eval["precision"]                    # [T, R, K, A, M]
    s = p[:, :, k, 0, 2] if iou_idx is None else p[iou_idx, :, k, 0, 2]
    s = s[s > -1]
    return float(np.mean(s)) * 100 if s.size else float("nan")


def per_class_recall(ev, k, iou_idx=None):
    r = ev.eval["recall"]                       # [T, K, A, M]
    s = r[:, k, 0, 2] if iou_idx is None else r[iou_idx, k, 0, 2]
    s = s[s > -1]
    return float(np.mean(s)) * 100 if s.size else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    work = Path(args.work_dir)
    gt_file = work / "s03_tiles" / "annotations" / ("instances_%s.json" % args.split)
    pred_file = work / "s04_train" / args.run / "eval" / "coco_instances_results.json"
    for p in (gt_file, pred_file):
        if not p.exists():
            raise SystemExit("not found: %s" % p)

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        coco = COCO(str(gt_file))
        dt = coco.loadRes(str(pred_file))
        ev = COCOeval(coco, dt, "segm")
        ev.evaluate()
        ev.accumulate()

    cat_ids = coco.getCatIds()
    id2name = dict((c["id"], c["name"]) for c in coco.loadCats(cat_ids))
    # index of IoU 0.50 and 0.75 in the standard 0.50:0.05:0.95 sweep
    i50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.50)))
    i75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))

    rows = []
    for k, cid in enumerate(cat_ids):
        name = id2name.get(cid, str(cid))
        a50, a75, aavg = (per_class(ev, k, i50), per_class(ev, k, i75),
                          per_class(ev, k))
        r50 = per_class_recall(ev, k, i50)
        drop = (a50 - a75) / a50 * 100 if a50 and a50 == a50 and a50 > 0 else float("nan")
        rows.append((name, aavg, a50, a75, drop, r50))

    order = [c for c in C.STRUCTURE_CLASSES]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)

    print("=" * 76)
    print("run %s  split %s" % (args.run, args.split))
    print("=" * 76)
    print("%-20s%8s%8s%8s%10s%9s" %
          ("class", "AP", "AP50", "AP75", "drop", "AR50"))
    print("-" * 76)
    for name, aavg, a50, a75, drop, r50 in rows:
        print("%-20s%8.1f%8.1f%8.1f%9.0f%%%9.1f" % (name, aavg, a50, a75, drop, r50))
    print("-" * 76)
    print("AP50   how often the object is found at all (loose overlap)")
    print("AP75   how often the mask is actually accurate")
    print("drop   how much is lost between the two, as a share of AP50")
    print("AR50   recall at loose overlap, ignoring precision")
    print()
    print("A large drop with a high AP50 means the boundary is the problem.")
    print("A low AP50 means the object is genuinely not being found.")


if __name__ == "__main__":
    main()