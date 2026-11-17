"""
Two questions, both raised by looking at the rendered masks:

1. Is the error DIRECTIONAL? The claim from the pictures is that predicted
   intima comes out too large and predicted media too small, both drifting
   towards the same line between them. Measured as predicted area divided by
   annotated area, per matched instance. Above 1 means too large.

2. How often is the nesting broken? intima should sit inside media, media
   inside vaisseau. The ground truth obeys this by construction, so the same
   figure on the ground truth is the sanity check: it should come out near 1.

If both hold, enforcing the nesting on the mask output has a measured problem
to fix and a baseline number to beat. If the ratios scatter around 1, the
reading of the pictures was wrong and the idea has no basis.

Usage:
  python check_nesting.py --run baseline_seed42
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

import config as C

NESTED = [("intima", "media"), ("media", "vaisseau"), ("intima", "vaisseau")]
RATIO_CLASSES = ["intima", "media", "vaisseau", "veine", "nsg"]


def to_rle(seg, h, w):
    if isinstance(seg, list):
        return maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):
        return maskUtils.frPyObjects(seg, h, w)
    return seg


def contained(child_rle, parent_rles):
    """Fraction of the child that lies inside the union of the parents."""
    a = maskUtils.area(child_rle)
    if a <= 0:
        return None
    if not parent_rles:
        return 0.0
    union = maskUtils.merge(parent_rles, intersect=False)
    inter = maskUtils.merge([child_rle, union], intersect=True)
    return float(maskUtils.area(inter)) / float(a)


def summarise(name, vals, fmt="%9.2f"):
    if not vals:
        return "%-24s%7d" % (name, 0)
    a = np.array(vals, dtype=float)
    line = "%-24s%7d" % (name, len(a))
    for v in (np.median(a), np.percentile(a, 25), np.percentile(a, 75), a.mean()):
        line += fmt % v
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--split", default="val")
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.3,
                    help="minimum overlap for a prediction to count as matching")
    args = ap.parse_args()

    work = Path(args.work_dir)
    gt_file = work / "s03_tiles" / "annotations" / ("instances_%s.json" % args.split)
    pred_file = work / "s04_train" / args.run / "eval" / "coco_instances_results.json"
    for p in (gt_file, pred_file):
        if not p.exists():
            raise SystemExit("not found: %s" % p)

    coco = COCO(str(gt_file))
    with open(str(pred_file)) as f:
        preds = [p for p in json.load(f) if p.get("score", 1.0) >= args.score]

    name2id = dict((c, i + 1) for i, c in enumerate(C.STRUCTURE_CLASSES))
    id2name = dict((v, k) for k, v in name2id.items())

    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)

    ratios = defaultdict(list)
    cont_pred = defaultdict(list)
    cont_gt = defaultdict(list)

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        pl = by_img.get(img_id, [])
        if not anns and not pl:
            continue

        g_by_class = defaultdict(list)
        for a in anns:
            g_by_class[id2name.get(a["category_id"])].append(to_rle(a["segmentation"], h, w))
        p_by_class = defaultdict(list)
        for p in pl:
            p_by_class[id2name.get(p["category_id"])].append(p["segmentation"])

        # 1. predicted area over annotated area, for matched pairs
        for cname in RATIO_CLASSES:
            g_list, p_list = g_by_class.get(cname, []), p_by_class.get(cname, [])
            if not g_list or not p_list:
                continue
            ious = np.asarray(maskUtils.iou(p_list, g_list, [0] * len(g_list)))
            ious = ious.reshape(len(p_list), len(g_list))
            for j, g in enumerate(g_list):
                i = int(np.argmax(ious[:, j]))
                if ious[i, j] < args.iou:
                    continue
                ga = float(maskUtils.area(g))
                pa = float(maskUtils.area(p_list[i]))
                if ga > 0:
                    ratios[cname].append(pa / ga)

        # 2. nesting, on predictions and on the ground truth
        for child, parent in NESTED:
            for c_rle in p_by_class.get(child, []):
                v = contained(c_rle, p_by_class.get(parent, []))
                if v is not None:
                    cont_pred["%s in %s" % (child, parent)].append(v)
            for c_rle in g_by_class.get(child, []):
                v = contained(c_rle, g_by_class.get(parent, []))
                if v is not None:
                    cont_gt["%s in %s" % (child, parent)].append(v)

    print("=" * 78)
    print("run %s  split %s  |  score >= %.2f, match IoU >= %.2f"
          % (args.run, args.split, args.score, args.iou))
    print("=" * 78)
    print("predicted area divided by annotated area  (above 1 = drawn too large)")
    print("%-24s%7s%9s%9s%9s%9s" % ("class", "n", "median", "p25", "p75", "mean"))
    print("-" * 78)
    for c in RATIO_CLASSES:
        print(summarise(c, ratios.get(c, [])))

    print()
    print("=" * 78)
    print("fraction of the child lying inside its parent")
    print("%-24s%7s%9s%9s%9s%9s" % ("pair", "n", "median", "p25", "p75", "mean"))
    print("-" * 78)
    print("on the annotations (should be near 1.00, they are nested by design):")
    for k, _ in [(("%s in %s" % p), None) for p in NESTED]:
        print("  " + summarise(k, cont_gt.get(k, [])))
    print("on the predictions:")
    for k, _ in [(("%s in %s" % p), None) for p in NESTED]:
        print("  " + summarise(k, cont_pred.get(k, [])))

    print()
    for k in ["%s in %s" % p for p in NESTED]:
        v = cont_pred.get(k, [])
        if v:
            a = np.array(v)
            print("%-24s %5.0f%% of predictions are less than half inside"
                  % (k, 100.0 * float((a < 0.5).mean())))
    print("=" * 78)


if __name__ == "__main__":
    main()