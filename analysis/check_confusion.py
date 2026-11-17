"""
Split every miss into "nothing was predicted there" versus "something was
predicted there but with the wrong label".

Plain recall merges the two, and they need opposite fixes. If a missed intima
turns out to be predicted as vaisseau, the problem is classification and no
amount of extra positive samples will help. If nothing is predicted there at
all, the object was never proposed, which is what the rare-class matching idea
is meant to address.

Works on the tile-level predictions the trainer already wrote, so no inference
run is needed.

Usage:
  python check_confusion.py --run baseline_seed42
  python check_confusion.py --run baseline_seed42 --iou 0.3 --score 0.3
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

import config as C


def gt_rle(coco, ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        return maskUtils.merge(rles)
    if isinstance(seg.get("counts"), list):
        return maskUtils.frPyObjects(seg, h, w)
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--split", default="val")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="a prediction counts as being 'there' above this mask IoU")
    ap.add_argument("--score", type=float, default=0.5,
                    help="ignore predictions below this confidence")
    args = ap.parse_args()

    work = Path(args.work_dir)
    gt_file = work / "s03_tiles" / "annotations" / ("instances_%s.json" % args.split)
    pred_file = work / "s04_train" / args.run / "eval" / "coco_instances_results.json"
    for p in (gt_file, pred_file):
        if not p.exists():
            raise SystemExit("not found: %s" % p)

    coco = COCO(str(gt_file))
    with open(str(pred_file)) as f:
        preds = json.load(f)

    id2name = dict((i + 1, c) for i, c in enumerate(C.STRUCTURE_CLASSES))

    by_img = defaultdict(list)
    n_low = 0
    for p in preds:
        if p.get("score", 1.0) < args.score:
            n_low += 1
            continue
        by_img[p["image_id"]].append(p)

    outcome = defaultdict(Counter)      # gt class -> {"correct"/"missed"/name}
    fp_class = Counter()                # predictions matching no gt
    n_pred_used = sum(len(v) for v in by_img.values())

    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        pl = by_img.get(img_id, [])

        if not anns and not pl:
            continue

        g_rles = [gt_rle(coco, a, h, w) for a in anns]
        p_rles = [p["segmentation"] for p in pl]

        if g_rles and p_rles:
            ious = maskUtils.iou(p_rles, g_rles, [0] * len(g_rles))
            ious = np.asarray(ious).reshape(len(p_rles), len(g_rles))
        else:
            ious = np.zeros((len(p_rles), len(g_rles)))

        for j, a in enumerate(anns):
            gname = id2name.get(a["category_id"], "?")
            if ious.shape[0] == 0:
                outcome[gname]["missed"] += 1
                continue
            i = int(np.argmax(ious[:, j]))
            if ious[i, j] < args.iou:
                outcome[gname]["missed"] += 1
            else:
                pname = id2name.get(pl[i]["category_id"], "?")
                if pname == gname:
                    outcome[gname]["correct"] += 1
                else:
                    outcome[gname][pname] += 1

        for i, p in enumerate(pl):
            if ious.shape[1] == 0 or ious[i].max() < args.iou:
                fp_class[id2name.get(p["category_id"], "?")] += 1

    print("=" * 78)
    print("run %s  split %s  |  IoU >= %.2f  score >= %.2f"
          % (args.run, args.split, args.iou, args.score))
    print("predictions kept %d, dropped below score %d" % (n_pred_used, n_low))
    print("=" * 78)
    print("%-20s%7s%9s%9s%9s   %s"
          % ("class", "n", "correct", "wrong", "missed", "mistaken for"))
    print("-" * 78)

    for c in C.STRUCTURE_CLASSES:
        o = outcome.get(c)
        if not o:
            print("%-20s%7d" % (c, 0))
            continue
        n = sum(o.values())
        correct = o.get("correct", 0)
        missed = o.get("missed", 0)
        wrong = n - correct - missed
        others = [(k, v) for k, v in o.items()
                  if k not in ("correct", "missed")]
        others.sort(key=lambda kv: -kv[1])
        detail = ", ".join("%s %d" % (k, v) for k, v in others[:3]) or "-"
        print("%-20s%7d%8.0f%%%8.0f%%%8.0f%%   %s"
              % (c, n, 100.0 * correct / n, 100.0 * wrong / n,
                 100.0 * missed / n, detail))

    print("-" * 78)
    print("predictions matching no ground truth (false positives):")
    for c in C.STRUCTURE_CLASSES:
        print("   %-20s%6d" % (c, fp_class.get(c, 0)))
    print("=" * 78)
    print("'wrong'  = an object WAS predicted in that place, under another name")
    print("'missed' = nothing was predicted there at all")


if __name__ == "__main__":
    main()