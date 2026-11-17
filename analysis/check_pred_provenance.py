"""
Are these files model predictions, or the annotations in another format?

Every instance in mrcnn_cmp/stage14_input carries a score of exactly 1.0, which
no Mask R-CNN produces, so at minimum the score column is synthetic. That alone
does not settle what the masks are: the per-class counts do not match the
annotations exactly either (vaisseau 77 against 69, veine 30 against 25), so it
is not a straight copy.

The shape of the IoU distribution separates the cases. Match each instance to
its best same-class annotation and look at where the mass sits:

  spread across 0.6-0.95, long left tail   -> a real model
  piled above 0.95, little below 0.9       -> annotations that have been through
                                              the pipeline and come back slightly
                                              altered by tiling and fusion

Usage:
  python check_pred_provenance.py --in-dir <stage14_input/test> --split test
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils

import config as C


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def gt_by_class(xml_path, H, W):
    out = defaultdict(list)
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
        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(canvas, [np.array(pts).round().astype(np.int32)], 1)
        if canvas.sum() > 0:
            out[label].append(maskUtils.encode(np.asfortranarray(canvas)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    work = Path(args.work_dir)
    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        xml_of = dict((r["source_key"], r["xml_path"])
                      for r in csv.DictReader(f) if r["split"] == args.split)

    files = sorted(Path(args.in_dir).glob("*.json"))
    files = [f for f in files if not f.name[0].isdigit() or True]
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no json in %s" % args.in_dir)

    ious = []
    per_class = defaultdict(list)
    scores = set()

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        if "instances" not in rec:
            continue
        src = rec.get("source_key", pf.stem)
        xmlp = xml_of.get(src)
        if xmlp is None:
            continue
        H, W = rec["height"], rec["width"]
        gt = gt_by_class(Path(xmlp), H, W)

        by_class = defaultdict(list)
        for it in rec["instances"]:
            scores.add(round(float(it.get("score", -1)), 4))
            by_class[it["class_name"]].append(rle_in(it["mask_rle"]))

        for cname, preds in by_class.items():
            g = gt.get(cname, [])
            if not g or not preds:
                continue
            m = np.asarray(maskUtils.iou(preds, g, [0] * len(g)))
            m = m.reshape(len(preds), len(g))
            best = m.max(axis=1)
            ious.extend(best.tolist())
            per_class[cname].extend(best.tolist())

    a = np.array(ious)
    print("=" * 70)
    print("instances compared: %d   distinct score values: %d %s"
          % (len(a), len(scores), sorted(scores)[:5]))
    print("=" * 70)
    edges = [0, .3, .5, .7, .8, .9, .95, .99, 1.01]
    hist, _ = np.histogram(a, bins=edges)
    for i in range(len(hist)):
        bar = "#" * int(60 * hist[i] / max(hist.max(), 1))
        print("%4.2f-%4.2f %7d  %s" % (edges[i], edges[i + 1], hist[i], bar))
    print("-" * 70)
    print("median %.3f   mean %.3f   above 0.95: %.1f%%   below 0.7: %.1f%%"
          % (np.median(a), a.mean(), 100 * (a > .95).mean(), 100 * (a < .7).mean()))
    print()
    print("%-20s%8s%9s%9s" % ("class", "n", "median", ">0.95"))
    for c in C.STRUCTURE_CLASSES:
        v = np.array(per_class.get(c, []))
        if v.size:
            print("%-20s%8d%9.3f%8.0f%%" % (c, v.size, np.median(v),
                                            100 * (v > .95).mean()))
    print("-" * 70)
    if len(scores) == 1:
        print("Every score is identical, so the score column is synthetic and")
        print("the score-ordered matching in 14_eval degenerates to arbitrary")
        print("order. Read the histogram above for what the masks themselves are.")
    print("=" * 70)


if __name__ == "__main__":
    main()