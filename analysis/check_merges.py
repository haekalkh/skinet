"""
How often does one prediction cover two annotated objects?

Legendre reports that the segmentation merges objects that should be separate.
The fusion stage is the obvious suspect: it merges same-class predictions whose
boxes and masks overlap by as little as 0.1, and renal tubules are packed
closely enough that two correctly predicted neighbours can exceed that at their
touching edges.

This measures it directly. For every prediction, count the annotated objects of
the same class of which it contains most of the area; a prediction containing
two or more has merged them. The reverse case is counted too, an annotation
covered by several predictions, since over-segmentation would be the opposite
failure and is worth knowing about at the same time.

The ceiling measurement does not test this, because annotations passed through
the pipeline never overlap one another, so nothing there can be merged.

Usage:
  python check_merges.py --run patient_100k --split test --tag _base050
  python check_merges.py --run patient_100k --split test --tag _stage0 \
      --in-dir <stage 07 output with --stop-after 0>

Running it on the output of --stop-after 0 and again on the final output
separates the two causes: merges present before fusion come from the model,
merges that appear afterwards come from the fusion stage.
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
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(pts).round().astype(np.int32)], 1)
        if m.sum() > 0:
            out[label].append(maskUtils.encode(np.asfortranarray(m)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--contain", type=float, default=0.5,
                    help="fraction of an annotation that must lie inside a "
                         "prediction for it to count as covered")
    ap.add_argument("--list-worst", type=int, default=8)
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run /
                          (args.split + args.tag) / "filtered")

    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        xml_of = dict((r["source_key"], r["xml_path"])
                      for r in csv.DictReader(f) if r["split"] == args.split)

    files = sorted(Path(args.in_dir).glob("*.json"))
    if not files:
        raise SystemExit("no stage 07 output in %s" % args.in_dir)

    merged_pred = defaultdict(int)     # predictions covering >= 2 annotations
    merged_gt = defaultdict(int)       # annotations swallowed by such a prediction
    split_gt = defaultdict(int)        # annotations covered by >= 2 predictions
    n_pred = defaultdict(int)
    n_gt = defaultdict(int)
    worst = []

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        xmlp = xml_of.get(rec["source_key"])
        if xmlp is None:
            continue
        H, W = rec["height"], rec["width"]
        gt = gt_by_class(Path(xmlp), H, W)

        pred = defaultdict(list)
        for it in rec["instances"]:
            if it["class_name"] in C.STRUCTURE_CLASSES:
                pred[it["class_name"]].append(rle_in(it["mask_rle"]))

        slide_merges = 0
        for c in C.STRUCTURE_CLASSES:
            g, p = gt.get(c, []), pred.get(c, [])
            n_gt[c] += len(g)
            n_pred[c] += len(p)
            if not g or not p:
                continue

            # containment of each annotation within each prediction:
            # intersection over the ANNOTATION's area, not the union, so a large
            # prediction swallowing a small object registers even though the
            # IoU would be low
            inter = np.asarray(maskUtils.area(
                [maskUtils.merge([pi, gi], intersect=True)
                 for pi in p for gi in g])).reshape(len(p), len(g))
            g_area = np.asarray(maskUtils.area(g)).reshape(1, len(g))
            cover = inter / np.maximum(g_area, 1)

            covered = cover >= args.contain
            per_pred = covered.sum(axis=1)
            per_gt = covered.sum(axis=0)

            m = int((per_pred >= 2).sum())
            merged_pred[c] += m
            merged_gt[c] += int(per_pred[per_pred >= 2].sum())
            split_gt[c] += int((per_gt >= 2).sum())
            slide_merges += m

        if slide_merges:
            worst.append((slide_merges, rec["source_key"]))

    print("=" * 78)
    print("merged and split objects | run %s  split %s   containment >= %.2f"
          % (args.run, args.split + args.tag, args.contain))
    print("=" * 78)
    print("%-22s%8s%8s%10s%10s%10s" %
          ("class", "n_gt", "n_pred", "merging", "swallowed", "split"))
    print("-" * 78)
    tot_m = tot_s = tot_sp = 0
    for c in C.STRUCTURE_CLASSES:
        if not n_gt[c] and not n_pred[c]:
            continue
        print("%-22s%8d%8d%10d%10d%10d"
              % (C.display(c, short=True), n_gt[c], n_pred[c],
                 merged_pred[c], merged_gt[c], split_gt[c]))
        tot_m += merged_pred[c]; tot_s += merged_gt[c]; tot_sp += split_gt[c]
    print("-" * 78)
    print("%-22s%8d%8d%10d%10d%10d"
          % ("total", sum(n_gt.values()), sum(n_pred.values()),
             tot_m, tot_s, tot_sp))
    print()
    print("'merging' counts predictions that contain two or more annotations;")
    print("'swallowed' counts the annotations inside them, so the objects lost")
    print("to merging is swallowed minus merging. 'split' counts annotations")
    print("covered by two or more predictions, the opposite failure.")
    if sum(n_gt.values()):
        lost = tot_s - tot_m
        print()
        print("Objects lost to merging: %d of %d annotations (%.1f%%)."
              % (lost, sum(n_gt.values()), 100.0 * lost / sum(n_gt.values())))

    if worst:
        print("\nregions with the most merged predictions:")
        for n, k in sorted(worst, reverse=True)[:args.list_worst]:
            print("   %-16s %d" % (k, n))
    print("=" * 78)


if __name__ == "__main__":
    main()