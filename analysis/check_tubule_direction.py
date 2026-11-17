"""
Which way does the tubule error run?

Atrophy with interstitial fibrosis is the strongest prognostic marker in a
kidney biopsy, so of the ten classes this is the one a nephrologist reads first.
Aggregate metrics hide what matters about it: an F1 of 0.67 says nothing about
whether the system is calling atrophic tubules healthy or healthy ones atrophic.
Those two failures have opposite clinical meaning. The first understates
disease; the second overstates it.

For every annotated tubule this reports what the system said about it: the
correct class, the other tubule class, or nothing at all. Counts are given per
slide as well as in total, because a systematic bias and a few bad slides look
identical in an aggregate.

Usage:
  python check_tubule_direction.py --run patient_40k --split test --tag _base050
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

PAIR = ["tubule_sain", "tubule_atrophique"]


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def gt_of(xml_path, H, W, wanted):
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
        if label not in wanted:
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
    ap.add_argument("--iou", type=float, default=0.30)
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

    tally = dict((c, OrderedDict(n_gt=0, correct=0, other=0, missed=0))
                 for c in PAIR)
    per_slide = []

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        xmlp = xml_of.get(rec["source_key"])
        if xmlp is None:
            continue
        H, W = rec["height"], rec["width"]
        gt = gt_of(Path(xmlp), H, W, set(PAIR))

        pred = defaultdict(list)
        for it in rec["instances"]:
            if it["class_name"] in PAIR:
                pred[it["class_name"]].append(rle_in(it["mask_rle"]))

        slide = {"source": rec["source_key"]}
        for c in PAIR:
            other = PAIR[1] if c == PAIR[0] else PAIR[0]
            g = gt.get(c, [])
            if not g:
                slide[c] = (0, 0, 0, 0)
                continue
            same = pred.get(c, [])
            cross = pred.get(other, [])

            def best(preds):
                if not preds:
                    return np.zeros(len(g))
                m = np.asarray(maskUtils.iou(preds, g, [0] * len(g)))
                return m.reshape(len(preds), len(g)).max(axis=0)

            b_same, b_cross = best(same), best(cross)
            correct = int(((b_same >= args.iou) &
                           (b_same >= b_cross)).sum())
            wrong = int(((b_cross >= args.iou) &
                         (b_cross > b_same)).sum())
            missed = len(g) - correct - wrong
            tally[c]["n_gt"] += len(g)
            tally[c]["correct"] += correct
            tally[c]["other"] += wrong
            tally[c]["missed"] += missed
            slide[c] = (len(g), correct, wrong, missed)
        per_slide.append(slide)

    print("=" * 76)
    print("what the system said about each annotated tubule | run %s  split %s"
          % (args.run, args.split + args.tag))
    print("matched at IoU >= %.2f" % args.iou)
    print("=" * 76)
    print("%-20s%8s%10s%10s%10s%12s" %
          ("annotated as", "n", "correct", "as other", "missed", "as other %"))
    print("-" * 76)
    for c in PAIR:
        t = tally[c]
        n = max(t["n_gt"], 1)
        print("%-20s%8d%10d%10d%10d%11.1f%%" %
              (c, t["n_gt"], t["correct"], t["other"], t["missed"],
               100.0 * t["other"] / n))
    print("-" * 76)

    a_as_h = tally["tubule_atrophique"]["other"]
    h_as_a = tally["tubule_sain"]["other"]
    print("atrophic called healthy : %d" % a_as_h)
    print("healthy called atrophic : %d" % h_as_a)
    if a_as_h + h_as_a > 0:
        net = a_as_h - h_as_a
        print()
        if net > 0:
            print("Net %d atrophic tubules are reported as healthy. The system"
                  % net)
            print("therefore understates atrophy, which is the direction that")
            print("understates disease severity.")
        elif net < 0:
            print("Net %d healthy tubules are reported as atrophic, so the"
                  % -net)
            print("system overstates atrophy rather than missing it.")
        else:
            print("The confusion is symmetric: neither class is systematically")
            print("favoured, and the error is one of discrimination rather than")
            print("of bias.")

    worst = sorted(per_slide,
                   key=lambda s: -s.get("tubule_atrophique", (0, 0, 0, 0))[2])[:5]
    print()
    print("slides where most atrophic tubules were called healthy:")
    for s in worst:
        n, cor, wrong, miss = s.get("tubule_atrophique", (0, 0, 0, 0))
        if wrong:
            print("   %-16s %d of %d" % (s["source"], wrong, n))
    print("=" * 76)


if __name__ == "__main__":
    main()