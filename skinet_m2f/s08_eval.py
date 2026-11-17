"""
Stage 08 - score the post-processed predictions against the annotations.

Whole-ROI level, not tile level. This is the number that means something
clinically: did the system find the structures on the slide, and did it draw
them where they are.

Matching is greedy in score order at two operating points:
  detection IoU 0.30 - the clinical question, "was it found"
  strict    IoU 0.50 - the same question with the boundary held to account

For matched pairs it also reports mean instance IoU and Dice, which separates
"found but drawn loosely" from "found and drawn well". A class can have good
recall and poor Dice, or the reverse, and the two need different fixes.

Outputs (into --out-dir):
  s08_eval.json        full report
  s08_per_class.csv    thesis table
  s08_per_source.json  TP/FP/FN per class per ROI, for resampling
  run_meta.json

Usage:
  python s08_eval.py --run baseline_seed42 --split test
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


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def gt_polygons(xml_path):
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
        if len(pts) >= 3:
            out.append((label, np.array(pts, dtype=np.float64)))
    return out


def poly_to_rle(points, H, W):
    canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(canvas, [points.round().astype(np.int32)], 1)
    if canvas.sum() == 0:
        return None
    return maskUtils.encode(np.asfortranarray(canvas))


def greedy_match(pred_rles, scores, gt_rles, thr):
    """Highest-scoring prediction claims its best free ground truth first."""
    npred, ngt = len(pred_rles), len(gt_rles)
    if npred == 0:
        return [], 0, 0, ngt
    if ngt == 0:
        return [], 0, npred, 0
    ious = np.asarray(maskUtils.iou(pred_rles, gt_rles, [0] * ngt))
    ious = ious.reshape(npred, ngt)
    taken = [False] * ngt
    matched = []
    for p in np.argsort(-np.asarray(scores)):
        best_j, best = -1, thr
        for j in range(ngt):
            if taken[j]:
                continue
            if ious[p, j] >= best:
                best, best_j = ious[p, j], j
        if best_j >= 0:
            taken[best_j] = True
            matched.append(float(ious[p, best_j]))
    tp = len(matched)
    return matched, tp, npred - tp, ngt - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--detect-iou", type=float, default=0.30)
    ap.add_argument("--strict-iou", type=float, default=0.50)
    ap.add_argument("--tag", default="",
                    help="suffix for the input and output folders, matching the "
                         "--tag given to stage 07")
    ap.add_argument("--sources-file", default=None,
                    help="restrict evaluation to the source keys listed in this "
                         "file, one per line. Use it to score two models on the "
                         "same subset — dropping sources changes the class mix, "
                         "so a comparison is only meaningful when both sides see "
                         "the same list.")
    ap.add_argument("--score", type=float, default=0.0,
                    help="extra score floor on top of what stage 07 already applied")
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run /
                          (args.split + args.tag) / "filtered")
    if args.out_dir is None:
        args.out_dir = str(work / "s08_eval" / args.run /
                           (args.split + args.tag))

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        xml_of = dict((r["source_key"], r["xml_path"])
                      for r in csv.DictReader(f) if r["split"] == args.split)

    files = sorted(in_dir.glob("*.json"))
    if args.sources_file:
        keep = set(Path(args.sources_file).read_text().split())
        files = [f for f in files if f.stem in keep]
        print("restricted to %d listed sources -> %d files" % (len(keep), len(files)))
    if not files:
        raise SystemExit("no stage 07 output in %s" % in_dir)

    print("=" * 84)
    print("stage 08 - evaluation | run %s  split %s  sources %d"
          % (args.run, args.split, len(files)))
    print("detection IoU %.2f   strict IoU %.2f" % (args.detect_iou, args.strict_iou))
    print("=" * 84)

    acc = dict((c, defaultdict(float)) for c in C.STRUCTURE_CLASSES)
    ious = dict((c, []) for c in C.STRUCTURE_CLASSES)
    # Per-source counts, kept so that the test set can be resampled. Bootstrap
    # over patients requires the scores to be recomputed on each resample, and
    # an aggregate cannot be taken apart afterwards. The matching is already
    # performed below; only the result was being discarded.
    per_source = OrderedDict()

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        src, H, W = rec["source_key"], rec["height"], rec["width"]
        xmlp = xml_of.get(src)
        if xmlp is None:
            print("[warn] no xml for %s" % src)
            continue

        pred = defaultdict(lambda: {"rles": [], "scores": []})
        for it in rec["instances"]:
            if it["score"] < args.score:
                continue
            pred[it["class_name"]]["rles"].append(rle_in(it["mask_rle"]))
            pred[it["class_name"]]["scores"].append(it["score"])

        gt = defaultdict(list)
        for label, pts in gt_polygons(Path(xmlp)):
            r = poly_to_rle(pts, H, W)
            if r is not None:
                gt[label].append(r)

        row = OrderedDict()
        for c in C.STRUCTURE_CLASSES:
            pr, ps, gr = pred[c]["rles"], pred[c]["scores"], gt[c]
            m, tp, fp, fn = greedy_match(pr, ps, gr, args.detect_iou)
            _, tp5, fp5, fn5 = greedy_match(pr, ps, gr, args.strict_iou)
            acc[c]["tp"] += tp; acc[c]["fp"] += fp; acc[c]["fn"] += fn
            acc[c]["tp5"] += tp5; acc[c]["fp5"] += fp5; acc[c]["fn5"] += fn5
            acc[c]["n_gt"] += len(gr); acc[c]["n_pred"] += len(pr)
            ious[c].extend(m)
            row[c] = OrderedDict(n_gt=len(gr), n_pred=len(pr),
                                 tp=int(tp), fp=int(fp), fn=int(fn),
                                 tp5=int(tp5), fp5=int(fp5), fn5=int(fn5),
                                 sum_iou=round(float(sum(m)), 6),
                                 n_iou=len(m))
        per_source[src] = row

    per_class = OrderedDict()
    for c in C.STRUCTURE_CLASSES:
        a = acc[c]
        p, r, f = prf(a["tp"], a["fp"], a["fn"])
        p5, r5, f5 = prf(a["tp5"], a["fp5"], a["fn5"])
        v = np.array(ious[c]) if ious[c] else np.array([])
        miou = float(v.mean()) if v.size else 0.0
        mdice = float(np.mean(2 * v / (1 + v))) if v.size else 0.0
        per_class[c] = OrderedDict(
            n_gt=int(a["n_gt"]), n_pred=int(a["n_pred"]),
            tp=int(a["tp"]), fp=int(a["fp"]), fn=int(a["fn"]),
            precision=round(p, 4), recall=round(r, 4), f1=round(f, 4),
            f1_strict=round(f5, 4),
            mean_iou=round(miou, 4), mean_dice=round(mdice, 4))

    def macro(field, keys=None):
        keys = keys or C.STRUCTURE_CLASSES
        return round(float(np.mean([per_class[c][field] for c in keys])), 4)

    hard = [c for c in C.STRUCTURE_CLASSES
            if c in ("intima", "media", "veine", "vaisseau",
                     "nsg_partiel", "nsg_complet", "tubule_atrophique")]
    easy = [c for c in C.STRUCTURE_CLASSES if c not in hard]

    report = OrderedDict(
        run=args.run, split=args.split, n_sources=len(files),
        sources_file=args.sources_file,
        detect_iou=args.detect_iou, strict_iou=args.strict_iou,
        per_class=per_class,
        macro_all=dict((k, macro(k)) for k in ("precision", "recall", "f1",
                                               "f1_strict", "mean_dice")),
        macro_hard=dict((k, macro(k, hard)) for k in ("precision", "recall", "f1")),
        macro_easy=dict((k, macro(k, easy)) for k in ("precision", "recall", "f1")))

    with open(str(out_dir / "s08_eval.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Written separately rather than folded into the report, which is read by
    # eye; this one is machine input.
    with open(str(out_dir / "s08_per_source.json"), "w") as f:
        json.dump(OrderedDict(run=args.run, split=args.split,
                              detect_iou=args.detect_iou,
                              strict_iou=args.strict_iou,
                              score=args.score,
                              classes=list(C.STRUCTURE_CLASSES),
                              sources=per_source), f, indent=1)

    with open(str(out_dir / "s08_per_class.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_gt", "n_pred", "TP", "FP", "FN",
                    "precision", "recall", "f1", "f1_strict",
                    "mean_iou", "mean_dice"])
        for c in C.STRUCTURE_CLASSES:
            m = per_class[c]
            w.writerow([c, m["n_gt"], m["n_pred"], m["tp"], m["fp"], m["fn"],
                        m["precision"], m["recall"], m["f1"], m["f1_strict"],
                        m["mean_iou"], m["mean_dice"]])

    print("\n%-20s%6s%6s%8s%8s%8s%8s%8s" %
          ("class", "nGT", "nPr", "P", "R", "F1", "F1@50", "Dice"))
    print("-" * 84)
    for c in C.STRUCTURE_CLASSES:
        m = per_class[c]
        print("%-20s%6d%6d%8.3f%8.3f%8.3f%8.3f%8.3f" %
              (c, m["n_gt"], m["n_pred"], m["precision"], m["recall"],
               m["f1"], m["f1_strict"], m["mean_dice"]))
    print("-" * 84)
    print(" macro all    F1 %.3f   Dice %.3f"
          % (report["macro_all"]["f1"], report["macro_all"]["mean_dice"]))
    print(" macro hard   F1 %.3f" % report["macro_hard"]["f1"])
    print(" macro easy   F1 %.3f" % report["macro_easy"]["f1"])
    print("=" * 84)
    C.write_run_meta(out_dir, "s08_eval", args, extra=report["macro_all"])


if __name__ == "__main__":
    main()