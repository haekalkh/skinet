"""
Stage 09 - find a score threshold per class, on validation only.

One threshold for every class is a compromise: the classes differ enormously in
how confidently the model speaks about them, so a cut that suits tubules starves
intima and lets media through. This sweeps each class independently and keeps
the value with the best F1.

Tuned on VALIDATION and applied to test. Never the other way round — a threshold
chosen on test would make the test number meaningless.

The sweep runs on the output of stage 07 at its default cut, so it can only
raise a threshold, not lower it. That is the useful direction here. Once the
values are chosen, rerun stage 07 with --thresholds so the cut happens where it
belongs: before fusion, not after.

Outputs (into --out-dir):
  thresholds.json      feed this to s07_postproc.py --thresholds
  s09_sweep.csv        every class at every value, for the thesis appendix
  run_meta.json

Usage:
  python s09_tune.py --run baseline_seed42
"""

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils

import config as C
from s08_eval import gt_polygons, poly_to_rle, greedy_match, prf, rle_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", help="tune on this split")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--detect-iou", type=float, default=0.30)
    ap.add_argument("--low", type=float, default=0.50)
    ap.add_argument("--high", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--tag", default="",
                    help="suffix for the input and output folders, matching the "
                         "--tag given to stage 07")
    ap.add_argument("--min-gain", type=float, default=0.005,
                    help="keep the default threshold unless a value beats it by "
                         "at least this much; stops chasing noise")
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit(
            "refusing to tune on test. Tune on val, then apply to test.")

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run /
                          (args.split + args.tag) / "filtered")
    if args.out_dir is None:
        args.out_dir = str(work / "s09_tune" / args.run / ("tune" + args.tag))

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        xml_of = dict((r["source_key"], r["xml_path"])
                      for r in csv.DictReader(f) if r["split"] == args.split)

    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit("no stage 07 output in %s -- run it on %s first"
                         % (in_dir, args.split))

    steps = int(round((args.high - args.low) / args.step)) + 1
    grid = [round(args.low + k * args.step, 4) for k in range(steps)]

    print("=" * 78)
    print("stage 09 - threshold sweep | run %s  split %s  sources %d"
          % (args.run, args.split, len(files)))
    print("grid %.2f to %.2f step %.2f   detection IoU %.2f"
          % (args.low, args.high, args.step, args.detect_iou))
    print("=" * 78)

    # load once: predictions grouped by class, and ground truth
    per_source = []
    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        xmlp = xml_of.get(rec["source_key"])
        if xmlp is None:
            print("[warn] no xml for %s" % rec["source_key"])
            continue
        H, W = rec["height"], rec["width"]
        pred = defaultdict(list)
        for it in rec["instances"]:
            pred[it["class_name"]].append((it["score"], rle_in(it["mask_rle"])))
        gt = defaultdict(list)
        for label, pts in gt_polygons(Path(xmlp)):
            r = poly_to_rle(pts, H, W)
            if r is not None:
                gt[label].append(r)
        per_source.append((rec["source_key"], pred, gt))
    print("loaded %d sources" % len(per_source))

    rows = [("class", "threshold", "TP", "FP", "FN", "precision", "recall", "f1")]
    best = OrderedDict()
    curves = OrderedDict()

    for c in C.STRUCTURE_CLASSES:
        curve = []
        for thr in grid:
            tp = fp = fn = 0
            for _, pred, gt in per_source:
                keep = [(s, r) for s, r in pred.get(c, []) if s >= thr]
                keep.sort(key=lambda t: -t[0])
                rles = [r for _, r in keep]
                scores = [s for s, _ in keep]
                _, a, b, d = greedy_match(rles, scores, gt.get(c, []),
                                          args.detect_iou)
                tp += a; fp += b; fn += d
            p, r, f = prf(tp, fp, fn)
            curve.append((thr, tp, fp, fn, p, r, f))
            rows.append((c, thr, tp, fp, fn, round(p, 4), round(r, 4), round(f, 4)))
        curves[c] = curve

        base_f1 = curve[0][6]
        top = max(curve, key=lambda t: t[6])
        chosen = top[0] if top[6] - base_f1 >= args.min_gain else grid[0]
        chosen_f1 = next(t[6] for t in curve if t[0] == chosen)
        best[c] = chosen
        mark = "" if chosen == grid[0] else "  <- raised"
        print("%-20s  %.2f -> %.2f   F1 %.3f -> %.3f%s"
              % (c, grid[0], chosen, base_f1, chosen_f1, mark))

    macro_base = float(np.mean([curves[c][0][6] for c in C.STRUCTURE_CLASSES]))
    macro_tuned = float(np.mean(
        [next(t[6] for t in curves[c] if t[0] == best[c])
         for c in C.STRUCTURE_CLASSES]))

    with open(str(out_dir / "thresholds.json"), "w") as f:
        json.dump(OrderedDict(run=args.run, tuned_on=args.split,
                              detect_iou=args.detect_iou,
                              macro_f1_before=round(macro_base, 4),
                              macro_f1_after=round(macro_tuned, 4),
                              thresholds=best), f, indent=2)

    with open(str(out_dir / "s09_sweep.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print("-" * 78)
    print("macro F1 on %s   before %.3f   after %.3f   (+%.3f)"
          % (args.split, macro_base, macro_tuned, macro_tuned - macro_base))
    print("=" * 78)
    print("wrote %s" % (out_dir / "thresholds.json"))
    print("\nNext: rerun stage 07 with --thresholds on val AND test, so the cut")
    print("happens before fusion rather than after, then score test with s08.")
    C.write_run_meta(out_dir, "s09_tune", args,
                     extra={"macro_f1_before": round(macro_base, 4),
                            "macro_f1_after": round(macro_tuned, 4),
                            "thresholds": best})


if __name__ == "__main__":
    main()