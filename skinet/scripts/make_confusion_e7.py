#!/usr/bin/env python
"""
Build a CLASS confusion matrix from a prediction dir vs source-level GT, in the
exact CSV format that confusion_compare_e7.py reads. Replaces the (never-built)
Stage 25 for the seminary's 3-panel confusion.

It reuses Stage 14's GT machinery verbatim (parse_gt_main, gt_to_rle, rle_in) and
its matching style (greedy, score-ordered, IoU threshold = EVAL.confusion_iou),
but matches CROSS-CLASS: each prediction is paired to the highest-IoU *unclaimed*
GT instance regardless of class, then tallied as confusion[gt_class][pred_class].
Unmatched GT -> __FN__ ; unmatched prediction -> __FP__ row.

  python make_confusion_e7.py --split test \
      --pred-dir <OUT>/postproc/test/10_filtered \
      --out <OUT>/report_m2f/25_confusion_matrix.csv

CPU only. GT path per source comes from the split CSV `xml_path` column.
"""
import argparse
import csv
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

import e7_config as C


def load_stage14(sdir):
    p = Path(sdir) / "14_eval_main_e7.py"
    spec = importlib.util.spec_from_file_location("e7_eval14", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cross_class_confusion(pred, gt, thr, conf):
    """pred = list of (class_name, score, rle); gt = list of (class_name, rle).
    Greedy score-ordered cross-class matching at IoU >= thr; fills `conf`."""
    n_p, n_g = len(pred), len(gt)
    if n_p == 0:
        for gc, _ in gt:
            conf[gc]["__FN__"] += 1
        return
    if n_g == 0:
        for pc, _, _ in pred:
            conf["__FP__"][pc] += 1
        return
    p_rles = [r for _, _, r in pred]
    g_rles = [r for _, r in gt]
    ious = np.asarray(mask_util.iou(p_rles, g_rles, [0] * n_g)).reshape(n_p, n_g)
    order = np.argsort(-np.asarray([s for _, s, _ in pred]))
    gt_taken = [False] * n_g
    for p in order:
        best_j, best_iou = -1, thr
        for j in range(n_g):
            if gt_taken[j]:
                continue
            if ious[p, j] >= best_iou:
                best_iou, best_j = ious[p, j], j
        if best_j >= 0:
            gt_taken[best_j] = True
            conf[gt[best_j][0]][pred[p][0]] += 1
        else:
            conf["__FP__"][pred[p][0]] += 1
    for j in range(n_g):
        if not gt_taken[j]:
            conf[gt[j][0]]["__FN__"] += 1


def main():
    ap = argparse.ArgumentParser("Build class confusion CSV (pred dir vs GT)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--iou", type=float, default=C.EVAL["confusion_iou"])
    args = ap.parse_args()

    import json
    ev = load_stage14(args.scripts_dir)
    classes = list(C.MAIN_CLASSES)
    xml_of = dict(zip(pd.read_csv(args.split_csv)["source_key"],
                      pd.read_csv(args.split_csv)["xml_path"]))

    pred_dir = Path(args.pred_dir)
    files = sorted(f for f in pred_dir.glob("*.json") if not re.match(r"^\d+_", f.name))
    if not files:
        raise FileNotFoundError(f"No per-source JSON in {pred_dir}")

    # conf[gt_class][pred_class or __FN__] ; conf["__FP__"][pred_class]
    conf = {c: defaultdict(int) for c in classes}
    conf["__FP__"] = defaultdict(int)

    n_src = 0
    for pf in files:
        rec = json.load(open(pf))
        if "source_key" not in rec:
            continue
        src, H, W = rec["source_key"], rec["height"], rec["width"]
        xmlp = xml_of.get(src)
        if xmlp is None:
            print(f"[WARN] no xml_path for {src}; skipping"); continue
        pred = [(it["class_name"], float(it.get("score", 1.0)), ev.rle_in(it["mask_rle"]))
                for it in rec["instances"] if it["class_name"] in classes]
        gt = []
        for g in ev.parse_gt_main(Path(xmlp)):
            r = ev.gt_to_rle(g["points"], H, W)
            if r is not None:
                gt.append((g["class_name"], r))
        cross_class_confusion(pred, gt, args.iou, conf)
        n_src += 1

    # write CSV: header "" , classes..., __FN__ ; rows gt_class ; last row __FP__
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = classes + ["__FN__"]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + cols)
        for gc in classes:
            w.writerow([gc] + [conf[gc].get(pc, 0) for pc in cols])
        w.writerow(["__FP__"] + [conf["__FP__"].get(pc, 0) for pc in classes] + [0])

    # console summary: per-class recall (diagonal / row sum)
    print(f"confusion ({n_src} sources, IoU>={args.iou}) -> {out}")
    print(f"{'GT class':<18}{'n_gt':>6}{'correct':>8}{'recall':>8}")
    for gc in classes:
        row = conf[gc]
        n_gt = sum(row.values())
        correct = row.get(gc, 0)
        rec_ = correct / n_gt if n_gt else 0.0
        print(f"{gc:<18}{n_gt:>6}{correct:>8}{rec_:>8.2f}")


if __name__ == "__main__":
    main()