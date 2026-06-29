#!/usr/bin/env python
"""
E7 Stage 21 - glomerulus GRADING evaluation (predicted vs GT).

Validates the derived grade (Stage 13 logic) against ground truth:
  - load predicted nsg / nsg_partiel / nsg_complet (Stage 10) and GT polygons
    (reuses Stage 14's parse_gt_main + gt_to_rle so coordinate handling is
    IDENTICAL to the detection eval),
  - derive each glomerulus' grade by containment (complet > partiel > normal),
    on BOTH predictions and GT,
  - match predicted glomeruli to GT glomeruli by mask IoU (greedy, count_iou),
  - report a 3x3 grade confusion matrix on matched glomeruli, grading accuracy,
    glomerulus detection P/R, and GT-vs-predicted % global sclerosis.

This quantifies the grading bias (e.g. complet over-prediction inflating
'global') precisely, instead of trusting a single % number.

CPU only.   python 21_eval_grade_e7.py --split val
"""
import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

import e7_config as C

# reuse the exact, proven Stage-14 GT helpers (same dir)
_s14 = Path(__file__).resolve().with_name("14_eval_main_e7.py")
_spec = importlib.util.spec_from_file_location("e7_eval14", _s14)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)
parse_gt_main, gt_to_rle, rle_in = _ev.parse_gt_main, _ev.gt_to_rle, _ev.rle_in

GRADES = ["normal", "segmental", "global"]
RANK = {"normal": 0, "segmental": 1, "global": 2}
GRADE_OF_CHILD = {"nsg_partiel": "segmental", "nsg_complet": "global"}


def derive_grades(nsg_rles, children, cover_thr):
    """class mode: grade per nsg by contained lesion type (complet>partiel>normal)."""
    kids = [(lab, r, float(mask_util.area(r))) for lab, r in children]
    grades = []
    for g in nsg_rles:
        grade = "normal"
        for lab, crle, a_ch in kids:
            if a_ch <= 0:
                continue
            inter = float(mask_util.area(mask_util.merge([crle, g], intersect=True)))
            if inter / a_ch >= cover_thr:
                cand = GRADE_OF_CHILD[lab]
                if RANK[cand] > RANK[grade]:
                    grade = cand
        grades.append(grade)
    return grades


def derive_grades_coverage(nsg_rles, children, global_thr, present_thr):
    """coverage mode: grade per nsg by fraction of glomerulus covered by ANY lesion
    (ignores the model's partiel/complet label)."""
    child_rles = [r for _, r in children]
    grades = []
    for g in nsg_rles:
        a_g = float(mask_util.area(g))
        cov = 0.0
        if a_g > 0 and child_rles:
            union = mask_util.merge(child_rles, intersect=False)
            inter = float(mask_util.area(mask_util.merge([union, g], intersect=True)))
            cov = inter / a_g
        if cov >= global_thr:
            grades.append("global")
        elif cov >= present_thr:
            grades.append("segmental")
        else:
            grades.append("normal")
    return grades


def grade_set(nsg_rles, children, args):
    if args.grade_by == "coverage":
        return derive_grades_coverage(nsg_rles, children,
                                      args.global_cover_thr, args.lesion_present_thr)
    return derive_grades(nsg_rles, children, args.cover_thr)


def match_pairs(pred_rles, pred_scores, gt_rles, thr):
    """Greedy score-ordered IoU match. Returns matched (pred_i, gt_j) pairs +
    unmatched pred indices + unmatched gt indices."""
    np_, ng = len(pred_rles), len(gt_rles)
    if np_ == 0 or ng == 0:
        return [], list(range(np_)), list(range(ng))
    ious = np.asarray(mask_util.iou(pred_rles, gt_rles, [0] * ng)).reshape(np_, ng)
    order = np.argsort(pred_scores)[::-1]
    gt_taken = [False] * ng
    pairs, matched_pred = [], set()
    for p in order:
        best_j, best = -1, thr
        for j in range(ng):
            if gt_taken[j]:
                continue
            if ious[p, j] >= best:
                best, best_j = ious[p, j], j
        if best_j >= 0:
            gt_taken[best_j] = True
            pairs.append((p, best_j))
            matched_pred.add(p)
    un_pred = [p for p in range(np_) if p not in matched_pred]
    un_gt = [j for j in range(ng) if not gt_taken[j]]
    return pairs, un_pred, un_gt


def main():
    ap = argparse.ArgumentParser("E7 Stage 21 glomerulus grading eval")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None, help="Stage 10 (or 12) per-source dir")
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--cover-thr", type=float, default=0.5,
                    help="(class mode) child counted inside a glomerulus if area(child&nsg)/area(child) >= thr")
    ap.add_argument("--grade-by", default="class", choices=["class", "coverage"],
                    help="class: grade = type of contained lesion (complet>partiel). "
                         "coverage: grade from fraction of glomerulus covered by ANY lesion")
    ap.add_argument("--global-cover-thr", type=float, default=0.70,
                    help="(coverage mode) lesion covers >= this fraction of glomerulus -> global")
    ap.add_argument("--lesion-present-thr", type=float, default=0.05,
                    help="(coverage mode) any coverage >= this -> at least segmental")
    ap.add_argument("--complet-min-score", type=float, default=0.0,
                    help="ignore predicted nsg_complet lesions below this score (calibration sweep)")
    ap.add_argument("--partiel-min-score", type=float, default=0.0,
                    help="ignore predicted nsg_partiel lesions below this score")
    ap.add_argument("--match-iou", type=float, default=None,
                    help="glomerulus match IoU (default C.EVAL count_iou)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "postproc" / args.split / "10_filtered")
    match_iou = args.match_iou if args.match_iou is not None else C.EVAL["count_iou"]
    df = pd.read_csv(args.split_csv)
    xml_of = dict(zip(df["source_key"], df["xml_path"]))
    files = sorted(f for f in in_dir.glob("*.json") if not f.name[0].isdigit())

    conf = defaultdict(int)                 # (gt_grade, pred_grade) -> count
    n_missed = Counter()                    # GT grade unmatched (FN)
    n_extra = Counter()                     # pred grade unmatched (FP)
    gt_dist, pred_dist = Counter(), Counter()

    for pf in files:
        rec = json.load(open(pf))
        if "source_key" not in rec:
            continue
        src, H, W = rec["source_key"], rec["height"], rec["width"]
        xmlp = xml_of.get(src)
        if xmlp is None:
            continue

        # predictions
        pby = defaultdict(list)
        pscore = defaultdict(list)
        for it in rec["instances"]:
            if it["class_name"] in ("nsg", "nsg_partiel", "nsg_complet"):
                pby[it["class_name"]].append(rle_in(it["mask_rle"]))
                pscore[it["class_name"]].append(it.get("score", 1.0))
        pred_nsg = pby["nsg"]
        pred_children = (
            [("nsg_partiel", r) for r, s in zip(pby["nsg_partiel"], pscore["nsg_partiel"])
             if s >= args.partiel_min_score]
            + [("nsg_complet", r) for r, s in zip(pby["nsg_complet"], pscore["nsg_complet"])
               if s >= args.complet_min_score])
        pred_grades = grade_set(pred_nsg, pred_children, args)

        # ground truth
        gby = defaultdict(list)
        for g in parse_gt_main(Path(xmlp)):
            if g["class_name"] in ("nsg", "nsg_partiel", "nsg_complet"):
                r = gt_to_rle(g["points"], H, W)
                if r is not None:
                    gby[g["class_name"]].append(r)
        gt_nsg = gby["nsg"]
        gt_children = ([("nsg_partiel", r) for r in gby["nsg_partiel"]]
                       + [("nsg_complet", r) for r in gby["nsg_complet"]])
        # GT grade is ALWAYS the annotated lesion class (the clinical ground
        # truth): a glomerulus is 'global' if it carries a complet annotation,
        # 'segmental' if partiel. Only the PREDICTION grade varies by --grade-by.
        gt_grades = derive_grades(gt_nsg, gt_children, args.cover_thr)

        for g in gt_grades:
            gt_dist[g] += 1
        for g in pred_grades:
            pred_dist[g] += 1

        pairs, un_pred, un_gt = match_pairs(pred_nsg, pscore["nsg"], gt_nsg, match_iou)
        for pi, gj in pairs:
            conf[(gt_grades[gj], pred_grades[pi])] += 1
        for gj in un_gt:
            n_missed[gt_grades[gj]] += 1
        for pi in un_pred:
            n_extra[pred_grades[pi]] += 1

    matched = sum(conf.values())
    correct = sum(conf[(g, g)] for g in GRADES)
    acc = correct / matched if matched else 0.0
    gt_tot, pred_tot = sum(gt_dist.values()), sum(pred_dist.values())
    n_gt_glom = gt_tot
    recall = matched / n_gt_glom if n_gt_glom else 0.0
    precision = matched / (matched + sum(n_extra.values())) if (matched + sum(n_extra.values())) else 0.0

    print("=" * 70)
    print(f"E7 Stage 21 - glomerulus grading eval | split={args.split}")
    print(f"  in {in_dir}")
    print(f"  match IoU {match_iou}   cover-thr {args.cover_thr}")
    print("=" * 70)
    print("grade confusion (rows = GT, cols = predicted), matched glomeruli only:")
    hdr = " " * 12 + "".join(f"{g[:6]:>9}" for g in GRADES) + f"{'FN(miss)':>10}"
    print(hdr)
    for gt_g in GRADES:
        row = "".join(f"{conf[(gt_g, pr)]:>9}" for pr in GRADES)
        print(f"  GT {gt_g:<8}{row}{n_missed[gt_g]:>10}")
    print(f"  {'FP(extra)':<10}" + "".join(f"{n_extra[pr]:>9}" for pr in GRADES))
    print("-" * 70)
    print(f"matched glomeruli      : {matched}")
    print(f"grading accuracy       : {acc*100:.1f}%   (matched, grade correct)")
    print(f"glomerulus detection   : recall {recall*100:.1f}%  precision {precision*100:.1f}%")
    print("-" * 70)
    def pg(dist, tot):
        return (dist['global'] / tot * 100) if tot else 0.0
    print(f"% global sclerosis  GT  : {pg(gt_dist, gt_tot):.1f}%   "
          f"({gt_dist['global']}/{gt_tot}: norm {gt_dist['normal']} seg {gt_dist['segmental']} glob {gt_dist['global']})")
    print(f"% global sclerosis PRED : {pg(pred_dist, pred_tot):.1f}%   "
          f"({pred_dist['global']}/{pred_tot}: norm {pred_dist['normal']} seg {pred_dist['segmental']} glob {pred_dist['global']})")
    print("\nREAD: diagonal = correct grade. Mass in (GT segmental -> pred global)")
    print("      = complet over-prediction inflating 'global'. Compare GT vs PRED %.")


if __name__ == "__main__":
    main()