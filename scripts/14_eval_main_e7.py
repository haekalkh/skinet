"""
E7 Stage 14 - Evaluate MAIN pipeline vs source-level GT.

Config-driven (imports e7_config). Compares the post-processed global instances
(Stage 10) against GT polygons parsed from each source XML, at the SOURCE level
(no tiling) -- i.e. the real clinical endpoint, not tile-level COCO AP.

Per class, greedy score-ordered matching by mask IoU at two operating points:
  - EVAL.count_iou (0.30): clinical detection rate -> P / R / F1 (Marechal style)
  - EVAL.ap_iou   (0.50): stricter operating point -> P / R / F1
For matched pairs: mean instance IoU and Dice (Dice = 2*IoU/(1+IoU)).
Also semantic IoU/Dice: per class, union all pred vs union all GT (pixel level).

Reports per-class + macro (all classes) + macro over KNOWN_HARD_CLASSES, and
writes a CSV for the thesis table.

Input : OUT_MAIN/postproc/{split}/10_filtered/{source}.json  + source XMLs
Output: OUT_MAIN/eval_pipeline/{split}/14_eval.json  + 14_eval_per_class.csv
"""

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

import e7_config as C


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rle_in(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def parse_gt_main(xml_path: Path):
    """Return list of {class_name, points} for the 10 MAIN classes."""
    out = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out
    for ann in root.iter():
        if "annotation" not in ann.tag.lower():
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = C.normalize_label(raw)
        if label not in C.MAIN_CLASSES:
            continue
        pts = []
        for coord in ann.iter():
            if coord.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(coord.attrib["X"]), float(coord.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) >= 3:
            out.append({"class_name": label,
                        "points": np.array(pts, dtype=np.float64)})
    return out


def gt_to_rle(points, H, W):
    canvas = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(canvas, [points.round().astype(np.int32)], 1)
    if canvas.sum() == 0:
        return None
    return mask_util.encode(np.asfortranarray(canvas))


def greedy_match(pred_rles, pred_scores, gt_rles, thr):
    """Greedy score-ordered matching. Returns (tp_ious, n_tp, n_fp, n_fn)."""
    n_pred, n_gt = len(pred_rles), len(gt_rles)
    if n_pred == 0:
        return [], 0, 0, n_gt
    if n_gt == 0:
        return [], 0, n_pred, 0
    ious = mask_util.iou(pred_rles, gt_rles, [0] * n_gt)  # [n_pred, n_gt]
    ious = np.asarray(ious).reshape(n_pred, n_gt)
    order = np.argsort(-np.asarray(pred_scores))
    gt_taken = [False] * n_gt
    tp_ious = []
    for p in order:
        best_j, best_iou = -1, thr
        for j in range(n_gt):
            if gt_taken[j]:
                continue
            if ious[p, j] >= best_iou:
                best_iou, best_j = ious[p, j], j
        if best_j >= 0:
            gt_taken[best_j] = True
            tp_ious.append(float(ious[p, best_j]))
    n_tp = len(tp_ious)
    return tp_ious, n_tp, n_pred - n_tp, n_gt - n_tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def semantic_iou(pred_rles, gt_rles):
    if not pred_rles and not gt_rles:
        return None, None
    if not pred_rles or not gt_rles:
        return 0.0, 0.0
    pu = mask_util.merge(pred_rles, intersect=False)
    gu = mask_util.merge(gt_rles, intersect=False)
    iou = float(np.asarray(mask_util.iou([pu], [gu], [0])).reshape(-1)[0])
    dice = 2 * iou / (1 + iou) if iou > 0 else 0.0
    return iou, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sources-file", default=None,
                    help="restrict eval to source_keys listed in this file (leakage-free subset)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "postproc" / args.split / "10_filtered")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "eval_pipeline" / args.split)
    ensure_dir(out_dir)

    files = sorted(f for f in in_dir.glob("*.json") if not re.match(r"^\d+_", f.name))
    if args.sources_file:
        keep_src = set(Path(args.sources_file).read_text().split())
        files = [f for f in files if f.stem in keep_src]
        print(f"[filter] restricted to {len(keep_src)} listed sources -> {len(files)} files")
    if not files:
        raise FileNotFoundError(f"No Stage 10 filtered files in {in_dir}")

    xml_of = dict(zip(pd.read_csv(args.split_csv)["source_key"],
                      pd.read_csv(args.split_csv)["xml_path"]))

    count_iou = C.EVAL["count_iou"]
    ap_iou = C.EVAL["ap_iou"]

    print("=" * 80)
    print(f"E7 Stage 14 - Evaluation | split={args.split}  sources={len(files)}")
    print(f"  detection IoU={count_iou}   strict IoU={ap_iou}")
    print("=" * 80)

    # accumulators per class
    acc = {c: defaultdict(float) for c in C.MAIN_CLASSES}
    inst_iou = {c: [] for c in C.MAIN_CLASSES}
    sem = {c: {"iou": [], "dice": []} for c in C.MAIN_CLASSES}

    for pf in files:
        with open(pf) as f:
            rec = json.load(f)
        if "source_key" not in rec:
            print(f"[skip] {pf.name}: not a per-source record")
            continue
        src, H, W = rec["source_key"], rec["height"], rec["width"]
        xmlp = xml_of.get(src)
        if xmlp is None:
            print(f"[WARN] no XML for {src}; skipping")
            continue

        # group predictions by class
        pred_by = defaultdict(lambda: {"rles": [], "scores": []})
        for it in rec["instances"]:
            pred_by[it["class_name"]]["rles"].append(rle_in(it["mask_rle"]))
            pred_by[it["class_name"]]["scores"].append(it["score"])

        # group GT by class
        gt_by = defaultdict(list)
        for g in parse_gt_main(Path(xmlp)):
            r = gt_to_rle(g["points"], H, W)
            if r is not None:
                gt_by[g["class_name"]].append(r)

        for c in C.MAIN_CLASSES:
            pr, ps = pred_by[c]["rles"], pred_by[c]["scores"]
            gr = gt_by[c]
            # detection operating point
            tp_ious, tp, fp, fn = greedy_match(pr, ps, gr, count_iou)
            acc[c]["tp"] += tp; acc[c]["fp"] += fp; acc[c]["fn"] += fn
            inst_iou[c].extend(tp_ious)
            # strict operating point
            _, tp2, fp2, fn2 = greedy_match(pr, ps, gr, ap_iou)
            acc[c]["tp50"] += tp2; acc[c]["fp50"] += fp2; acc[c]["fn50"] += fn2
            acc[c]["n_gt"] += len(gr); acc[c]["n_pred"] += len(pr)
            # semantic
            si, sd = semantic_iou(pr, gr)
            if si is not None:
                sem[c]["iou"].append(si); sem[c]["dice"].append(sd)

    # build per-class report
    per_class = OrderedDict()
    for c in C.MAIN_CLASSES:
        a = acc[c]
        p, r, f = prf(a["tp"], a["fp"], a["fn"])
        p5, r5, f5 = prf(a["tp50"], a["fp50"], a["fn50"])
        miou = float(np.mean(inst_iou[c])) if inst_iou[c] else 0.0
        mdice = float(np.mean([2 * i / (1 + i) for i in inst_iou[c]])) if inst_iou[c] else 0.0
        siou = float(np.mean(sem[c]["iou"])) if sem[c]["iou"] else 0.0
        sdice = float(np.mean(sem[c]["dice"])) if sem[c]["dice"] else 0.0
        per_class[c] = OrderedDict(
            n_gt=int(a["n_gt"]), n_pred=int(a["n_pred"]),
            tp=int(a["tp"]), fp=int(a["fp"]), fn=int(a["fn"]),
            precision=round(p, 4), recall=round(r, 4), f1=round(f, 4),
            precision50=round(p5, 4), recall50=round(r5, 4), f1_50=round(f5, 4),
            mean_instance_iou=round(miou, 4), mean_instance_dice=round(mdice, 4),
            semantic_iou=round(siou, 4), semantic_dice=round(sdice, 4),
            hard=c in C.KNOWN_HARD_CLASSES)

    def macro(keys, field):
        vals = [per_class[c][field] for c in keys]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    all_c = C.MAIN_CLASSES
    hard_c = [c for c in C.MAIN_CLASSES if c in C.KNOWN_HARD_CLASSES]
    easy_c = [c for c in C.MAIN_CLASSES if c not in C.KNOWN_HARD_CLASSES]

    report = OrderedDict(
        split=args.split, detection_iou=count_iou, strict_iou=ap_iou,
        num_sources=len(files), per_class=per_class,
        macro_all={k: macro(all_c, k) for k in ("precision", "recall", "f1",
                                                "mean_instance_dice", "semantic_dice")},
        macro_hard={k: macro(hard_c, k) for k in ("precision", "recall", "f1",
                                                  "mean_instance_dice", "semantic_dice")},
        macro_easy={k: macro(easy_c, k) for k in ("precision", "recall", "f1",
                                                  "mean_instance_dice", "semantic_dice")},
    )

    with open(out_dir / "14_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    # CSV for the thesis table
    csv_path = out_dir / "14_eval_per_class.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "n_gt", "n_pred", "TP", "FP", "FN",
                    "P@0.30", "R@0.30", "F1@0.30", "F1@0.50",
                    "inst_IoU", "inst_Dice", "sem_IoU", "sem_Dice", "hard"])
        for c in C.MAIN_CLASSES:
            m = per_class[c]
            w.writerow([c, m["n_gt"], m["n_pred"], m["tp"], m["fp"], m["fn"],
                        m["precision"], m["recall"], m["f1"], m["f1_50"],
                        m["mean_instance_iou"], m["mean_instance_dice"],
                        m["semantic_iou"], m["semantic_dice"], m["hard"]])

    # console table
    print(f"\n{'class':<18}{'nGT':>5}{'nPr':>6}{'P':>7}{'R':>7}{'F1':>7}"
          f"{'F1@50':>7}{'iIoU':>7}{'iDice':>7}{'sDice':>7}")
    for c in C.MAIN_CLASSES:
        m = per_class[c]
        tag = "*" if m["hard"] else " "
        print(f"{tag}{c:<17}{m['n_gt']:>5}{m['n_pred']:>6}{m['precision']:>7.3f}"
              f"{m['recall']:>7.3f}{m['f1']:>7.3f}{m['f1_50']:>7.3f}"
              f"{m['mean_instance_iou']:>7.3f}{m['mean_instance_dice']:>7.3f}"
              f"{m['semantic_dice']:>7.3f}")
    print("-" * 78)
    print(f" macro all   F1={report['macro_all']['f1']:.3f}  "
          f"Dice={report['macro_all']['mean_instance_dice']:.3f}")
    print(f" macro hard  F1={report['macro_hard']['f1']:.3f}  "
          f"Dice={report['macro_hard']['mean_instance_dice']:.3f}  (* classes)")
    print(f" macro easy  F1={report['macro_easy']['f1']:.3f}")
    print(f"\nSaved: {out_dir / '14_eval.json'}\n       {csv_path}")


if __name__ == "__main__":
    main()