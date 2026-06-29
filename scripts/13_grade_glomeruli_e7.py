#!/usr/bin/env python
"""
E7 Stage 13 - glomerulus grading by containment (morphometry numerator).

Now that nsg (glomerulus parent) and nsg_partiel / nsg_complet (nested sclerosis
sub-regions) coexist as instances, the clinical GRADE of each glomerulus is
DERIVED from which lesion it contains -- no entangled 3-way score decision:

  glomerulus contains an nsg_complet child  -> global sclerosis
  else contains an nsg_partiel child        -> segmental sclerosis
  else                                      -> normal

Containment = area(child & nsg) / area(child) >= --cover-thr.

Outputs, per source and aggregate:
  - per-glomerulus grade,
  - counts {normal, segmental, global},
  - % global sclerosis = global / total glomeruli  (Marechal's key endpoint).

CPU only. Reads Stage 10 (or Stage 12) per-source instances.

  python 13_grade_glomeruli_e7.py --split val
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from pycocotools import mask as mask_util

import e7_config as C

PARENT = "nsg"
CHILDREN = ("nsg_partiel", "nsg_complet")
GRADE_OF_CHILD = {"nsg_complet": "global", "nsg_partiel": "segmental"}
GRADE_RANK = {"global": 2, "segmental": 1, "normal": 0}   # complet dominates


def rle_in(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def main():
    ap = argparse.ArgumentParser("E7 Stage 13 glomerulus grading")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None, help="Stage 10 (or 12) per-source dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cover-thr", type=float, default=0.5,
                    help="child counted inside a glomerulus if area(child&nsg)/area(child) >= thr")
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "postproc" / args.split / "10_filtered")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "morphometry" / args.split / "13_glomerulus_grade")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(in_dir.glob("*.json")) if not f.name[0].isdigit()]
    if not files:
        raise FileNotFoundError(f"No per-source JSONs in {in_dir}")

    agg = Counter()
    orphan_children = 0
    per_source_rows = []

    for pf in files:
        rec = json.load(open(pf))
        if "instances" not in rec:
            continue
        src = rec["source_key"]
        nsg, children = [], []
        for it in rec["instances"]:
            if it["class_name"] == PARENT:
                nsg.append(it)
            elif it["class_name"] in CHILDREN:
                children.append(it)

        child_rles = [(c, rle_in(c["mask_rle"]), float(mask_util.area(rle_in(c["mask_rle"]))))
                      for c in children]
        child_assigned = [False] * len(child_rles)

        glomeruli = []
        for g in nsg:
            g_rle = rle_in(g["mask_rle"])
            grade = "normal"
            kids = {"nsg_partiel": 0, "nsg_complet": 0}
            for ci, (c, c_rle, a_ch) in enumerate(child_rles):
                if a_ch <= 0:
                    continue
                inter = float(mask_util.area(mask_util.merge([c_rle, g_rle], intersect=True)))
                if inter / a_ch >= args.cover_thr:
                    child_assigned[ci] = True
                    kids[c["class_name"]] += 1
                    cand = GRADE_OF_CHILD[c["class_name"]]
                    if GRADE_RANK[cand] > GRADE_RANK[grade]:
                        grade = cand
            glomeruli.append({"bbox_global": g["bbox_global"], "score": g.get("score"),
                              "grade": grade, "n_partiel": kids["nsg_partiel"],
                              "n_complet": kids["nsg_complet"]})
            agg[grade] += 1

        n_orphan = sum(1 for a in child_assigned if not a)
        orphan_children += n_orphan
        counts = Counter(g["grade"] for g in glomeruli)
        total = len(glomeruli)
        pct_global = (counts["global"] / total) if total else 0.0
        out = {"source_key": src, "n_glomeruli": total,
               "counts": {"normal": counts["normal"], "segmental": counts["segmental"],
                          "global": counts["global"]},
               "pct_global_sclerosis": round(pct_global, 4),
               "orphan_lesions": n_orphan, "glomeruli": glomeruli}
        json.dump(out, open(out_dir / f"{src}.json", "w"))
        per_source_rows.append((src, total, counts["normal"], counts["segmental"],
                                counts["global"], pct_global))

    total_glom = sum(agg.values())
    pct_global = (agg["global"] / total_glom) if total_glom else 0.0
    pct_scler = ((agg["global"] + agg["segmental"]) / total_glom) if total_glom else 0.0

    print("=" * 74)
    print(f"E7 Stage 13 - glomerulus grading | split={args.split}")
    print(f"  in : {in_dir}")
    print(f"  cover-thr {args.cover_thr}")
    print("=" * 74)
    print(f"{'source':<22}{'glom':>6}{'norm':>6}{'seg':>5}{'glob':>6}{'%global':>9}")
    for src, tot, nm, sg, gl, pg in per_source_rows:
        print(f"{src:<22}{tot:>6}{nm:>6}{sg:>5}{gl:>6}{pg*100:>8.1f}%")
    print("-" * 74)
    print(f"TOTAL glomeruli: {total_glom}  | normal {agg['normal']}  "
          f"segmental {agg['segmental']}  global {agg['global']}")
    print(f"% global sclerosis  = {pct_global*100:.1f}%   (global / total)")
    print(f"% any sclerosis     = {pct_scler*100:.1f}%   ((segmental+global) / total)")
    if orphan_children:
        print(f"NOTE: {orphan_children} lesion(s) had no parent glomerulus (detection gap).")
    summary = {"split": args.split, "cover_thr": args.cover_thr,
               "total_glomeruli": total_glom,
               "counts": {"normal": agg["normal"], "segmental": agg["segmental"],
                          "global": agg["global"]},
               "pct_global_sclerosis": round(pct_global, 4),
               "pct_any_sclerosis": round(pct_scler, 4),
               "orphan_lesions": orphan_children}
    json.dump(summary, open(out_dir / "13_grade_summary.json", "w"), indent=2)
    print(f"\nSaved per-source grades + summary -> {out_dir}")


if __name__ == "__main__":
    main()