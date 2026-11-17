"""
Stage 02 - split the slides into train / val / test.

The split is at SLIDE level, never at tile level: every tile from one slide
lands in exactly one split, so no tile from a training slide can appear in
validation or test.

This reproduces the legacy split exactly -- same seed, same fractions, same
two-step sklearn call on the slides sorted by name -- so the numbers can be
checked against what the old pipeline produced before anything is changed on
purpose.

Outputs (into --out-dir):
  s02_split.csv        one row per slide, with its split and class counts
  s02_summary.json     per-split slide counts and per-class instance counts
  run_meta.json

Usage:
  python s02_split.py
  python s02_split.py --audit-csv <path> --out-dir <dir>
"""

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

from sklearn.model_selection import train_test_split

import config as C


def read_audit(path):
    with open(str(path)) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("audit csv is empty: {}".format(path))
    return rows


def n_structures(row):
    return sum(int(row.get("n_" + c, 0) or 0) for c in C.STRUCTURE_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv",
                    default=str(Path(C.WORK_DIR) / "s01_audit" / "s01_audit.csv"))
    ap.add_argument("--out-dir", default=str(Path(C.WORK_DIR) / "s02_split"))
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_audit(args.audit_csv)
    n_all = len(rows)

    usable = [r for r in rows
              if r["has_image"] == "1" and r["parse_ok"] == "1"
              and n_structures(r) > 0]
    usable.sort(key=lambda r: r["source_key"])

    dropped = n_all - len(usable)
    if len(usable) < 3:
        sys.exit("too few usable slides to split: {}".format(len(usable)))

    train_frac, val_frac, test_frac = C.SPLIT_FRACTIONS
    test_within_rest = test_frac / (val_frac + test_frac)

    print("=" * 70)
    print("stage 02 - slide-level split")
    print("slides in audit : {}".format(n_all))
    print("usable          : {}  (dropped {})".format(len(usable), dropped))
    print("fractions       : train {} / val {} / test {}".format(
        train_frac, val_frac, test_frac))
    print("seed            : {}".format(args.seed))
    print("=" * 70)

    train, rest = train_test_split(
        usable, train_size=train_frac, random_state=args.seed, shuffle=True)
    val, test = train_test_split(
        rest, test_size=test_within_rest, random_state=args.seed, shuffle=True)

    assigned = OrderedDict()
    for name, group in (("train", train), ("val", val), ("test", test)):
        for r in group:
            assigned[r["source_key"]] = name

    if len(assigned) != len(usable):
        sys.exit("a slide landed in more than one split -- refusing to write")

    order = {"train": 0, "val": 1, "test": 2}
    out_rows = sorted(usable,
                      key=lambda r: (order[assigned[r["source_key"]]],
                                     r["source_key"]))

    count_cols = C.STRUCTURE_CLASSES + C.REGION_CLASSES + C.OTHER_CLASSES
    header = (["source_key", "split", "image_path", "xml_path",
               "width", "height", "n_polygons"]
              + ["n_" + c for c in count_cols])

    csv_path = out_dir / "s02_split.csv"
    with open(str(csv_path), "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in out_rows:
            w.writerow([r["source_key"], assigned[r["source_key"]],
                        r["image_path"], r["xml_path"],
                        r.get("width", ""), r.get("height", ""),
                        r.get("n_polygons", "")]
                       + [r.get("n_" + c, 0) for c in count_cols])

    def per_class(split_name):
        d = OrderedDict()
        for c in C.STRUCTURE_CLASSES + C.OTHER_CLASSES:
            d[c] = sum(int(r.get("n_" + c, 0) or 0) for r in out_rows
                       if assigned[r["source_key"]] == split_name)
        return d

    summary = OrderedDict()
    summary["seed"] = int(args.seed)
    summary["fractions"] = list(C.SPLIT_FRACTIONS)
    summary["n_slides_total"] = n_all
    summary["n_slides_usable"] = len(usable)
    summary["n_slides_dropped"] = dropped
    summary["slides_per_split"] = OrderedDict(
        (s, sum(1 for v in assigned.values() if v == s))
        for s in ["train", "val", "test"])
    summary["instances_per_split"] = OrderedDict(
        (s, per_class(s)) for s in ["train", "val", "test"])

    empty = []
    for s in ["train", "val", "test"]:
        for c, n in summary["instances_per_split"][s].items():
            if n == 0:
                empty.append("{} has no {}".format(s, c))
    summary["empty_class_split_pairs"] = empty

    with open(str(out_dir / "s02_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("slides per split: {}".format(dict(summary["slides_per_split"])))
    print("-" * 70)
    print("{:<22}{:>9}{:>9}{:>9}{:>9}".format(
        "class", "train", "val", "test", "total"))
    print("-" * 70)
    for c in C.STRUCTURE_CLASSES + C.OTHER_CLASSES:
        tr = summary["instances_per_split"]["train"][c]
        va = summary["instances_per_split"]["val"][c]
        te = summary["instances_per_split"]["test"][c]
        print("{:<22}{:>9}{:>9}{:>9}{:>9}".format(c, tr, va, te, tr + va + te))
    print("-" * 70)
    tot = sum(summary["instances_per_split"][s][c]
              for s in ["train", "val", "test"] for c in C.STRUCTURE_CLASSES)
    print("{:<22}{:>36}".format("structure total", tot))

    if empty:
        print("\nWARNING - a class is missing from a split entirely:")
        for e in empty:
            print("  " + e)

    print("=" * 70)
    print("wrote {}".format(csv_path))
    C.write_run_meta(out_dir, "s02_split", args, extra=summary)


if __name__ == "__main__":
    main()