"""
Count annotated instances per class per split.

Table 3.1 reports these figures, and they change whenever the partition
changes. The counts are of annotation polygons, not tiles: tiles overlap, so a
tile-level count reports the same object several times and is roughly twice as
large.

Usage:
  python count_split_classes.py
  python count_split_classes.py --latex     # rows ready to paste
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        rows = list(csv.DictReader(f))

    counts = defaultdict(lambda: defaultdict(int))
    n_src = defaultdict(int)
    for r in rows:
        n_src[r["split"]] += 1
        try:
            root = ET.parse(r["xml_path"]).getroot()
        except Exception:
            continue
        for ann in root.iter():
            if ann.tag.lower().split("}")[-1] != "annotation":
                continue
            raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
                   or ann.attrib.get("Name") or ann.attrib.get("Group"))
            label = C.canonical_label(raw)
            if label in C.STRUCTURE_CLASSES:
                counts[label][r["split"]] += 1

    splits = ["train", "val", "test"]
    if args.latex:
        for i, c in enumerate(C.STRUCTURE_CLASSES, 1):
            print("    %2d & \\texttt{%s} & %s & %5d & %5d & %5d \\\\"
                  % (i, c.replace("_", "\\_"), C.display(c),
                     counts[c]["train"], counts[c]["val"], counts[c]["test"]))
        tot = [sum(counts[c][s] for c in C.STRUCTURE_CLASSES) for s in splits]
        print("    \\midrule")
        print("    & & Total & %5d & %5d & %5d \\\\" % tuple(tot))
        return

    print("regions per split: " + "  ".join("%s %d" % (s, n_src[s]) for s in splits))
    print("%-22s%8s%8s%8s%9s" % ("class", "train", "val", "test", "total"))
    print("-" * 55)
    grand = 0
    for c in C.STRUCTURE_CLASSES:
        row = [counts[c][s] for s in splits]
        t = sum(row)
        grand += t
        print("%-22s%8d%8d%8d%9d" % (c, row[0], row[1], row[2], t))
    print("-" * 55)
    print("%-22s%8d%8d%8d%9d"
          % ("total", sum(counts[c]["train"] for c in C.STRUCTURE_CLASSES),
             sum(counts[c]["val"] for c in C.STRUCTURE_CLASSES),
             sum(counts[c]["test"] for c in C.STRUCTURE_CLASSES), grand))


if __name__ == "__main__":
    main()