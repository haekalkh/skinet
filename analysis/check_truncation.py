"""
How badly is each class truncated by the tile grid?

Reads the COCO files written by stage 03 and reports, per class, how much of
each object was actually present in the tile it was stored in.

The point: an object stored at 20% visibility carries the same label as one
stored whole. For classes whose identity depends on shape, that is direct
label noise.

Usage:
  python check_truncation.py
  python check_truncation.py --splits train
"""

import argparse
import json
import statistics as stat
from collections import defaultdict
from pathlib import Path

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir",
                    default=str(Path(C.WORK_DIR) / "s03_tiles"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    id2name = dict((i + 1, c) for i, c in enumerate(C.STRUCTURE_CLASSES))

    for split in args.splits:
        path = Path(args.tiles_dir) / "annotations" / ("instances_%s.json" % split)
        if not path.exists():
            print("missing: %s" % path)
            continue
        with open(str(path)) as f:
            coco = json.load(f)

        vals = defaultdict(list)
        for a in coco["annotations"]:
            name = id2name.get(a["category_id"])
            if name and "visible_ratio" in a:
                vals[name].append(float(a["visible_ratio"]))

        print("\n" + "=" * 76)
        print("%s  --  %d tiles, %d objects"
              % (split, len(coco["images"]), len(coco["annotations"])))
        print("=" * 76)
        print("%-20s%8s%10s%10s%10s%10s"
              % ("class", "n", "median", "<0.5", "<0.3", "whole"))
        print("-" * 76)

        for c in C.STRUCTURE_CLASSES:
            v = vals.get(c, [])
            if not v:
                print("%-20s%8d" % (c, 0))
                continue
            n = len(v)
            below5 = sum(1 for x in v if x < 0.5) / float(n)
            below3 = sum(1 for x in v if x < 0.3) / float(n)
            whole = sum(1 for x in v if x >= 0.9) / float(n)
            print("%-20s%8d%10.2f%9.0f%%%9.0f%%%9.0f%%"
                  % (c, n, stat.median(v), 100 * below5, 100 * below3,
                     100 * whole))

        allv = [x for v in vals.values() for x in v]
        if allv:
            n = len(allv)
            print("-" * 76)
            print("%-20s%8d%10.2f%9.0f%%%9.0f%%%9.0f%%"
                  % ("ALL", n, stat.median(allv),
                     100 * sum(1 for x in allv if x < 0.5) / float(n),
                     100 * sum(1 for x in allv if x < 0.3) / float(n),
                     100 * sum(1 for x in allv if x >= 0.9) / float(n)))

    print("\n'<0.5' = share of stored objects less than half present in their tile")
    print("'whole' = share at least 90% present")


if __name__ == "__main__":
    main()