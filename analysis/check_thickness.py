"""
How thick are these structures, in pixels, and how many cells is that at the
resolution where Mask2Former actually decides the mask?

The model computes masks on features at stride 4 -- quarter resolution -- and
upsamples afterwards. A ring only a couple of cells thick at that resolution
cannot be drawn accurately no matter how well it is located.

Thickness is estimated as 2 * area / perimeter, which is the correct width for
a long thin ribbon or a narrow annulus. For a compact blob the same formula
returns the radius instead, so the shape column says which reading applies:

  shape ~ 1      compact, treat the number as a radius
  shape >> 1     elongated or annular, treat it as a thickness

Reads the annotation XMLs directly. No GPU, no predictions.

Usage:
  python check_thickness.py
"""

import argparse
import csv
import math
import statistics as stat
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config as C

REPORT = ["intima", "media", "vaisseau", "veine", "nsg",
          "tubule_sain", "tubule_atrophique", "pac"]


def polygons(xml_path):
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
        if label not in REPORT:
            continue
        pts = []
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) >= 3:
            out.append((label, np.array(pts, dtype=np.float32)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv",
                    default=str(Path(C.WORK_DIR) / "s02_split" / "s02_split.csv"))
    ap.add_argument("--stride", type=int, default=4,
                    help="resolution at which the mask is decided")
    ap.add_argument("--out-csv", default="thickness_per_instance.csv")
    args = ap.parse_args()

    with open(args.split_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("empty split csv: %s" % args.split_csv)

    thick = defaultdict(list)
    shape = defaultdict(list)
    lines = ["source,label,area_px,perimeter_px,thickness_px,shape_index"]

    for i, r in enumerate(rows, 1):
        xp = Path(r["xml_path"])
        if not xp.exists():
            continue
        for label, pts in polygons(xp):
            area = abs(cv2.contourArea(pts))
            per = cv2.arcLength(pts, True)
            if area <= 0 or per <= 0:
                continue
            t = 2.0 * area / per
            s = per * per / (4.0 * math.pi * area)     # 1 for a circle
            thick[label].append(t)
            shape[label].append(s)
            lines.append("%s,%s,%.0f,%.1f,%.2f,%.2f"
                         % (r["source_key"], label, area, per, t, s))
        if i % 50 == 0:
            print("  %d/%d slides" % (i, len(rows)))
            sys.stdout.flush()

    Path(args.out_csv).write_text("\n".join(lines))

    print("\n" + "=" * 80)
    print("mask is decided at stride %d, so one cell spans %d full-resolution pixels"
          % (args.stride, args.stride))
    print("=" * 80)
    print("%-20s%7s%9s%9s%9s%9s%8s" %
          ("class", "n", "p25", "median", "p75", "cells", "shape"))
    print("-" * 80)
    for c in REPORT:
        v = thick.get(c, [])
        if not v:
            print("%-20s%7d" % (c, 0))
            continue
        a = np.array(v)
        med = float(np.median(a))
        print("%-20s%7d%9.1f%9.1f%9.1f%9.1f%8.1f"
              % (c, len(v), np.percentile(a, 25), med, np.percentile(a, 75),
                 med / args.stride, stat.median(shape[c])))
    print("-" * 80)
    print("cells  = median thickness divided by the stride")
    print("shape  = 1 is a circle; larger means elongated or ring-like, and only")
    print("         then does the thickness column read as a true width")
    print()
    print("Below about 3 cells the boundary cannot be placed precisely: a")
    print("one-cell error is already a large fraction of the whole structure.")


if __name__ == "__main__":
    main()