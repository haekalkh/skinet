"""
Is the 1920-wide group a different magnification, or just a smaller field?

Compares the typical polygon area of each class between image-width groups.

  same magnification  -> areas match across groups
  2x magnification    -> areas differ by about 4x

Uses tubule_sain, which has thousands of instances, so the answer does not
rest on a handful of glomeruli.

Usage:
  python check_scale.py
"""

import argparse
import csv
import statistics as stat
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config as C

REPORT = ["tubule_sain", "tubule_atrophique", "nsg", "vaisseau", "intima"]


def polygon_areas(xml_path):
    out = defaultdict(list)
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
        if len(pts) < 3:
            continue
        a = abs(cv2.contourArea(np.array(pts, dtype=np.float32)))
        if a > 0:
            out[label].append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv",
                    default=str(Path(C.WORK_DIR) / "s02_split" / "s02_split.csv"))
    args = ap.parse_args()

    with open(args.split_csv) as f:
        rows = list(csv.DictReader(f))

    by_width = defaultdict(lambda: defaultdict(list))
    slides = defaultdict(int)

    for i, r in enumerate(rows, 1):
        try:
            width = int(r["width"])
        except (KeyError, ValueError):
            continue
        group = width if width in (1920, 3840) else "other"
        slides[group] += 1
        for label, areas in polygon_areas(Path(r["xml_path"])).items():
            by_width[group][label].extend(areas)
        if i % 50 == 0:
            print("  {}/{} slides".format(i, len(rows)))

    groups = [g for g in (1920, 3840, "other") if slides.get(g)]

    print("\n" + "=" * 72)
    print("slides per group: " + ", ".join(
        "{}={}".format(g, slides[g]) for g in groups))
    print("=" * 72)
    print("{:<20}".format("class")
          + "".join("{:>17}".format("width " + str(g)) for g in groups)
          + "{:>10}".format("ratio"))
    print("-" * 72)

    for label in REPORT:
        cells, meds = [], {}
        for g in groups:
            vals = by_width[g].get(label, [])
            if vals:
                meds[g] = stat.median(vals)
                cells.append("{:>10.0f} (n{:>4})".format(meds[g], len(vals)))
            else:
                cells.append("{:>17}".format("-"))
        ratio = ""
        if 1920 in meds and 3840 in meds and meds[3840] > 0:
            ratio = "{:>10.2f}".format(meds[1920] / meds[3840])
        print("{:<20}".format(label) + "".join(cells) + ratio)

    print("-" * 72)
    print("ratio is median area at width 1920 divided by the same at 3840")
    print("about 1.0  -> same magnification, the 1920 images are just a")
    print("              narrower field of view")
    print("about 4.0  -> the 1920 images are twice the magnification")
    print("about 0.25 -> the 3840 images are twice the magnification")
    print("=" * 72)


if __name__ == "__main__":
    main()