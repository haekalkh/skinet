"""
Stage 01 - inventory the raw data.

Pairs every annotation XML with its slide image, parses the annotations, and
writes one row per slide. Nothing is filtered or modified here; this stage only
reports what is on disk, so later stages start from a checked foundation.

Outputs (into --out-dir):
  s01_audit.csv        one row per slide, with per-class polygon counts
  s01_summary.json     dataset totals, and any labels not recognised
  run_meta.json        what produced these files

Usage:
  python s01_audit.py
  python s01_audit.py --img-dir <dir> --xml-dir <dir> --out-dir <dir>
"""

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import config as C


def image_size(path):
    """Width, height without decoding the whole slide when possible."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(str(path)) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None, None
        return int(img.shape[1]), int(img.shape[0])
    except Exception:
        return None, None


def parse_xml(xml_path):
    """Return (counts_by_label, n_polygons_total, unknown_labels, ok)."""
    counts = defaultdict(int)
    unknown = defaultdict(int)
    total = 0
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception as e:
        print("[parse failed] {}: {}".format(xml_path.name, e))
        return counts, 0, unknown, False

    # Match only real <Annotation> elements. The wrapper tags
    # <ASAP_Annotations> and <Annotations> also contain the word "annotation"
    # and hold every coordinate in the file, so a loose match counts each file
    # twice as one giant unlabelled polygon.
    for ann in root.iter():
        if ann.tag.lower().split("}")[-1] != "annotation":
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = C.canonical_label(raw)

        n_pts = 0
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                n_pts += 1
        if n_pts < 3:
            continue

        total += 1
        if label is None:
            counts["_no_label"] += 1
        elif label in C.ALL_KNOWN:
            counts[label] += 1
        else:
            counts["_unknown"] += 1
            unknown[label] += 1

    return counts, total, unknown, True


def find_image(img_dir, stem):
    for ext in C.IMAGE_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=C.RAW_IMG_DIR)
    ap.add_argument("--xml-dir", default=C.RAW_XML_DIR)
    ap.add_argument("--out-dir", default=str(Path(C.WORK_DIR) / "s01_audit"))
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N slides")
    ap.add_argument("--no-size", action="store_true",
                    help="skip reading image dimensions (much faster)")
    args = ap.parse_args()

    img_dir, xml_dir = Path(args.img_dir), Path(args.xml_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not xml_dir.is_dir():
        sys.exit("xml-dir does not exist: {}".format(xml_dir))
    if not img_dir.is_dir():
        sys.exit("img-dir does not exist: {}".format(img_dir))

    xmls = sorted(xml_dir.glob("*.xml"))
    if not xmls:
        sys.exit("no .xml files under {}".format(xml_dir))
    if args.limit > 0:
        xmls = xmls[:args.limit]

    print("=" * 70)
    print("stage 01 - audit")
    print("images : {}".format(img_dir))
    print("xml    : {}  ({} files)".format(xml_dir, len(xmls)))
    print("=" * 70)

    count_cols = (C.STRUCTURE_CLASSES + C.REGION_CLASSES + C.OTHER_CLASSES
                  + ["_no_label", "_unknown"])
    header = (["source_key", "image_path", "xml_path", "has_image", "parse_ok",
               "width", "height", "n_polygons"]
              + ["n_" + c for c in count_cols])

    rows = []
    totals = defaultdict(int)
    unknown_all = defaultdict(int)
    n_missing_img = 0
    n_parse_fail = 0

    for i, xp in enumerate(xmls, 1):
        stem = xp.stem
        ip = find_image(img_dir, stem)
        counts, total, unknown, ok = parse_xml(xp)

        if ip is None:
            n_missing_img += 1
        if not ok:
            n_parse_fail += 1
        for k, v in unknown.items():
            unknown_all[k] += v
        for k, v in counts.items():
            totals[k] += v

        w = h = ""
        if ip is not None and not args.no_size:
            ww, hh = image_size(ip)
            w = "" if ww is None else ww
            h = "" if hh is None else hh

        row = [stem, "" if ip is None else str(ip), str(xp),
               int(ip is not None), int(ok), w, h, total]
        row += [counts.get(c, 0) for c in count_cols]
        rows.append(row)

        if i % 25 == 0 or i == len(xmls):
            print("  {}/{} slides".format(i, len(xmls)))
            sys.stdout.flush()

    csv_path = out_dir / "s01_audit.csv"
    with open(str(csv_path), "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    usable = sum(1 for r in rows if r[3] == 1 and r[4] == 1
                 and sum(r[8:8 + len(C.STRUCTURE_CLASSES)]) > 0)

    summary = OrderedDict()
    summary["n_xml_files"] = len(xmls)
    summary["n_missing_image"] = n_missing_img
    summary["n_parse_failed"] = n_parse_fail
    summary["n_usable_slides"] = usable
    summary["structure_totals"] = OrderedDict(
        (c, int(totals.get(c, 0))) for c in C.STRUCTURE_CLASSES)
    summary["region_totals"] = OrderedDict(
        (c, int(totals.get(c, 0))) for c in C.REGION_CLASSES)
    summary["other_totals"] = OrderedDict(
        (c, int(totals.get(c, 0))) for c in C.OTHER_CLASSES)
    summary["polygons_without_label"] = int(totals.get("_no_label", 0))
    cortex_idx = 8 + len(C.STRUCTURE_CLASSES) + C.REGION_CLASSES.index("cortex")
    no_cortex = [r[0] for r in rows if r[3] == 1 and r[cortex_idx] == 0]
    summary["n_slides_without_cortex_polygon"] = len(no_cortex)
    summary["slides_without_cortex_polygon"] = no_cortex
    summary["unrecognised_labels"] = OrderedDict(
        sorted(unknown_all.items(), key=lambda kv: -kv[1]))

    with open(str(out_dir / "s01_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("slides with an xml        : {}".format(len(xmls)))
    print("without a matching image  : {}".format(n_missing_img))
    print("xml failed to parse       : {}".format(n_parse_fail))
    print("usable (image + structures): {}".format(usable))
    print("-" * 70)
    print("{:<22}{:>10}".format("structure class", "polygons"))
    for c in C.STRUCTURE_CLASSES:
        print("{:<22}{:>10}".format(c, totals.get(c, 0)))
    print("-" * 70)
    for c in C.REGION_CLASSES + C.OTHER_CLASSES:
        print("{:<22}{:>10}".format(c, totals.get(c, 0)))
    print("{:<22}{:>10}".format("no label", totals.get("_no_label", 0)))
    print("{:<22}{:>10}".format("unrecognised", totals.get("_unknown", 0)))
    print("-" * 70)
    print("slides with no cortex polygon: {}".format(
        summary["n_slides_without_cortex_polygon"]))

    if unknown_all:
        print("\nlabels not recognised (add to LABEL_ALIASES in config.py):")
        for k, v in sorted(unknown_all.items(), key=lambda kv: -kv[1]):
            print("  {:<28}{}".format(k, v))

    print("=" * 70)
    print("wrote {}".format(csv_path))
    C.write_run_meta(out_dir, "s01_audit", args, extra=summary)


if __name__ == "__main__":
    main()