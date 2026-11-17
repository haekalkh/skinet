"""
Stage 10 - the quantities a pathologist reads.

Stages 07 and 08 stop at counts and pixel areas. Neither is what a report
contains: glomerular density is expressed per square millimetre of cortex, and
interstitial fibrosis and tubular atrophy as a percentage of cortical area. Both
conversions need the pixel size and the cortical area, and both are meaningful
only over the whole biopsy rather than over one sampled field, which is why the
legacy system computes them after merging every region of a case.

Cortical area comes from the region model when the slide was processed with
--predict-cortex, and from the annotation otherwise. Where neither is available
the densities are omitted rather than computed against the whole image, which
would understate them by however much of the field is not cortex.

Usage:
  python s10_morphometry.py --run patient_100k --split test --tag _wsi
  python s10_morphometry.py --run patient_100k --split test --tag _wsi \
      --case-of HI15011717B1=case01
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils

import config as C

# what the report is actually about; the rest is reported but not interpreted
TUBULAR = ["tubule_sain", "tubule_atrophique"]
GLOMERULAR = ["nsg", "nsg_complet", "nsg_partiel"]
SCLEROTIC = ["pac"]


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def cortex_area_from_xml(xml_path, H, W):
    """Area of the annotated cortex, in native pixels."""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return None
    import cv2
    mask = np.zeros((H, W), dtype=np.uint8)
    found = False
    for ann in root.iter():
        if ann.tag.lower().split("}")[-1] != "annotation":
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        if C.canonical_label(raw) != "cortex":
            continue
        pts = []
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts).round().astype(np.int32)], 1)
            found = True
    return float(mask.sum()) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--cortex-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mpp", type=float, default=C.MPP)
    ap.add_argument("--case-of", nargs="+", default=None,
                    metavar="SOURCE=CASE",
                    help="group regions into cases, so that densities are "
                         "computed over the whole biopsy rather than over each "
                         "region separately. Without it every region is its own "
                         "case, which is only correct for a whole slide.")
    args = ap.parse_args()

    work = Path(args.work_dir)
    stem = self_tag = args.split + args.tag
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run / stem / "filtered")
    if args.cortex_dir is None:
        args.cortex_dir = str(work / "s05_infer" / args.run / stem / "cortex")
    if args.out_dir is None:
        args.out_dir = str(work / "s10_morphometry" / args.run / stem)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_of = {}
    for pair in (args.case_of or []):
        k, _, v = pair.partition("=")
        case_of[k] = v

    xml_of = {}
    split_csv = work / "s02_split" / "s02_split.csv"
    if split_csv.exists():
        with open(str(split_csv)) as f:
            for r in csv.DictReader(f):
                xml_of[r["source_key"]] = r["xml_path"]

    files = sorted(Path(args.in_dir).glob("*.json"))
    if not files:
        raise SystemExit("no stage 07 output in %s" % args.in_dir)

    px_mm2 = (args.mpp / 1000.0) ** 2      # one pixel, in mm²
    per_case = defaultdict(lambda: OrderedDict(
        regions=[], cortex_px=0.0, cortex_known=True,
        count=defaultdict(int), area_px=defaultdict(float)))

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        key = rec["source_key"]
        case = case_of.get(key, key)
        agg = per_case[case]
        agg["regions"].append(key)

        cpath = Path(args.cortex_dir) / (key + ".json")
        cortex_px = None
        if cpath.exists():
            with open(str(cpath)) as f:
                cortex_px = float(json.load(f)["area_px"])
        elif key in xml_of:
            cortex_px = cortex_area_from_xml(Path(xml_of[key]),
                                             rec["height"], rec["width"])
        if cortex_px is None:
            agg["cortex_known"] = False
        else:
            agg["cortex_px"] += cortex_px

        for it in rec["instances"]:
            c = it["class_name"]
            agg["count"][c] += 1
            agg["area_px"][c] += float(maskUtils.area(rle_in(it["mask_rle"])))

    rows = []
    for case in sorted(per_case):
        a = per_case[case]
        cortex_mm2 = a["cortex_px"] * px_mm2 if a["cortex_known"] else None

        tub_px = sum(a["area_px"][c] for c in TUBULAR)
        atr_px = a["area_px"]["tubule_atrophique"]
        glom = sum(a["count"][c] for c in GLOMERULAR)
        scl = sum(a["count"][c] for c in SCLEROTIC)

        row = OrderedDict()
        row["case"] = case
        row["regions"] = len(a["regions"])
        row["cortex_mm2"] = round(cortex_mm2, 4) if cortex_mm2 else ""
        row["glomeruli"] = glom
        row["sclerotic_glomeruli"] = scl
        row["sclerotic_pct"] = (round(100.0 * scl / (glom + scl), 1)
                                if glom + scl else "")
        row["glomerular_density_per_mm2"] = (round(glom / cortex_mm2, 2)
                                             if cortex_mm2 else "")
        # a proxy for IFTA: the atrophic share of tubular area. It is not the
        # graded score, which also counts interstitial fibrosis, and it should
        # not be reported as one.
        row["atrophic_pct_of_tubular_area"] = (round(100.0 * atr_px / tub_px, 1)
                                               if tub_px else "")
        row["tubular_pct_of_cortex"] = (
            round(100.0 * tub_px * px_mm2 / cortex_mm2, 1) if cortex_mm2 else "")
        for c in C.STRUCTURE_CLASSES:
            row["n_" + c] = a["count"][c]
            row["mm2_" + c] = round(a["area_px"][c] * px_mm2, 5)
        rows.append(row)

    csv_path = out_dir / "morphometry.csv"
    with open(str(csv_path), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(str(out_dir / "morphometry.json"), "w") as f:
        json.dump(OrderedDict(run=args.run, split=args.split, tag=args.tag,
                              mpp=args.mpp, cases=rows), f, indent=1)

    print("=" * 76)
    print("stage 10 - morphometry | run %s  %s   %.3f um/px"
          % (args.run, stem, args.mpp))
    print("=" * 76)
    print("%-18s%8s%10s%8s%9s%11s%10s"
          % ("case", "regions", "cortex", "glom", "sclerotic",
             "glom/mm2", "atrophic"))
    print("%-18s%8s%10s%8s%9s%11s%10s"
          % ("", "", "mm2", "", "%", "", "% tub area"))
    print("-" * 76)
    for r in rows:
        print("%-18s%8d%10s%8d%9s%11s%10s"
              % (r["case"][:18], r["regions"],
                 r["cortex_mm2"] or "--", r["glomeruli"],
                 r["sclerotic_pct"] if r["sclerotic_pct"] != "" else "--",
                 r["glomerular_density_per_mm2"] or "--",
                 r["atrophic_pct_of_tubular_area"] or "--"))
    print("-" * 76)
    missing = [r["case"] for r in rows if r["cortex_mm2"] == ""]
    if missing:
        print("Cortical area unknown for %d case(s), so densities are omitted"
              % len(missing))
        print("rather than computed against the whole field.")
    print()
    print("The atrophic percentage is the atrophic share of TUBULAR area, not")
    print("a graded IFTA score: the latter also counts interstitial fibrosis,")
    print("which this pipeline does not segment.")
    print()
    print("wrote %s" % csv_path)
    print("=" * 76)


if __name__ == "__main__":
    main()