#!/usr/bin/env python
"""
Statistik final ala MRCNN untuk output M2F (Cara B: WSI-normal + cortex gating).

Per WSI:
  - count + area per kelas struktur (dari 10b output, sudah cortex-gated)
  - fuse_base_class: inject CORTEX sebagai 1 instance (area dari cortex mask)
  - density = count / cortex_area (per mm^2 bila mpp diberikan; else per px)
    ala Marechal et al. (glomerular density per cortical area)
Bandingkan vs MRCNN _stats.json bila tersedia.

  python compute_stats.py \
    --gated-dir .../10b_cortex_gated \
    --cortex-dir .../cortex_masks_main/test \
    --mrcnn-dir  .../results/test_main_.../  \
    --out .../stats_m2f.json \
    [--mpp 0.0]      # microns-per-pixel; 0 -> density per px

CPU stage.
"""
import argparse
import json
import glob
import os
from collections import Counter

import numpy as np
from pycocotools import mask as mask_util

CLASSES = ["tubule_sain", "tubule_atrophique", "nsg_complet", "nsg_partiel",
           "pac", "vaisseau", "veine", "nsg", "intima", "media"]


def area_from_instances(instances):
    """Sum area (px) per kelas dari RLE mask."""
    area = Counter()
    count = Counter()
    for it in instances:
        r = it["mask_rle"]
        if isinstance(r["counts"], str):
            r = {"size": r["size"], "counts": r["counts"].encode("ascii")}
        a = int(mask_util.area(r))
        area[it["class_name"]] += a
        count[it["class_name"]] += 1
    return count, area


def cortex_area_px(cortex_png, cortex_label=1):
    import cv2
    m = cv2.imread(cortex_png, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return 0
    return int((m == cortex_label).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated-dir", required=True, help="output 10b (cortex-gated)")
    ap.add_argument("--cortex-dir", required=True, help="cortex mask PNG per source")
    ap.add_argument("--mrcnn-dir", default=None, help="MRCNN results (utk banding)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mpp", type=float, default=0.0,
                    help="microns-per-pixel; 0 -> density per px^2")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.gated_dir, "*.json")))
    files = [f for f in files if not os.path.basename(f).startswith(("10", "0"))
             and "summary" not in f and "manifest" not in f]

    px2mm2 = (args.mpp * 1e-3) ** 2 if args.mpp > 0 else None  # px -> mm^2
    all_stats = {}
    agg_m2f = Counter(); agg_mrcnn = Counter()

    for f in files:
        src = os.path.basename(f).replace(".json", "")
        d = json.load(open(f))
        count, area = area_from_instances(d["instances"])

        # fuse_base_class: cortex sebagai 1 instance
        cpng = os.path.join(args.cortex_dir, f"{src}.png")
        cortex_px = cortex_area_px(cpng)

        # denominator utk density
        denom = cortex_px * px2mm2 if px2mm2 else cortex_px
        unit = "per_mm2" if px2mm2 else "per_px"

        rec = {"source": src, "cortex_area_px": cortex_px, "structures": {}}
        for c in CLASSES:
            cnt = count.get(c, 0)
            ar = area.get(c, 0)
            dens = (cnt / denom) if denom > 0 else 0.0
            rec["structures"][c] = {"count": cnt, "area_px": ar,
                                    f"density_{unit}": dens}
            agg_m2f[c] += cnt

        # banding MRCNN
        if args.mrcnn_dir:
            sf = os.path.join(args.mrcnn_dir, src, f"{src}_stats.json")
            if os.path.exists(sf):
                mr = json.load(open(sf))
                rec["mrcnn"] = {c: mr[c]["count"] for c in CLASSES if c in mr}
                for c in CLASSES:
                    agg_mrcnn[c] += rec["mrcnn"].get(c, 0)

        all_stats[src] = rec

    out = {"n_sources": len(all_stats), "unit": unit,
           "mpp": args.mpp, "per_source": all_stats,
           "aggregate": {
               "m2f": dict(agg_m2f),
               "mrcnn": dict(agg_mrcnn) if args.mrcnn_dir else None,
           }}
    with open(args.out, "w") as fo:
        json.dump(out, fo, indent=2)

    # ringkasan
    print(f"[done] {len(all_stats)} WSI -> {args.out}")
    print(f"{'kelas':18s} {'M2F':>6s} {'MRCNN':>6s}")
    print("-" * 34)
    for c in CLASSES:
        print(f"{c:18s} {agg_m2f[c]:6d} {agg_mrcnn.get(c,0):6d}")
    print("-" * 34)
    print(f"{'TOTAL':18s} {sum(agg_m2f.values()):6d} {sum(agg_mrcnn.values()):6d}")


if __name__ == "__main__":
    main()
