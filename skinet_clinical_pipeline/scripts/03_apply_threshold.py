#!/usr/bin/env python
"""
Terapkan per-class score threshold (val-tuned) ke prediksi stitched
SEBELUM 10b post-processing. Menggabungkan threshold tuning (yg hasilkan
0.723 global) dengan cortex gating (10b).

Baca postfusion_threshold.json (per-kelas), buang instance score < threshold
kelasnya, tulis ke folder baru. Lalu 10b jalan di output ini.

  python apply_threshold.py \
    --in-dir  .../postproc/test/08_stitched \
    --out-dir .../postproc/test/08b_thresholded \
    --thr-json .../postproc/val/postfusion_threshold.json \
    [--sources ML9360N14]

CPU stage.
"""
import argparse
import json
import glob
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thr-json", required=True)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--default-thr", type=float, default=0.5,
                    help="threshold utk kelas yg tak ada di json")
    args = ap.parse_args()

    thr = json.load(open(args.thr_json))
    print(f"[info] threshold per-kelas: {thr}")

    ind = Path(args.in_dir)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    files = sorted(ind.glob("*.json"))
    files = [f for f in files if "summary" not in f.name
             and "manifest" not in f.name and not f.name[0].isdigit()]
    if args.sources:
        want = set(args.sources)
        files = [f for f in files if f.stem in want]

    tot_before = tot_after = 0
    for f in files:
        d = json.load(open(f))
        insts = d.get("instances", [])
        kept = []
        for it in insts:
            t = thr.get(it["class_name"], args.default_thr)
            if it.get("score", 1.0) >= t:
                kept.append(it)
        d["instances"] = kept
        d["num_instances"] = len(kept)
        json.dump(d, open(outd / f.name, "w"))
        tot_before += len(insts)
        tot_after += len(kept)
        print(f"  {d.get('source_key', f.stem)}: {len(insts)} -> {len(kept)}")

    print(f"[done] {len(files)} source | total {tot_before} -> {tot_after} "
          f"(buang {tot_before - tot_after}) -> {outd}")


if __name__ == "__main__":
    main()
