"""
E7 Stage 15 - Overlay predictions onto the full WSI.

Config-driven (imports e7_config). Renders the post-processed instances from any
stage (08 stitched / 09 fused / 10 filtered) back onto the source image -- the
final "whole image + segmented masks" view. CPU only.

Design for huge WSIs (~30720 px): the source is decoded at reduced resolution
(cv2.IMREAD_REDUCED_*) and shown at --long-side; predicted masks are filled
per-class (one RLE union per class, so ~10 decodes per source, not one per
instance); per-instance bboxes + a legend are drawn on top. With --with-gt, GT
polygons are drawn as outlines (scaled directly from the XML, no rasterisation)
so you can eyeball pred-fill vs GT-outline.

Input : OUT_MAIN/postproc/{split}/{08_stitched|09_fused|10_filtered}/{source}.json
        + source images (via split CSV)  [+ XML for --with-gt]
Output: OUT_MAIN/postproc/{split}/15_overlays/{stage}/{source}.png
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

import e7_config as C


STAGE_DIR = {"08": "08_stitched", "09": "09_fused", "10": "10_filtered"}

PALETTE = [
    (66, 135, 245), (245, 66, 66), (66, 245, 96), (245, 212, 66),
    (200, 66, 245), (66, 245, 233), (245, 144, 66), (150, 150, 150),
    (245, 66, 158), (120, 200, 60),
]
CLASS_COLOR = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(C.MAIN_CLASSES)}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rle_in(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def reduced_flag(factor):
    return {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[factor]


def read_display(path, H, W, long_side):
    long = max(H, W)
    factor = 1
    while factor < 8 and long // (factor * 2) >= long_side:
        factor *= 2
    img = cv2.imread(str(path), reduced_flag(factor))
    if img is None:
        return None, None
    ratio = long_side / float(long) if long > long_side else 1.0
    disp = cv2.resize(img, (max(1, round(W * ratio)), max(1, round(H * ratio))),
                      interpolation=cv2.INTER_AREA)
    return disp, ratio


def parse_gt_polys(xml_path):
    out = defaultdict(list)
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out
    for ann in root.iter():
        if "annotation" not in ann.tag.lower():
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = C.normalize_label(raw)
        if label not in C.MAIN_CLASSES:
            continue
        pts = []
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) >= 3:
            out[label].append(np.array(pts, dtype=np.float64))
    return out


def draw_legend(disp, counts):
    x, y = 10, 10
    for c in C.MAIN_CLASSES:
        col = CLASS_COLOR[c]
        cv2.rectangle(disp, (x, y), (x + 18, y + 18), col, -1)
        cv2.rectangle(disp, (x, y), (x + 18, y + 18), (30, 30, 30), 1)
        cv2.putText(disp, f"{c} ({counts.get(c, 0)})", (x + 24, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        y += 24


def render_source(rec, image_path, long_side, alpha, with_gt, xml_path):
    H, W = rec["height"], rec["width"]
    disp, ratio = read_display(image_path, H, W, long_side)
    if disp is None:
        return None
    dh, dw = disp.shape[:2]

    # group instances by class
    by_class = defaultdict(list)
    for it in rec["instances"]:
        by_class[it["class_name"]].append(it)

    # filled color layer: one union-decode per class (memory-light)
    color_layer = disp.copy()
    counts = {}
    for c, items in by_class.items():
        counts[c] = len(items)
        union = mask_util.merge([rle_in(it["mask_rle"]) for it in items],
                                intersect=False)
        full = mask_util.decode(union)                      # [H, W]
        small = cv2.resize(full, (dw, dh), interpolation=cv2.INTER_NEAREST)
        color_layer[small > 0] = CLASS_COLOR[c]
        del full, small
    out = cv2.addWeighted(color_layer, alpha, disp, 1 - alpha, 0)

    # per-instance bbox (crisp, on top)
    for c, items in by_class.items():
        col = CLASS_COLOR[c]
        for it in items:
            x, y, w, h = [v * ratio for v in it["bbox_global"]]
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), col, 1)

    # GT outlines
    if with_gt and xml_path is not None:
        for c, polys in parse_gt_polys(xml_path).items():
            col = CLASS_COLOR[c]
            for poly in polys:
                pts = (poly * ratio).round().astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(out, [pts], True, col, 2, cv2.LINE_AA)

    draw_legend(out, counts)
    note = "fill+bbox = prediction" + ("   outline = GT" if with_gt else "")
    cv2.putText(out, note, (10, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--stage", default="10", choices=["08", "09", "10"])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--long-side", type=int, default=2048)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--with-gt", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else \
        (C.OUT_MAIN / "postproc" / args.split / STAGE_DIR[args.stage])
    out_dir = Path(args.out_dir) if args.out_dir else \
        (C.OUT_MAIN / "postproc" / args.split / "15_overlays" / STAGE_DIR[args.stage])
    ensure_dir(out_dir)

    df = pd.read_csv(args.split_csv)
    img_of = dict(zip(df["source_key"], df["image_path"]))
    xml_of = dict(zip(df["source_key"], df["xml_path"]))

    files = sorted(f for f in in_dir.glob("*.json")
                   if not f.name[:2] in ("08", "09", "10"))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError(f"No instance JSONs in {in_dir}")

    print("=" * 80)
    print(f"E7 Stage 15 - Overlay | split={args.split} stage={args.stage} "
          f"long_side={args.long_side} with_gt={args.with_gt}")
    print(f"  in : {in_dir}\n  out: {out_dir}")
    print("=" * 80)

    n_ok = 0
    for pf in files:
        with open(pf) as f:
            rec = json.load(f)
        src = rec["source_key"]
        ipath = img_of.get(src)
        if ipath is None:
            print(f"[WARN] no image path for {src}")
            continue
        img = render_source(rec, ipath, args.long_side, args.alpha,
                            args.with_gt, xml_of.get(src))
        if img is None:
            print(f"[WARN] could not render {src} (image unreadable)")
            continue
        outp = out_dir / f"{src}.png"
        cv2.imwrite(str(outp), img)
        n_ok += 1
        print(f"  {src}: {rec.get('num_after_orphan', len(rec['instances']))} "
              f"instances -> {outp.name}")

    print("\n" + "=" * 80)
    print(f"Rendered {n_ok}/{len(files)} overlays to {out_dir}")


if __name__ == "__main__":
    main()