"""
Draw the pipeline's own output on the image it came from.

render_wsi_overlay.py runs the model itself and paints tile by tile, which is
useful for a quick look but is not what the pipeline produces: it skips fusion,
filtering and the priority rules entirely. This reads the stage 07 output
instead, so what appears is exactly the set of objects the system reports.

Instances are decoded one at a time and released. A whole-slide mask is 534
megapixels, so holding several at once is not possible; holding one at a time
is, and the decode is C so the cost is bearable.

Usage:
  python render_from_stage07.py --run patient_100k --split test --tag _wsi
  python render_from_stage07.py --run patient_100k --split test --tag _wsi \
      --crop 12000 6000 3000 2000 --scale 1.0
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils

try:
    from tqdm import tqdm
except ImportError:      # not worth a hard dependency for a progress bar
    def tqdm(it, **kw):
        return it

import config as C


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def legend(canvas, entries, colors, scale, thickness):
    if not entries:
        return
    fs = 0.6 * scale
    pad, row, sw = int(12 * scale), int(28 * scale), int(24 * scale)
    widest = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                 thickness)[0][0] for _, t in entries)
    w = pad * 2 + sw + int(12 * scale) + widest
    h = pad * 2 + row * len(entries)
    x0, y0 = pad, pad
    if canvas.shape[0] < y0 + h or canvas.shape[1] < x0 + w:
        return
    panel = canvas[y0:y0 + h, x0:x0 + w]
    cv2.addWeighted(np.full_like(panel, 255), 0.88, panel, 0.12, 0, panel)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (150, 150, 150),
                  max(thickness - 1, 1))
    y = y0 + pad + int(row * 0.6)
    for k, text in entries:
        cv2.rectangle(canvas, (x0 + pad, y - int(10 * scale)),
                      (x0 + pad + sw, y + int(4 * scale)), colors[k], -1)
        cv2.rectangle(canvas, (x0 + pad, y - int(10 * scale)),
                      (x0 + pad + sw, y + int(4 * scale)), (110, 110, 110), 1)
        cv2.putText(canvas, text, (x0 + pad + sw + int(12 * scale),
                                   y + int(2 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (25, 25, 25), thickness,
                    cv2.LINE_AA)
        y += row


def scalebar(canvas, scale, thickness, mpp, microns):
    px = int(microns / mpp * scale)
    if px < 20 or px > canvas.shape[1] * 0.5:
        return
    h, m = int(8 * scale), int(16 * scale)
    x1, y1 = canvas.shape[1] - m, canvas.shape[0] - m
    x0, y0 = x1 - px, y1 - h
    cv2.rectangle(canvas, (x0 - int(6 * scale), y0 - int(22 * scale)),
                  (x1 + int(6 * scale), y1 + int(6 * scale)), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (20, 20, 20), -1)
    txt = "%d um" % microns
    tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale,
                         thickness)[0][0]
    cv2.putText(canvas, txt, (x0 + (px - tw) // 2, y0 - int(6 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (20, 20, 20),
                thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--image", default=None,
                    help="the source image; for a slide outside the split, "
                         "which the split csv cannot supply")
    ap.add_argument("--out-dir", default="stage07_overlays")
    ap.add_argument("--sources", nargs="+", default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--crop", nargs=4, type=int, default=None,
                    metavar=("X", "Y", "W", "H"))
    ap.add_argument("--scale", type=float, default=0.125)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--outline-width", type=int, default=0,
                    help="contour thickness in canvas pixels; 0 draws none. A "
                         "one-pixel contour on a canvas downscaled eightfold "
                         "stands for eight native pixels, which is why it reads "
                         "as heavy at low scale.")
    ap.add_argument("--mpp", type=float, default=0.227)
    ap.add_argument("--scalebar", type=int, default=500)
    ap.add_argument("--legible", action="store_true",
                    help="lighten the two glomerular subtypes, which the "
                         "clinical palette renders almost identically")
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run /
                          (args.split + args.tag) / "filtered")
    colors = C.LEGIBLE_COLOR if args.legible else C.CLASS_COLOR

    img_of = {}
    if args.image:
        img_of[Path(args.image).stem] = args.image
    else:
        with open(str(work / "s02_split" / "s02_split.csv")) as f:
            for r in csv.DictReader(f):
                img_of[r["source_key"]] = r["image_path"]

    files = sorted(Path(args.in_dir).glob("*.json"))
    if args.sources:
        keep = set(args.sources)
        files = [f for f in files if f.stem in keep]
    if not files:
        raise SystemExit("nothing to render from %s" % args.in_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    th = max(int(2 * args.scale * 2), 1)

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        ip = img_of.get(rec["source_key"])
        if ip is None:
            print("[skip] no image for %s" % rec["source_key"])
            continue
        img = cv2.imread(str(ip))
        if img is None:
            print("[skip] cannot read %s" % ip)
            continue

        H, W = rec["height"], rec["width"]
        cx, cy, cw, ch = args.crop if args.crop else (0, 0, W, H)
        cw, ch = min(cw, W - cx), min(ch, H - cy)
        ow, oh = int(cw * args.scale), int(ch * args.scale)

        base = cv2.resize(img[cy:cy + ch, cx:cx + cw], (ow, oh),
                          interpolation=cv2.INTER_AREA)
        del img
        paint = np.zeros_like(base)
        edges = np.zeros_like(base)
        painted = np.zeros((oh, ow), dtype=bool)

        counts = Counter()
        # the decode dominates, and there are hundreds of instances per slide,
        # so the bar is the only indication that anything is happening
        drawable = [it for it in rec["instances"]
                    if it["class_name"] in colors
                    and not (args.only and it["class_name"] not in args.only)]
        for it in tqdm(drawable, desc=rec["source_key"], unit="obj",
                       ncols=76, leave=True):
            name = it["class_name"]
            rle = rle_in(it["mask_rle"])
            bx, by, bw, bh = [int(v) for v in maskUtils.toBbox(rle)]
            if bx + bw <= cx or by + bh <= cy or bx >= cx + cw or by >= cy + ch:
                continue

            # decoded one at a time; a whole-slide mask is 534 Mpx and several
            # cannot be held at once
            m = maskUtils.decode(rle)[by:by + bh, bx:bx + bw]

            sx0, sy0 = int((bx - cx) * args.scale), int((by - cy) * args.scale)
            sw_, sh_ = max(int(bw * args.scale), 1), max(int(bh * args.scale), 1)
            if sx0 >= ow or sy0 >= oh or sx0 + sw_ <= 0 or sy0 + sh_ <= 0:
                del m
                continue
            small = cv2.resize(m, (sw_, sh_), interpolation=cv2.INTER_NEAREST)
            del m

            ax0, ay0 = max(sx0, 0), max(sy0, 0)
            ax1, ay1 = min(sx0 + sw_, ow), min(sy0 + sh_, oh)
            sub = small[ay0 - sy0:ay1 - sy0, ax0 - sx0:ax1 - sx0].astype(bool)
            if not sub.any():
                continue

            paint[ay0:ay1, ax0:ax1][sub] = colors[name]
            painted[ay0:ay1, ax0:ax1] |= sub
            if args.outline_width > 0:
                res = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
                cnts = res[1] if len(res) == 3 else res[0]
                cv2.drawContours(edges[ay0:ay1, ax0:ax1], cnts, -1,
                                 colors[name], args.outline_width, cv2.LINE_AA)
            counts[name] += 1

        canvas = base.copy()
        canvas[painted] = (args.alpha * paint[painted] +
                           (1 - args.alpha) * base[painted]).astype(np.uint8)
        if args.outline_width > 0:
            lit = edges.any(axis=2)
            canvas[lit] = edges[lit]

        entries = [(c, "%s  (%d)" % (C.display(c, short=True), counts[c]))
                   for c in C.STRUCTURE_CLASSES if counts.get(c)]
        legend(canvas, entries, colors, args.scale * 4, th)
        if args.scalebar:
            scalebar(canvas, args.scale * 4, th, args.mpp, args.scalebar)

        name = rec["source_key"] + ("_crop" if args.crop else "") + ".jpg"
        cv2.imwrite(str(out_dir / name), canvas,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print("%-16s %4d of %d objects drawn   %s" %
              (rec["source_key"], sum(counts.values()), len(drawable),
               "  ".join("%s %d" % (c, counts[c])
                         for c in C.STRUCTURE_CLASSES if counts.get(c))))

    print("\nwrote to %s" % out_dir)
    print("These are the objects the pipeline reports, after fusion, filtering")
    print("and the priority rules -- not raw tile predictions.")


if __name__ == "__main__":
    main()