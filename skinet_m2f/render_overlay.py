"""
Draw the predictions on the tissue, next to the annotations.

Numbers say that a third of the atrophic tubules are reported as healthy. They
do not say what those tubules look like, and that is the question a pathologist
will ask first. This renders each ROI twice, side by side: the annotations on
the left, the predictions on the right, in the same colours.

With --focus the tubule pair is singled out and everything is coloured by
outcome instead of by class, so the confusion is visible directly:

  green   annotated atrophic, predicted atrophic
  red     annotated atrophic, predicted healthy   <- the failure that matters
  orange  annotated healthy, predicted atrophic
  grey    annotated healthy, predicted healthy
  blue    annotated, nothing predicted

Usage:
  python render_overlay.py --run patient_40k --split test --tag _base050
  python render_overlay.py --run patient_40k --split test --tag _base050 \
      --focus tubule --sources ML7379B3 ML3952B1 EM4831B1
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils

import config as C

COLOR = C.CLASS_COLOR

# Outcome colours for --focus, where the question is not which class a thing is
# but whether the system got it right.
OUTCOME_COLOR = {
    "atrophic_correct": (80, 200, 80),
    "atrophic_as_sain": (50, 50, 235),
    "sain_as_atrophic": (40, 160, 235),
    "sain_correct":     (170, 170, 170),
    "missed":           (220, 140, 60),
}
OUTCOME_LABEL = [
    ("atrophic_correct", "atrophic, found"),
    ("atrophic_as_sain", "atrophic, called non atrophic"),
    ("sain_as_atrophic", "non atrophic, called atrophic"),
    ("sain_correct",     "non atrophic, found"),
    ("missed",           "annotated, nothing predicted"),
]
PAIR = ["tubule_sain", "tubule_atrophique"]


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def gt_instances(xml_path, H, W, wanted):
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
        if label not in wanted:
            continue
        pts = []
        for c in ann.iter():
            if c.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(c.attrib["X"]), float(c.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) >= 3:
            out.append((label, np.array(pts).round().astype(np.int32)))
    return out


def draw_polys(canvas, items, colors, thickness, alpha):
    """Outline plus a light fill, so shapes read without hiding the tissue."""
    if alpha > 0:
        fill = canvas.copy()
        for label, poly in items:
            cv2.fillPoly(fill, [poly], colors[label])
        cv2.addWeighted(fill, alpha, canvas, 1 - alpha, 0, canvas)
    for label, poly in items:
        cv2.polylines(canvas, [poly], True, colors[label], thickness,
                      cv2.LINE_AA)


def draw_masks(canvas, items, colors, thickness, alpha):
    if alpha > 0:
        fill = canvas.copy()
        for label, m in items:
            fill[m > 0] = colors[label]
        cv2.addWeighted(fill, alpha, canvas, 1 - alpha, 0, canvas)
    for label, m in items:
        res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = res[1] if len(res) == 3 else res[0]
        cv2.drawContours(canvas, cnts, -1, colors[label], thickness, cv2.LINE_AA)


def banner(canvas, text, scale, thickness):
    h = int(46 * scale)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], h), (255, 255, 255), -1)
    cv2.putText(canvas, text, (int(14 * scale), int(32 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (30, 30, 30),
                thickness, cv2.LINE_AA)


def legend(canvas, entries, colors, scale, thickness):
    """A white panel in the corner. Drawn over the tissue the text is unreadable
    against a stained background, which is what the earlier version did."""
    if not entries:
        return
    fs = 0.62 * scale
    pad = int(12 * scale)
    row = int(30 * scale)
    sw = int(26 * scale)
    widest = max(cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs,
                                 thickness)[0][0] for _, lab in entries)
    w = pad * 2 + sw + int(12 * scale) + widest
    h = pad * 2 + row * len(entries)

    x0, y0 = pad, int(56 * scale)
    panel = canvas[y0:y0 + h, x0:x0 + w]
    if panel.shape[0] < h or panel.shape[1] < w:
        return
    white = np.full_like(panel, 255)
    cv2.addWeighted(white, 0.88, panel, 0.12, 0, panel)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (150, 150, 150),
                  max(thickness - 1, 1))

    y = y0 + pad + int(row * 0.62)
    for key, label in entries:
        cv2.rectangle(canvas, (x0 + pad, y - int(11 * scale)),
                      (x0 + pad + sw, y + int(5 * scale)), colors[key], -1)
        cv2.rectangle(canvas, (x0 + pad, y - int(11 * scale)),
                      (x0 + pad + sw, y + int(5 * scale)), (110, 110, 110), 1)
        cv2.putText(canvas, label, (x0 + pad + sw + int(12 * scale),
                                    y + int(3 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (25, 25, 25), thickness,
                    cv2.LINE_AA)
        y += row


def scalebar(canvas, scale, thickness, mpp, microns):
    """Every histology figure needs one. Length is computed from the output
    scale and the microns per pixel, so it stays true whatever --scale is."""
    px = int(microns / mpp * scale)
    if px < 20 or px > canvas.shape[1] * 0.5:
        return
    h = int(9 * scale)
    m = int(18 * scale)
    x1, y1 = canvas.shape[1] - m, canvas.shape[0] - m
    x0, y0 = x1 - px, y1 - h
    cv2.rectangle(canvas, (x0 - int(6 * scale), y0 - int(24 * scale)),
                  (x1 + int(6 * scale), y1 + int(6 * scale)), (255, 255, 255), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (20, 20, 20), -1)
    txt = "%d um" % microns
    tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                         thickness)[0][0]
    cv2.putText(canvas, txt, (x0 + (px - tw) // 2, y0 - int(7 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (20, 20, 20),
                thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default="overlays")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="source keys to render; default is all")
    ap.add_argument("--focus", default=None, choices=["tubule"],
                    help="colour by outcome for the tubule pair instead of by class")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict to these classes")
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--mpp", type=float, default=0.227,
                    help="microns per pixel, confirmed for this cohort")
    ap.add_argument("--scalebar", type=int, default=100,
                    help="scale bar length in microns; 0 disables it. Some ROIs "
                         "already carry one burnt into the image, and two bars "
                         "in one figure read as an error.")
    ap.add_argument("--legible", action="store_true",
                    help="lighten the two glomerular subtypes, which the "
                         "clinical palette renders almost identically")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="output scale; 0.5 halves a 3840 px ROI")
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run /
                          (args.split + args.tag) / "filtered")

    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        rows = dict((r["source_key"], r) for r in csv.DictReader(f)
                    if r["split"] == args.split)

    global COLOR
    if args.legible:
        COLOR = C.LEGIBLE_COLOR

    wanted = set(args.classes) if args.classes else set(C.STRUCTURE_CLASSES)
    if args.focus == "tubule":
        wanted = set(PAIR)

    files = sorted(Path(args.in_dir).glob("*.json"))
    if args.sources:
        keep = set(args.sources)
        files = [f for f in files if f.stem in keep]
    if not files:
        raise SystemExit("nothing to render from %s" % args.in_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    th = max(int(3 * args.scale), 1)

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        row = rows.get(rec["source_key"])
        if row is None:
            continue
        img = cv2.imread(row["image_path"])
        if img is None:
            print("[skip] cannot read %s" % row["image_path"])
            continue
        H, W = rec["height"], rec["width"]
        gt = gt_instances(Path(row["xml_path"]), H, W, wanted)
        preds = [(it["class_name"], maskUtils.decode(rle_in(it["mask_rle"])))
                 for it in rec["instances"] if it["class_name"] in wanted]

        left, right = img.copy(), img.copy()

        if args.focus == "tubule":
            # classify every annotation by what the system said about it
            by_class = defaultdict(list)
            for label, m in preds:
                by_class[label].append(maskUtils.encode(np.asfortranarray(m)))
            gt_items, pr_items = [], []
            for label, poly in gt:
                g = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(g, [poly], 1)
                grle = maskUtils.encode(np.asfortranarray(g))
                best = {}
                for c in PAIR:
                    lst = by_class.get(c, [])
                    best[c] = (float(np.max(maskUtils.iou(lst, [grle], [0])))
                               if lst else 0.0)
                same, other = (label, PAIR[0] if label == PAIR[1] else PAIR[1])
                if best[same] < args.iou and best[other] < args.iou:
                    key = "missed"
                elif best[other] > best[same]:
                    key = ("atrophic_as_sain" if label == "tubule_atrophique"
                           else "sain_as_atrophic")
                else:
                    key = ("atrophic_correct" if label == "tubule_atrophique"
                           else "sain_correct")
                gt_items.append((key, poly))
                pr_items.append((key, poly))
            draw_polys(left, gt_items, OUTCOME_COLOR, th, args.alpha)
            draw_masks(right, [(c, m) for c, m in preds], COLOR, th,
                       args.alpha)
        else:
            draw_polys(left, gt, COLOR, th, args.alpha)
            draw_masks(right, preds, COLOR, th, args.alpha)

        if args.scale != 1.0:
            left = cv2.resize(left, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_AREA)

        title = ("%s  what the system said about each annotated tubule"
                 % rec["source_key"] if args.focus
                 else "%s  annotations (%d)" % (rec["source_key"], len(gt)))
        banner(left, title, args.scale * 2, th)
        if not args.focus:
            banner(right, "predictions (%d)" % len(preds), args.scale * 2, th)
        if args.scalebar:
            for panel in (left, right):
                scalebar(panel, args.scale * 2, th, args.mpp, args.scalebar)
        if args.focus == "tubule":
            legend(left, OUTCOME_LABEL, OUTCOME_COLOR, args.scale * 2, th)
        else:
            seen = [c for c in C.STRUCTURE_CLASSES
                    if c in wanted and (any(l == c for l, _ in gt) or
                                        any(l == c for l, _ in preds))]
            legend(left, [(c, C.display(c, short=True)) for c in seen],
                   COLOR, args.scale * 2, th)

        if args.focus:
            # One panel only. In focus mode the left panel is coloured by
            # outcome and the right by class, so red means "atrophic called non
            # atrophic" on one side and "non atrophic tubule" on the other --
            # the same colour carrying opposite meanings within one figure. The
            # left panel already states what the system did with every
            # annotation, so the right adds nothing but that collision.
            sheet = left
        else:
            gap = np.full((left.shape[0], max(int(12 * args.scale), 4), 3), 255,
                          dtype=np.uint8)
            sheet = np.hstack([left, gap, right])
        name = "%s%s.jpg" % (rec["source_key"],
                             "_tubule" if args.focus else "")
        cv2.imwrite(str(out_dir / name), sheet,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("%-16s %4d annotations  %4d predictions" %
              (rec["source_key"], len(gt), len(preds)))

    print("\nwrote %d overlays to %s" % (len(files), out_dir))


if __name__ == "__main__":
    main()