"""
Confusion matrices at two levels: objects and pixels.

They answer different questions and the legacy system reports both. The object
matrix asks what the system said about each annotated structure -- the right
class, another class, or nothing -- and it is what a count-based readout
depends on. The pixel matrix asks the same of every pixel, and it is what an
area-based readout depends on. A system can do well on one and poorly on the
other: many small structures missed barely moves the pixel matrix, while one
large structure mislabelled barely moves the object matrix.

The Background row and column hold what has no counterpart on the other side:
predictions matching no annotation, and annotations matching no prediction.

Written at print size with document fonts, like the other figures, and the
LaTeX block is printed for pasting.

Usage:
  python fig_confusion.py --run patient_100k --split test --tag _base050
  python fig_confusion.py --run patient_100k --split test --tag _base050 --level pixel
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C


def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def gt_of(xml_path, H, W, wanted):
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
        if len(pts) < 3:
            continue
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(pts).round().astype(np.int32)], 1)
        if m.sum() > 0:
            out.append((label, m))
    return out


def set_style(style, base):
    if style == "cm":
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["cmr10", "CMU Serif", "DejaVu Serif"]
        plt.rcParams["mathtext.fontset"] = "cm"
    elif style == "times":
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman",
                                      "DejaVu Serif"]
    plt.rcParams["font.size"] = base
    plt.rcParams["axes.unicode_minus"] = False


def draw(matrix, labels, title, path, args, normalized):
    n = len(labels)
    show = matrix.astype(float)
    if normalized:
        rs = show.sum(axis=1, keepdims=True)
        show = np.divide(show, rs, out=np.zeros_like(show), where=rs > 0)

    set_style(args.style, args.font_size)
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.86))
    im = ax.imshow(show, cmap="Blues", vmin=0,
                   vmax=1.0 if normalized else show.max())

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Annotated")
    if title:
        ax.set_title(title, pad=8)

    hi = show.max() if show.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if v == 0:
                continue
            txt = ("%.2f" % show[i, j] if normalized else
                   ("%.0fk" % (v / 1000.0) if v >= 10000 else "%d" % v))
            ax.text(j, i, txt, ha="center", va="center", fontsize=args.font_size - 2,
                    color="white" if show[i, j] > 0.55 * hi else "black")

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout(pad=0.3)
    fig.savefig(str(path), dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--level", default="both", choices=["object", "pixel", "both"])
    ap.add_argument("--iou", type=float, default=0.30)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict and order the matrix; default is all ten")
    ap.add_argument("--style", default="cm", choices=["cm", "times", "sans"])
    ap.add_argument("--width", type=float, default=5.71)
    ap.add_argument("--font-size", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s07_postproc" / args.run /
                          (args.split + args.tag) / "filtered")
    classes = args.classes or list(C.STRUCTURE_CLASSES)
    labels = [C.display(c, short=True) for c in classes] + ["Background"]
    idx = dict((c, i) for i, c in enumerate(classes))
    bg = len(classes)
    n = bg + 1

    with open(str(work / "s02_split" / "s02_split.csv")) as f:
        xml_of = dict((r["source_key"], r["xml_path"])
                      for r in csv.DictReader(f) if r["split"] == args.split)

    files = sorted(Path(args.in_dir).glob("*.json"))
    if not files:
        raise SystemExit("no stage 07 output in %s" % args.in_dir)

    obj = np.zeros((n, n), dtype=np.int64)
    pix = np.zeros((n, n), dtype=np.int64)

    for pf in files:
        with open(str(pf)) as f:
            rec = json.load(f)
        xmlp = xml_of.get(rec["source_key"])
        if xmlp is None:
            continue
        H, W = rec["height"], rec["width"]
        gt = gt_of(Path(xmlp), H, W, set(classes))
        preds = [(it["class_name"], rle_in(it["mask_rle"]))
                 for it in rec["instances"] if it["class_name"] in idx]

        # ---- object level -------------------------------------------------
        prle = [r for _, r in preds]
        claimed = [False] * len(preds)
        for label, m in gt:
            grle = maskUtils.encode(np.asfortranarray(m))
            if prle:
                ious = np.asarray(maskUtils.iou(prle, [grle], [0])).reshape(len(prle))
                best = int(np.argmax(ious))
                if ious[best] >= args.iou:
                    obj[idx[label], idx[preds[best][0]]] += 1
                    claimed[best] = True
                    continue
            obj[idx[label], bg] += 1          # annotated, nothing predicted
        for k, (label, _) in enumerate(preds):
            if not claimed[k]:
                obj[bg, idx[label]] += 1      # predicted, nothing annotated

        # ---- pixel level --------------------------------------------------
        if args.level in ("pixel", "both"):
            # last write wins where classes overlap; this mirrors how a rendered
            # map is read, and post-processing has already removed most overlap
            g_lab = np.full((H, W), bg, dtype=np.uint8)
            for label, m in gt:
                g_lab[m > 0] = idx[label]
            p_lab = np.full((H, W), bg, dtype=np.uint8)
            for label, r in preds:
                p_lab[maskUtils.decode(r) > 0] = idx[label]
            flat = np.bincount((g_lab.astype(np.int64) * n +
                                p_lab.astype(np.int64)).ravel(),
                               minlength=n * n)
            pix += flat.reshape(n, n)
        print("  %s" % rec["source_key"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s%s" % (args.run, args.split, args.tag)

    wanted = ([("object", obj)] if args.level == "object" else
              [("pixel", pix)] if args.level == "pixel" else
              [("object", obj), ("pixel", pix)])
    for kind, mat in wanted:
        for norm in (False, True):
            suffix = "%s%s" % (kind, "_norm" if norm else "")
            path = out_dir / ("confusion_%s_%s.%s" % (stem, suffix, args.fmt))
            draw(mat, labels, args.title, path, args, norm)
            print("wrote %s" % path)
        np.savetxt(str(out_dir / ("confusion_%s_%s.csv" % (stem, kind))),
                   mat, fmt="%d", delimiter=",",
                   header=",".join(labels), comments="")

    # a short reading of the object matrix, since that is the one a clinician
    # cares about
    print("\n" + "=" * 70)
    print("%-24s%8s%10s%10s" % ("annotated as", "n", "correct", "as other"))
    for i, c in enumerate(classes):
        row = obj[i]
        tot = row.sum()
        if tot == 0:
            continue
        other = tot - row[i] - row[bg]
        print("%-24s%8d%10d%10d" % (C.display(c, short=True), tot, row[i], other))
    print("=" * 70)
    print(r"\begin{figure}[t]\centering")
    print(r"  \includegraphics{figures/confusion_%s_object_norm.%s}" % (stem, args.fmt))
    print(r"  \caption{Object-level confusion at IoU~$\geq$~%.2f on the %s set."
          % (args.iou, args.split))
    print(r"  Rows are annotations, columns predictions; the Background row and")
    print(r"  column hold predictions with no matching annotation and")
    print(r"  annotations with no matching prediction. Rows are normalised.}")
    print(r"  \label{fig:confusion-object}")
    print(r"\end{figure}")


if __name__ == "__main__":
    main()