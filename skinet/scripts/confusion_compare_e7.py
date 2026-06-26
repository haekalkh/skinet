#!/usr/bin/env python
"""
Three-panel confusion-matrix comparison for the seminary:
    [ Mask R-CNN | Mask2Former | Mask2Former + SAM ]
rendered from each config's confusion CSV with a SHARED colour scale, so the
panels are directly comparable.

Input CSVs use the E7 confusion format (same as 08_e7_confusion_matrix_val.csv):
  header  : "" , <class_1..N> , __FN__
  rows    : <gt_class> , counts... , <false-neg count>
  last row: __FP__ , <false-pos per predicted class> , 0
cell[gt][pred] = #GT-of-row-class matched to pred-of-col-class.

Each panel is row-normalised (per GT class -> % distribution across predicted
labels + missed), so the diagonal reads as per-class recall. Restricted by
--classes to the labels the models share (MRCNN MAIN only outputs 6), so the
comparison is apples-to-apples.

Generate the three CSVs first by running the eval/report stage on each prediction
dir, e.g.:
  python 25_results_report_e7.py --split test --pred-dir .../10_filtered        --out-dir report_m2f
  python 25_results_report_e7.py --split test --pred-dir .../12_refined          --out-dir report_m2f_sam
  python 25_results_report_e7.py --split test --pred-dir .../10_filtered_mrcnn   --out-dir report_mrcnn
then point this tool at the three 25_confusion_matrix.csv files.

CPU only.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

COMMON = ["tubule_sain", "tubule_atrophique", "nsg", "pac", "veine", "vaisseau"]


def parse_confusion(path):
    rows = list(csv.reader(open(path)))
    header = rows[0]
    pred_cols = header[1:]                      # classes... + __FN__
    mat = {}
    for r in rows[1:]:
        if not r or r[0] == "":
            continue
        gt = r[0]
        mat[gt] = {pred_cols[i]: float(r[1 + i] or 0) for i in range(len(pred_cols))}
    return pred_cols, mat


def submatrix(pred_cols, mat, classes):
    """Return an (len(classes) x len(classes)+1) row-normalised % matrix:
    columns = classes + 'missed' (__FN__), rows = classes (GT)."""
    cols = classes + ["missed"]
    M = np.zeros((len(classes), len(cols)), float)
    for i, gt in enumerate(classes):
        row = mat.get(gt, {})
        for j, pc in enumerate(classes):
            M[i, j] = row.get(pc, 0.0)
        M[i, -1] = row.get("__FN__", 0.0)
        s = M[i].sum()
        if s > 0:
            M[i] = 100.0 * M[i] / s
    return M, cols


def render(panels, classes, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = classes + ["missed"]
    short = [c.replace("tubule_", "tub_").replace("_", "\n") for c in classes] + ["missed"]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.8), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    im = None
    for ax, (name, M) in zip(axes, panels):
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([c.replace("tubule_", "tub_") for c in classes], fontsize=8)
        ax.set_title(name, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("predicted", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("ground truth", fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if v >= 1:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v < 55 else "black",
                            fontweight="bold" if i == j else "normal")
        # outline the diagonal (recall)
        for i in range(len(classes)):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#39FF14", lw=1.6))
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("% of GT class (row-normalised) → diagonal = recall", fontsize=9)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser("3-panel confusion comparison")
    ap.add_argument("--mrcnn-csv", required=True)
    ap.add_argument("--m2f-csv", required=True)
    ap.add_argument("--m2f-sam-csv", required=True)
    ap.add_argument("--classes", default=",".join(COMMON),
                    help="comma list (default = 6 classes MRCNN and M2F share)")
    ap.add_argument("--title", default="Confusion matrices — Mask R-CNN vs Mask2Former vs M2F+SAM")
    ap.add_argument("--out", default="confusion_compare.png")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    panels = []
    for name, path in [("Mask R-CNN", args.mrcnn_csv),
                       ("Mask2Former", args.m2f_csv),
                       ("Mask2Former + SAM", args.m2f_sam_csv)]:
        pc, mat = parse_confusion(path)
        M, _ = submatrix(pc, mat, classes)
        panels.append((name, M))
        rec = np.diag(M[:, :len(classes)])
        print(f"  {name:<20} per-class recall (diag %): " +
              "  ".join(f"{c[:6]}={r:.0f}" for c, r in zip(classes, rec)))

    render(panels, classes, args.out, args.title)
    print(f"\nwrote -> {args.out}")
    print("NOTE: MRCNN's near-perfect diagonal likely reflects GT-derived predictions")
    print("      (circularity), not genuine accuracy — frame honestly in the deck.")


if __name__ == "__main__":
    main()