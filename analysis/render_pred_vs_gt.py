"""
Look at where the model puts the boundary, against where the annotation puts it.

For each chosen class, crops the tissue around an instance and draws the ground
truth outline in green and the model's outline in red, with their IoU printed on
the cell. Instances are picked to span the whole IoU range, so the sheet shows
the good, the middling and the bad together rather than only the failures.

Three things to look for:

  outlines are the same ring, offset or one thicker  -> boundary placement
  prediction is ragged, broken, or not a ring at all -> shape representation
  the ground truth itself looks arbitrary            -> annotation ceiling, and
                                                        no architecture fixes it

Usage:
  python render_pred_vs_gt.py --run baseline_seed42 --classes intima media
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

import config as C

GT_COLOR = (80, 220, 80)      # BGR
PR_COLOR = (60, 60, 235)


def gt_rle(ann, h, w):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return maskUtils.merge(maskUtils.frPyObjects(seg, h, w))
    if isinstance(seg.get("counts"), list):
        return maskUtils.frPyObjects(seg, h, w)
    return seg


def outline(mask, canvas, x0, y0, colour, thickness):
    res = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                           cv2.CHAIN_APPROX_SIMPLE)
    contours = res[1] if len(res) == 3 else res[0]
    for c in contours:
        c = c.copy()
        c[:, 0, 0] -= x0
        c[:, 0, 1] -= y0
        cv2.drawContours(canvas, [c], -1, colour, thickness)


def make_cell(tile, gt_mask, pr_mask, iou, pad, thumb, label):
    ys, xs = np.where(gt_mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    side = int(max(x1 - x0, y1 - y0) * (1.0 + 2.0 * pad))
    side = max(side, 64)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax0, ay0 = cx - side // 2, cy - side // 2

    H, W = tile.shape[:2]
    canvas = np.full((side, side, 3), 150, dtype=np.uint8)
    sx0, sy0 = max(ax0, 0), max(ay0, 0)
    sx1, sy1 = min(ax0 + side, W), min(ay0 + side, H)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    canvas[sy0 - ay0:sy1 - ay0, sx0 - ax0:sx1 - ax0] = tile[sy0:sy1, sx0:sx1]

    t = max(side // 250, 1)
    outline(gt_mask, canvas, ax0, ay0, GT_COLOR, t)
    if pr_mask is not None:
        outline(pr_mask, canvas, ax0, ay0, PR_COLOR, t)

    cell = cv2.resize(canvas, (thumb, thumb), interpolation=cv2.INTER_AREA)
    txt = "%s  IoU %.2f" % (label, iou) if pr_mask is not None else "%s  no prediction" % label
    for col, th in ((0, 0, 0), 3), ((255, 255, 255), 1):
        cv2.putText(cell, txt, (6, thumb - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, col, th, cv2.LINE_AA)
    return cell


def montage(cells, cols, thumb):
    rows = int(np.ceil(len(cells) / float(cols)))
    sheet = np.full((rows * thumb, cols * thumb, 3), 255, dtype=np.uint8)
    for i, c in enumerate(cells):
        r, q = divmod(i, cols)
        sheet[r * thumb:(r + 1) * thumb, q * thumb:(q + 1) * thumb] = c
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--split", default="val")
    ap.add_argument("--classes", nargs="+", default=["intima", "media"])
    ap.add_argument("--out-dir", default="pred_vs_gt")
    ap.add_argument("--per-class", type=int, default=16)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--thumb", type=int, default=420)
    ap.add_argument("--pad", type=float, default=0.6)
    ap.add_argument("--score", type=float, default=0.3)
    args = ap.parse_args()

    work = Path(args.work_dir)
    tiles = work / "s03_tiles"
    gt_file = tiles / "annotations" / ("instances_%s.json" % args.split)
    pred_file = work / "s04_train" / args.run / "eval" / "coco_instances_results.json"
    for p in (gt_file, pred_file):
        if not p.exists():
            raise SystemExit("not found: %s" % p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(gt_file))
    with open(str(pred_file)) as f:
        preds = [p for p in json.load(f) if p.get("score", 1.0) >= args.score]

    name2id = dict((c, i + 1) for i, c in enumerate(C.STRUCTURE_CLASSES))
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)

    for cname in args.classes:
        cid = name2id.get(cname)
        if cid is None:
            print("unknown class %s" % cname)
            continue

        found = []
        for img_id in coco.getImgIds():
            anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                    if a["category_id"] == cid]
            if not anns:
                continue
            info = coco.loadImgs(img_id)[0]
            h, w = info["height"], info["width"]
            pl = by_img.get(img_id, [])
            p_rles = [p["segmentation"] for p in pl]

            for a in anns:
                g = gt_rle(a, h, w)
                best_i, best_iou = -1, 0.0
                if p_rles:
                    ious = maskUtils.iou(p_rles, [g], [0])
                    ious = np.asarray(ious).reshape(len(p_rles))
                    best_i = int(np.argmax(ious))
                    best_iou = float(ious[best_i])
                found.append((best_iou, img_id, info["file_name"], g,
                              pl[best_i] if best_i >= 0 and best_iou > 0 else None))

        if not found:
            print("no %s in %s" % (cname, args.split))
            continue

        # spread the picks evenly over the IoU range instead of showing only the worst
        found.sort(key=lambda t: t[0])
        n = min(args.per_class, len(found))
        idx = np.linspace(0, len(found) - 1, n).round().astype(int)

        cells = []
        for k in idx:
            iou, img_id, fname, g, pred = found[k]
            tile = cv2.imread(str(tiles / fname))
            if tile is None:
                continue
            gm = maskUtils.decode(g)
            pm = maskUtils.decode(pred["segmentation"]) if pred is not None else None
            pname = (C.STRUCTURE_CLASSES[pred["category_id"] - 1]
                     if pred is not None else "")
            tag = cname if (pname in ("", cname)) else "%s -> %s" % (cname, pname)
            cell = make_cell(tile, gm, pm, iou, args.pad, args.thumb, tag)
            if cell is not None:
                cells.append(cell)

        if not cells:
            print("nothing rendered for %s" % cname)
            continue
        path = out_dir / ("%s.jpg" % cname)
        cv2.imwrite(str(path), montage(cells, args.cols, args.thumb),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        ious = [f[0] for f in found]
        print("%s: %d instances, median IoU %.2f -> %s"
              % (cname, len(found), float(np.median(ious)), path))

    print("\ngreen = annotation, red = model")
    print("cells are ordered worst IoU first, spread across the whole range")


if __name__ == "__main__":
    main()