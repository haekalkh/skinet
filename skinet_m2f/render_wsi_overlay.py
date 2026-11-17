"""
Run the model across a whole slide and paint the result as one picture.

The regular pipeline cannot be used here. Stage 06 builds a full-image canvas
per instance, which at 534 megapixels is impossible, and the fusion and
filtering stages exist to produce countable objects rather than a picture. For a
figure none of that is needed: each tile's predictions are painted straight onto
a downscaled canvas as they are produced, and nothing is held in memory but the
canvas itself.

Two things to be aware of when reading the output. There is no cortex
annotation for a fresh slide, so nothing restricts the model to the cortex and
it will also predict in medulla and capsule, where it was never trained. And
overlapping tiles paint over one another, so an object near a tile seam may be
drawn twice; the counts printed at the end are tile-level and are not object
counts.

Usage:
  python render_wsi_overlay.py --run patient_100k \
      --image /work/.../HI15011717B1.tif --fp16
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2 import model_zoo

import config as C

COLOR = C.CLASS_COLOR


def grid_starts(length, tile, stride):
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def build_cfg(weights, base_yaml, arch, floor, classes):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    if arch == "mask2former":
        from mask2former import add_maskformer2_config
        add_maskformer2_config(cfg)
    cfg.merge_from_file(base_yaml)
    n = len(classes)
    if arch == "mask2former":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "bitmask"
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n
        cfg.MODEL.RETINANET.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "polygon"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = floor
    cfg.MODEL.WEIGHTS = weights
    cfg.INPUT.MIN_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = C.TRAIN["test_size"]
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def cortex_mask_for_slide(store, H, W, base_yaml, args):
    """Run the cortex model at its own magnification and return a mask.

    The mask comes back at the downscaled resolution the model works in; the
    caller maps tile coordinates into it rather than the other way round, which
    avoids ever holding a full-resolution mask of the slide.
    """
    spec = C.CORTEX_MODEL
    ratio = float(spec["target_long_side"]) / max(H, W)
    dw, dh = int(round(W * ratio)), int(round(H * ratio))
    print("cortex stage: resizing %d x %d to %d x %d (ratio %.4f)"
          % (W, H, dw, dh, ratio))

    small = np.full((dh, dw, 3), 255, dtype=np.uint8)
    band = 4096
    for by in range(0, H, band):
        bh = min(band, H - by)
        chunk = np.asarray(store[by:by + bh])
        if chunk.ndim == 2:
            chunk = cv2.cvtColor(chunk, cv2.COLOR_GRAY2BGR)
        else:
            chunk = chunk[:, :, ::-1]
        ty0 = int(by * ratio)
        ty1 = min(int((by + bh) * ratio), dh)
        if ty1 > ty0:
            small[ty0:ty1] = cv2.resize(np.ascontiguousarray(chunk),
                                        (dw, ty1 - ty0),
                                        interpolation=cv2.INTER_AREA)
        del chunk

    cfg = build_cfg(spec["weights"], base_yaml, "mask2former",
                    spec["score"], C.CLASS_SETS["cortex"])
    pred = DefaultPredictor(cfg)

    ts, st = spec["tile_size"], spec["stride"]
    mask = np.zeros((dh, dw), dtype=bool)
    n = 0
    for y0 in grid_starts(dh, ts, st):
        for x0 in grid_starts(dw, ts, st):
            tile = np.full((ts, ts, 3), 255, dtype=np.uint8)
            y1, x1 = min(y0 + ts, dh), min(x0 + ts, dw)
            tile[: y1 - y0, : x1 - x0] = small[y0:y1, x0:x1]
            out = pred(tile)["instances"].to("cpu")
            n += 1
            if len(out) == 0:
                continue
            masks = out.pred_masks.numpy()
            scores = out.scores.numpy()
            cls = out.pred_classes.numpy()
            for i in range(len(out)):
                if scores[i] < spec["score"]:
                    continue
                if C.CLASS_SETS["cortex"][int(cls[i])] != "cortex":
                    continue
                mask[y0:y1, x0:x1] |= masks[i][: y1 - y0, : x1 - x0].astype(bool)

    cover = 100.0 * mask.mean()
    print("cortex stage: %d tile(s), cortex covers %.1f%% of the slide" % (n, cover))
    if cover < 1.0:
        print("  WARNING: almost nothing was found. Check that the resize ratio")
        print("  matches how this model was trained.")
    del small
    return mask, ratio


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--run", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--arch", default="mask2former", choices=sorted(C.ARCH.keys()))
    ap.add_argument("--class-set", default="main", choices=sorted(C.CLASS_SETS.keys()),
                    help="main is the ten structural classes; cortex is the "
                         "three tissue regions and needs a model trained on them")
    ap.add_argument("--only", nargs="+", default=None,
                    help="draw only these classes. Ten classes at once is a lot "
                         "at this scale; one family per figure reads better.")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base-yaml", default=None)
    ap.add_argument("--out-dir", default="wsi_overlays")
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--score", type=float, default=0.50)
    ap.add_argument("--out-scale", type=float, default=0.125,
                    help="downscale of the painted canvas. 0.125 turns a 30720 "
                         "px slide into 3840, the size of one annotated region; "
                         "0.25 gives 7680 and holds up when a reader zooms in.")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--line-width", type=int, default=1,
                    help="outline thickness on the canvas. At out-scale 0.25 or "
                         "above a value of 2 reads better in print.")
    ap.add_argument("--skip-blank", type=float, default=235.0,
                    help="skip tiles whose mean intensity is above this; a slide "
                         "is mostly empty glass and those tiles cost time for "
                         "nothing. Set 255 to process everything.")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--crop", nargs=4, type=int, default=None,
                    metavar=("X", "Y", "W", "H"),
                    help="process only this region of the slide, in full-"
                         "resolution pixels")
    ap.add_argument("--cortex", action="store_true",
                    help="run the cortex model first and restrict the main "
                         "model to what it finds, as the production system does")
    ap.add_argument("--cortex-min", type=float, default=C.MIN_TILE_CORTEX,
                    help="fraction of a tile that must be cortex for it to be "
                         "processed")
    ap.add_argument("--draw-cortex", action="store_true",
                    help="outline the cortex region on the canvas")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import tifffile

    classes = C.CLASS_SETS[args.class_set]
    draw = [c for c in classes if not args.only or c in args.only]
    if not draw:
        raise SystemExit("--only matched none of the %s classes" % args.class_set)

    work = Path(args.work_dir)
    spec = C.ARCH[args.arch]
    if args.weights is None:
        args.weights = str(work / "s04_train" / args.run / "model_best.pth")
    if args.base_yaml is None:
        if spec["from_model_zoo"]:
            args.base_yaml = model_zoo.get_config_file(spec["yaml"])
        else:
            p = Path(spec["yaml"] or C.BASE_YAML)
            args.base_yaml = str(p if p.is_absolute() else here / p)

    # memory-map: an uncompressed single-page tiff can be read in pieces, so
    # 1.5 GB never has to sit in RAM at once
    store = tifffile.memmap(args.image)
    H, W = store.shape[:2]
    if args.crop:
        cx, cy, cw, ch = args.crop
    else:
        cx, cy, cw, ch = 0, 0, W, H
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))
    xs = [cx + x for x in grid_starts(cw, args.tile_size, stride)]
    ys = [cy + y for y in grid_starts(ch, args.tile_size, stride)]

    oh = int(ch * args.out_scale)
    ow = int(cw * args.out_scale)

    print("=" * 72)
    print("slide %s" % Path(args.image).name)
    print("full size %d x %d   region %d x %d at (%d, %d)" % (W, H, cw, ch, cx, cy))
    print("tiles %d x %d = %d   stride %d   score >= %.2f%s"
          % (len(xs), len(ys), len(xs) * len(ys), stride, args.score,
             "   fp16" if args.fp16 else ""))
    print("canvas %d x %d" % (ow, oh))
    print()
    print("No cortex annotation exists for this slide, so nothing restricts the")
    print("model to the cortex. Predictions in medulla and capsule are outside")
    print("the distribution it was trained on and should be read as such.")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise SystemExit(
            "no GPU visible. This has to run on a compute node -- submit it "
            "with qsub rather than running it on the login node.")

    cmask = cratio = None
    if args.cortex:
        cmask, cratio = cortex_mask_for_slide(store, H, W, args.base_yaml, args)

    predictor = DefaultPredictor(
        build_cfg(args.weights, args.base_yaml, args.arch, args.score, classes))
    use_amp = args.fp16

    # base canvas, built in horizontal bands so the full slide never has to be
    # materialised: 534 megapixels would be about 1.6 GB in one array
    base = np.full((oh, ow, 3), 255, dtype=np.uint8)
    band = 4096
    for by in range(0, ch, band):
        bh = min(band, ch - by)
        chunk = np.asarray(store[cy + by:cy + by + bh, cx:cx + cw])
        if chunk.ndim == 2:
            chunk = cv2.cvtColor(chunk, cv2.COLOR_GRAY2BGR)
        else:
            chunk = chunk[:, :, ::-1]
        ty0 = int(by * args.out_scale)
        ty1 = min(int((by + bh) * args.out_scale), oh)
        if ty1 <= ty0:
            continue
        base[ty0:ty1] = cv2.resize(np.ascontiguousarray(chunk), (ow, ty1 - ty0),
                                   interpolation=cv2.INTER_AREA)
        del chunk
    paint = np.zeros_like(base)
    edges = np.zeros_like(base)
    painted = np.zeros((oh, ow), dtype=bool)

    per_class = defaultdict(int)
    n_run = n_blank = n_outside = 0
    t0 = time.time()
    todo = [(x, y) for y in ys for x in xs]
    if args.limit:
        todo = todo[:args.limit]

    for k, (x0, y0) in enumerate(todo, 1):
        y1, x1 = min(y0 + args.tile_size, H), min(x0 + args.tile_size, W)
        raw = np.asarray(store[y0:y1, x0:x1])
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        else:
            raw = raw[:, :, ::-1]
        if raw.mean() > args.skip_blank:
            n_blank += 1
            continue

        if cmask is not None:
            # map this tile into the cortex model's coordinate frame
            gy0, gy1 = int(y0 * cratio), max(int(y1 * cratio), int(y0 * cratio) + 1)
            gx0, gx1 = int(x0 * cratio), max(int(x1 * cratio), int(x0 * cratio) + 1)
            patch = cmask[gy0:gy1, gx0:gx1]
            if patch.size == 0 or patch.mean() < args.cortex_min:
                n_outside += 1
                continue

        tile = np.full((args.tile_size, args.tile_size, 3), 255, dtype=np.uint8)
        tile[: y1 - y0, : x1 - x0] = raw
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                out = predictor(tile)["instances"].to("cpu")
        else:
            out = predictor(tile)["instances"].to("cpu")
        n_run += 1
        if len(out) == 0:
            continue

        masks = out.pred_masks.numpy()
        scores = out.scores.numpy()
        cls_idx = out.pred_classes.numpy()
        # paint the tile's masks at canvas scale, then blit into place
        tw = max(int(args.tile_size * args.out_scale), 1)
        for i in range(len(out)):
            if scores[i] < args.score:
                continue
            name = classes[int(cls_idx[i])]
            if name not in draw:
                continue
            small = cv2.resize(masks[i].astype(np.uint8), (tw, tw),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if not small.any():
                continue
            px = int((x0 - cx) * args.out_scale)
            py = int((y0 - cy) * args.out_scale)
            ph = min(tw, oh - py)
            pw = min(tw, ow - px)
            if ph <= 0 or pw <= 0:
                continue
            sub = small[:ph, :pw]
            paint[py:py + ph, px:px + pw][sub] = COLOR[name]
            painted[py:py + ph, px:px + pw] |= sub

            # Outlines, drawn into a tile-local buffer first. An object cut by
            # a tile edge has its contour closed along that cut, which is not a
            # real boundary; repeated over a thousand tiles those cuts form a
            # visible lattice. The ring along a seam is therefore erased before
            # the buffer is merged in. Nothing is lost: tiles overlap by a
            # third, so the neighbouring tile holds the same object whole and
            # draws its true outline.
            buf = np.zeros((ph, pw, 3), dtype=np.uint8)
            res = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
            cnts = res[1] if len(res) == 3 else res[0]
            cv2.drawContours(buf, cnts, -1, COLOR[name], args.line_width,
                             cv2.LINE_AA)
            r = max(args.line_width + 1, 2)
            if x0 > cx:
                buf[:, :r] = 0
            if y0 > cy:
                buf[:r, :] = 0
            if x0 + args.tile_size < cx + cw:
                buf[:, -r:] = 0
            if y0 + args.tile_size < cy + ch:
                buf[-r:, :] = 0
            np.maximum(edges[py:py + ph, px:px + pw], buf,
                       out=edges[py:py + ph, px:px + pw])
            per_class[name] += 1

        if k % 100 == 0:
            el = time.time() - t0
            print("  %d/%d tiles   %d run, %d blank, %d outside cortex   "
                  "%.0fs elapsed, ~%.0fs left"
                  % (k, len(todo), n_run, n_blank, n_outside, el,
                     el / k * (len(todo) - k)))

    canvas = base.copy()
    canvas[painted] = (args.alpha * paint[painted] +
                       (1 - args.alpha) * base[painted]).astype(np.uint8)
    if cmask is not None and args.draw_cortex:
        cm = cv2.resize(cmask.astype(np.uint8), (ow, oh),
                        interpolation=cv2.INTER_NEAREST)
        res = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = res[1] if len(res) == 3 else res[0]
        cv2.drawContours(edges, cnts, -1, C.CLASS_COLOR["cortex"], 2, cv2.LINE_AA)

    lit = edges.any(axis=2)
    canvas[lit] = edges[lit]

    # legend, drawn on a white strip below the image so it never covers tissue
    present = [c for c in draw if per_class.get(c)]
    if present:
        rows = (len(present) + 2) // 3
        lh = 22 + rows * 26
        strip = np.full((lh, ow, 3), 255, dtype=np.uint8)
        colw = ow // 3
        for i, c in enumerate(present):
            r, col = divmod(i, 3)
            x = 14 + col * colw
            y = 26 + r * 26
            cv2.rectangle(strip, (x, y - 11), (x + 26, y + 5), COLOR[c], -1)
            cv2.rectangle(strip, (x, y - 11), (x + 26, y + 5), (90, 90, 90), 1)
            cv2.putText(strip, "%s  (%d)" % (C.display(c, short=True),
                                             per_class[c]),
                        (x + 34, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (30, 30, 30), 1, cv2.LINE_AA)
        canvas = np.vstack([canvas, strip])
        base_pad = np.vstack([base, np.full((lh, ow, 3), 255, dtype=np.uint8)])
    else:
        base_pad = base

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    side = np.hstack([base_pad,
                      np.full((canvas.shape[0], 12, 3), 255, dtype=np.uint8),
                      canvas])
    cv2.imwrite(str(out_dir / ("%s_overlay.jpg" % stem)), canvas,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    cv2.imwrite(str(out_dir / ("%s_sidebyside.jpg" % stem)), side,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print("\n" + "=" * 72)
    print("tiles run %d   blank %d   outside cortex %d   elapsed %.0f s"
          % (n_run, n_blank, n_outside, time.time() - t0))
    print("-" * 72)
    for c in draw:
        if per_class.get(c):
            print("   %-32s%8d" % (C.display(c), per_class[c]))
    print("-" * 72)
    print("These are tile-level detections, not object counts: overlapping")
    print("tiles see the same object more than once.")
    print("wrote %s" % (out_dir / ("%s_overlay.jpg" % stem)))
    print("=" * 72)


if __name__ == "__main__":
    main()