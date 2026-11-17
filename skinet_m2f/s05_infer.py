"""
Stage 05 - run the trained model over whole ROIs.

Walks the SAME tile grid stage 03 used, and applies the SAME cortex gate. That
parity matters: if training only ever saw tiles inside the cortex, inference
must not be asked about anything else, or the model is judged on a distribution
it was never shown.

Unlike the COCO test set, every tile in the grid is visited here, including the
ones that hold no annotation. That is the real working condition.

Each detected instance is stored with its tile offset and a tile-local RLE.
Stage 06 maps those into whole-ROI coordinates; nothing is fused or filtered
here beyond a deliberately low score floor.

Outputs (into --out-dir):
  preds/<source_key>.json
  s05_summary.json
  run_meta.json

Usage:
  python s05_infer.py --run baseline_seed42 --split test
  python s05_infer.py --run baseline_seed42 --split test --limit 2
"""

import argparse
import contextlib
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2 import model_zoo
# mask2former is imported lazily inside build_cfg: importing it is what
# registers the MaskFormer architecture, and the Mask R-CNN arm must not need it.

import config as C


def grid_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def cut_tile(image, x0, y0, tile):
    h, w = image.shape[:2]
    x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
    out = np.full((tile, tile, 3), 255, dtype=np.uint8)
    out[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
    return out


def predict_cortex_polygons(image, base_yaml, spec):
    """Delineate the cortex with the region model, returning polygons.

    On an annotated region the cortex comes from the XML. A fresh slide has no
    annotation, so it has to be predicted -- and the region model works at its
    own magnification, on an image resized so the long side is 2048 and tiled at
    2048. A single tile then covers the whole biopsy, which is what that model
    was trained on.

    Polygons are returned rather than a mask, in native coordinates, because
    that is what the gating and masking code already consumes. Everything
    downstream is then identical to the annotated case.
    """
    from detectron2.engine import DefaultPredictor

    H, W = image.shape[:2]
    ratio = float(spec["target_long_side"]) / max(H, W)
    dw, dh = int(round(W * ratio)), int(round(H * ratio))
    small = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)

    cfg = build_cfg(spec["weights"], base_yaml, "mask2former", spec["score"],
                    classes=C.CLASS_SETS["cortex"])
    pred = DefaultPredictor(cfg)

    ts, st = spec["tile_size"], spec["stride"]
    mask = np.zeros((dh, dw), dtype=np.uint8)
    for y0 in grid_starts(dh, ts, st):
        for x0 in grid_starts(dw, ts, st):
            tile = np.full((ts, ts, 3), 255, dtype=np.uint8)
            y1, x1 = min(y0 + ts, dh), min(x0 + ts, dw)
            tile[: y1 - y0, : x1 - x0] = small[y0:y1, x0:x1]
            out = pred(tile)["instances"].to("cpu")
            if len(out) == 0:
                continue
            m = out.pred_masks.numpy()
            sc = out.scores.numpy()
            cl = out.pred_classes.numpy()
            for k in range(len(out)):
                if sc[k] < spec["score"]:
                    continue
                if C.CLASS_SETS["cortex"][int(cl[k])] != "cortex":
                    continue
                mask[y0:y1, x0:x1] |= m[k][: y1 - y0, : x1 - x0].astype(np.uint8)

    del pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = res[1] if len(res) == 3 else res[0]
    polys = []
    for c in cnts:
        if len(c) < 3:
            continue
        polys.append((c.reshape(-1, 2).astype(np.float64) / ratio))

    # area is taken from the mask, not from the polygons: external contours
    # enclose any holes, and a cortex with medulla inside it would be
    # overcounted. Scaling by ratio squared returns it to native pixels, which
    # is the same conversion the legacy build summary records.
    area_native = float(mask.sum()) / (ratio * ratio)
    return polys, float(mask.mean()), area_native


def cortex_polygons(xml_path):
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
            out.append(np.array(pts, dtype=np.float64))
    return out


def cortex_mask(polys, h, w):
    """Full-image binary mask of the annotated cortex."""
    m = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    return m


def apply_cortex_mask(image, mask, fill):
    """Blank everything outside the cortex, as the legacy system does.

    The legacy pipeline does not filter tiles: it blacks out every pixel outside
    the cortex with a bitwise_and, crops to the cortex bounding box, and only
    then divides into tiles. Its MAIN model therefore never sees non-cortex
    tissue at all. Tile gating is a weaker version of the same idea — a tile that
    passes the gate still carries whatever non-cortex tissue it contains, and the
    model still predicts on it.

    Cropping is left out here on purpose: it would change the tile grid and make
    the comparison with the ungated run harder to read. Masking is the part that
    changes what the model sees.
    """
    out = np.full_like(image, fill)
    keep = mask.astype(bool)
    out[keep] = image[keep]
    return out


def cortex_coverage(polys, x0, y0, tile):
    m = np.zeros((tile, tile), dtype=np.uint8)
    for p in polys:
        shifted = p - np.array([x0, y0], dtype=np.float64)
        cv2.fillPoly(m, [shifted.round().astype(np.int32)], 1)
    return float(m.sum()) / float(tile * tile)


def build_cfg(weights, base_yaml, arch, score_floor, classes=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    if arch == "mask2former":
        from mask2former import add_maskformer2_config
        add_maskformer2_config(cfg)
    cfg.merge_from_file(base_yaml)

    n = len(classes if classes is not None else C.STRUCTURE_CLASSES)
    if arch == "mask2former":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "bitmask"
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n
        cfg.MODEL.RETINANET.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "polygon"
        # Let the same low floor through as for Mask2Former, so both arms hand
        # post-processing the same kind of candidate list. Detection cap stays
        # at Detectron2's default 100, which matches Mask2Former's 100 queries.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_floor

    cfg.MODEL.WEIGHTS = weights
    cfg.INPUT.MIN_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = C.TRAIN["test_size"]
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def encode(mask_bool):
    rle = maskUtils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--run", required=True, help="training run to load weights from")
    ap.add_argument("--arch", default="mask2former", choices=sorted(C.ARCH.keys()),
                    help="must match the architecture the run was trained with")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base-yaml", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--min-tile-cortex", type=float, default=C.MIN_TILE_CORTEX)
    ap.add_argument("--score-floor", type=float, default=C.INFER["score_floor"])
    ap.add_argument("--mask-cortex", action="store_true",
                    help="blank the image outside the annotated cortex before "
                         "tiling, the way the legacy system does. Note the "
                         "training tiles were NOT masked, so this introduces a "
                         "mismatch -- see the note printed at startup.")
    ap.add_argument("--mask-fill", type=int, default=255,
                    help="value to blank with. 255 matches the white padding "
                         "stage 03 already used at tile edges; the legacy system "
                         "used 0.")
    ap.add_argument("--fp16", action="store_true",
                    help="run inference under autocast. Costs nothing and needs "
                         "no retraining, so the only question is whether the "
                         "predictions change -- which is what stages 06 to 08 "
                         "are for.")
    ap.add_argument("--out-dir-tag", default="",
                    help="suffix for the output folder, so one configuration "
                         "never overwrites another")
    ap.add_argument("--images", nargs="+", default=None,
                    help="run on these image files instead of the split, for "
                         "slides that carry no annotation. Implies "
                         "--predict-cortex unless the gate is switched off.")
    ap.add_argument("--predict-cortex", action="store_true",
                    help="obtain the cortex from the region model rather than "
                         "from the annotation file")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.weights is None:
        args.weights = str(work / "s04_train" / args.run / "model_best.pth")
    spec = C.ARCH[args.arch]
    if args.base_yaml is None:
        if spec["from_model_zoo"]:
            args.base_yaml = model_zoo.get_config_file(spec["yaml"])
        else:
            p = Path(spec["yaml"] or C.BASE_YAML)
            args.base_yaml = str(p if p.is_absolute() else here / p)
    if args.out_dir is None:
        args.out_dir = str(work / "s05_infer" / args.run / args.split)
    if args.out_dir_tag:
        args.out_dir += args.out_dir_tag

    for p in (args.weights, args.base_yaml):
        if not Path(p).exists():
            raise SystemExit("not found: %s" % p)

    out_dir = Path(args.out_dir)
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))

    if args.images:
        # slides given directly, with no annotation to draw the cortex from
        rows = [OrderedDict(source_key=Path(p).stem, image_path=str(p),
                            xml_path="") for p in args.images]
        if not args.predict_cortex and args.min_tile_cortex > 0.0:
            args.predict_cortex = True
    else:
        with open(str(work / "s02_split" / "s02_split.csv")) as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("no sources for split %s" % args.split)

    print("=" * 70)
    print("stage 05 - inference | run %s  arch %s  split %s"
          % (args.run, args.arch, args.split))
    print("weights %s" % args.weights)
    print("tile %d  stride %d  cortex gate %.2f  score floor %.2f  %s"
          % (args.tile_size, stride, args.min_tile_cortex, args.score_floor,
             "autocast fp16" if args.fp16 else "fp32"))
    if args.predict_cortex:
        print("cortex from the region model, not from annotation")
    if args.mask_cortex:
        print("cortex masking ON, fill %d" % args.mask_fill)
        print("  The model was trained on tiles that still contained non-cortex")
        print("  tissue, so masking only at inference is a train/test mismatch.")
        print("  If accuracy drops, that is the first thing to suspect, and the")
        print("  clean version needs stage 03 rebuilt with masking and a retrain.")
    print("=" * 70)

    predictor = DefaultPredictor(
        build_cfg(args.weights, args.base_yaml, args.arch, args.score_floor))

    use_amp = args.fp16 and torch.cuda.is_available()

    def predict(tile):
        if not use_amp:
            return predictor(tile)
        with torch.autocast("cuda", dtype=torch.float16):
            return predictor(tile)

    s = OrderedDict(n_sources=0, n_tiles_in_grid=0, n_tiles_run=0,
                    n_gated_out_cortex=0, n_slides_no_cortex=0,
                    n_masked=0, n_instances=0)
    per_class = defaultdict(int)

    for i, r in enumerate(rows, 1):
        image = cv2.imread(r["image_path"])
        if image is None:
            print("[skip] cannot read %s" % r["image_path"])
            continue
        oh, ow = image.shape[:2]
        if args.predict_cortex:
            cortex, frac, cortex_area = predict_cortex_polygons(
                image, args.base_yaml, C.CORTEX_MODEL)
            cortex_dir = out_dir / "cortex"
            cortex_dir.mkdir(parents=True, exist_ok=True)
            with open(str(cortex_dir / (r["source_key"] + ".json")), "w") as cf:
                json.dump(OrderedDict(
                    source_key=r["source_key"], height=oh, width=ow,
                    fraction=frac, area_px=cortex_area,
                    polygons=[p.tolist() for p in cortex]), cf)
            print("   cortex predicted: %d region(s), %.1f%% of the slide, "
                  "%.2f mm2 at %.3f um/px"
                  % (len(cortex), 100 * frac,
                     cortex_area * (C.MPP / 1000.0) ** 2, C.MPP))
            if frac < 0.01:
                print("   WARNING: almost nothing found. Check that the resize")
                print("   ratio matches how the region model was trained.")
        else:
            cortex = cortex_polygons(Path(r["xml_path"]))
        gate_on = args.min_tile_cortex > 0.0 and len(cortex) > 0

        if args.mask_cortex and cortex:
            image = apply_cortex_mask(image, cortex_mask(cortex, oh, ow),
                                      args.mask_fill)
            s["n_masked"] = s.get("n_masked", 0) + 1
        if args.min_tile_cortex > 0.0 and not cortex:
            s["n_slides_no_cortex"] += 1

        ys = grid_starts(oh, args.tile_size, stride)
        xs = grid_starts(ow, args.tile_size, stride)
        s["n_sources"] += 1
        s["n_tiles_in_grid"] += len(xs) * len(ys)

        instances = []
        n_run = 0
        for y0 in ys:
            for x0 in xs:
                if gate_on and cortex_coverage(cortex, x0, y0, args.tile_size) < args.min_tile_cortex:
                    s["n_gated_out_cortex"] += 1
                    continue
                out = predict(cut_tile(image, x0, y0, args.tile_size))["instances"].to("cpu")
                n_run += 1
                if len(out) == 0:
                    continue
                masks = out.pred_masks.numpy()
                scores = out.scores.numpy()
                classes = out.pred_classes.numpy()
                for k in range(len(out)):
                    if scores[k] < args.score_floor or masks[k].sum() <= 0:
                        continue
                    name = C.STRUCTURE_CLASSES[int(classes[k])]
                    m = masks[k]
                    # Did this tile see the object whole? A mask touching the
                    # tile edge was cut by the grid, so whatever lies past the
                    # edge -- for a glomerulus, possibly the vascular pole that
                    # decides complete versus partial -- was never visible when
                    # the label was chosen. Post-processing uses this to prefer
                    # the tile that saw the most.
                    touches = bool(m[0, :].any() or m[-1, :].any()
                                   or m[:, 0].any() or m[:, -1].any())
                    instances.append({
                        "class_name": name,
                        "category_id": int(classes[k]) + 1,
                        "score": float(scores[k]),
                        "tile_x": int(x0), "tile_y": int(y0),
                        "whole_view": (not touches),
                        "mask_rle": encode(m),
                    })
                    per_class[name] += 1

        s["n_tiles_run"] += n_run
        s["n_instances"] += len(instances)
        with open(str(preds_dir / ("%s.json" % r["source_key"])), "w") as f:
            json.dump(OrderedDict(source_key=r["source_key"],
                                  height=int(oh), width=int(ow),
                                  tile_size=args.tile_size, stride=stride,
                                  n_tiles=n_run, instances=instances), f)
        print("[%d/%d] %s  %dx%d  %d/%d tiles  %d instances"
              % (i, len(rows), r["source_key"], ow, oh, n_run,
                 len(xs) * len(ys), len(instances)))
        sys.stdout.flush()

    s["instances_per_class"] = OrderedDict(
        (c, int(per_class.get(c, 0))) for c in C.STRUCTURE_CLASSES)
    with open(str(out_dir / "s05_summary.json"), "w") as f:
        json.dump(s, f, indent=2)

    print("\n" + "=" * 70)
    print("sources %d   tiles run %d of %d   gated out %d   instances %d"
          % (s["n_sources"], s["n_tiles_run"], s["n_tiles_in_grid"],
             s["n_gated_out_cortex"], s["n_instances"]))
    for c in C.STRUCTURE_CLASSES:
        print("   %-20s%7d" % (c, per_class.get(c, 0)))
    print("=" * 70)
    C.write_run_meta(out_dir, "s05_infer", args, extra=s)


if __name__ == "__main__":
    main()