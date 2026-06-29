"""
E7 Stage 3 - Build MAIN COCO tiles.

Config-driven (imports e7_config). Native-resolution tiling for the 10 MAIN
structural classes. Rules ported from the MRCNN SKINET divider, with the
diagnostic fix:

  - Fragment retention = MAIN_MIN_VISIBLE_RATIO (0.10) of the WHOLE object,
    plus an absolute floor MAIN_MIN_VISIBLE_AREA_PX. (E7's old 0.03 leaked
    boundary slivers into the noisy COCO "small" bucket.)
  - Optional cortex-aware tile gating (MAIN_MIN_TILE_CORTEX): drop tiles whose
    cortex coverage is below threshold. Uses GT cortex polygons found in the
    SAME main XML. DEFAULT OFF (0.0) for the MAIN-first build, because inference
    cannot gate by cortex until the cortex model exists (train/infer parity).
  - Targeted offline augmentation off by default (USE_TARGETED_AUG); online LSJ
    in the M2F mapper handles augmentation.

Each tile image entry stores source_key / tile_x / tile_y / tile_size /
resize_ratio(=1.0) / orig dims, which the stitching stage needs.

Input : 02_e7_source_split.csv
Output: COCO_MAIN_DIR/{train,val,test}/images + annotations/instances_{split}.json
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict, defaultdict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd

import e7_config as C


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_asap_xml(xml_path: Path):
    """Return (main_anns, cortex_polys).
    main_anns: list of {label, points} for the 10 MAIN classes.
    cortex_polys: list of points arrays labelled 'cortex' (for optional gating).
    """
    main_anns, cortex_polys = [], []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return main_anns, cortex_polys
    for ann in root.iter():
        if "annotation" not in ann.tag.lower():
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = C.normalize_label(raw)
        pts = []
        for coord in ann.iter():
            if coord.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(coord.attrib["X"]), float(coord.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.float64)
        if label in C.MAIN_CLASSES:
            main_anns.append({"label": label, "points": arr})
        elif label == "cortex":
            cortex_polys.append(arr)
    return main_anns, cortex_polys


def compute_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def make_tile_image(image, x0, y0, tile_size):
    h, w = image.shape[:2]
    x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
    tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    tile[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
    return tile


def polygon_to_tile_mask(points, x0, y0, tile_size):
    shifted = points.copy()
    shifted[:, 0] -= x0
    shifted[:, 1] -= y0
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted.round().astype(np.int32)], 1)
    return mask


def cortex_coverage(cortex_polys, x0, y0, tile_size):
    if not cortex_polys:
        return None  # no cortex GT -> cannot gate this source
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
    for poly in cortex_polys:
        shifted = poly.copy()
        shifted[:, 0] -= x0
        shifted[:, 1] -= y0
        cv2.fillPoly(mask, [shifted.round().astype(np.int32)], 1)
    return float(mask.sum()) / float(tile_size * tile_size)


def mask_to_coco_segm(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    segm = []
    for c in contours:
        if cv2.contourArea(c.astype(np.float32)) <= 0 or len(c) < 3:
            continue
        segm.append(c.reshape(-1).astype(float).tolist())
    if not segm:
        return None
    ys, xs = np.where(mask > 0)
    x, y = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return {"segmentation": segm, "bbox": [x, y, bw, bh], "area": float(mask.sum())}


def apply_tf_img(img, tf):
    if tf == "original": return img
    if tf == "flip_h": return np.ascontiguousarray(img[:, ::-1])
    if tf == "flip_v": return np.ascontiguousarray(img[::-1, :])
    if tf == "rot90": return np.ascontiguousarray(np.rot90(img, 1))
    if tf == "rot180": return np.ascontiguousarray(np.rot90(img, 2))
    if tf == "rot270": return np.ascontiguousarray(np.rot90(img, 3))
    return img


def apply_tf_mask(mask, tf):
    return apply_tf_img(mask, tf)


def build_split(split_name, rows, out_dir, tile_size, stride,
                min_visible_ratio, min_visible_area, min_tile_cortex,
                use_aug):
    img_out = out_dir / split_name / "images"
    ensure_dir(img_out)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for c, i in C.MAIN_ID.items()]}
    image_id, ann_id = 1, 1
    s = OrderedDict(num_tiles=0, num_annotations=0, num_base_tiles=0,
                    num_aug_tiles=0, num_skipped_empty=0,
                    num_skipped_sliver=0, num_gated_out=0,
                    num_sources_no_cortex_gt=0, num_sources=0,
                    class_counts=defaultdict(int))

    for _, r in rows.iterrows():
        s["num_sources"] += 1
        image = cv2.imread(str(r["image_path"]))
        if image is None:
            print(f"[WARN] cannot read image: {r['image_path']}")
            continue
        oh, ow = image.shape[:2]
        anns, cortex_polys = parse_asap_xml(Path(r["xml_path"]))
        if not anns:
            continue

        gating_on = min_tile_cortex > 0.0
        if gating_on and not cortex_polys:
            s["num_sources_no_cortex_gt"] += 1  # keep all tiles for this source

        for y0 in compute_starts(oh, tile_size, stride):
            for x0 in compute_starts(ow, tile_size, stride):
                # cortex gate (only if enabled AND this source has cortex GT)
                if gating_on and cortex_polys:
                    cov = cortex_coverage(cortex_polys, x0, y0, tile_size)
                    if cov is not None and cov < min_tile_cortex:
                        s["num_gated_out"] += 1
                        continue

                tile_img = make_tile_image(image, x0, y0, tile_size)
                tile_masks, tile_labels = [], []
                for a in anns:
                    full_area = abs(cv2.contourArea(a["points"].astype(np.float32)))
                    if full_area <= 0:
                        continue
                    mask = polygon_to_tile_mask(a["points"], x0, y0, tile_size)
                    vis = float(mask.sum())
                    if vis <= 0:
                        continue
                    if vis < min_visible_area or (vis / full_area) < min_visible_ratio:
                        s["num_skipped_sliver"] += 1
                        continue
                    tile_masks.append(mask)
                    tile_labels.append(a["label"])

                if not tile_masks:
                    s["num_skipped_empty"] += 1
                    continue

                transforms = ["original"]
                if use_aug and split_name == "train" and \
                        any(l in C.TARGETED_AUG_CLASSES for l in tile_labels):
                    transforms += C.AUG_TRANSFORMS

                for tf in transforms:
                    out_img = apply_tf_img(tile_img, tf)
                    out_masks = [apply_tf_mask(m, tf) for m in tile_masks]
                    suffix = "" if tf == "original" else f"_{tf}"
                    fname = f"{r['source_key']}_tile_{image_id:05d}_x{x0}_y{y0}{suffix}.jpg"

                    objs = []
                    for m, lab in zip(out_masks, tile_labels):
                        obj = mask_to_coco_segm(m)
                        if obj is not None:
                            objs.append((lab, obj))
                    if not objs:
                        continue

                    cv2.imwrite(str(img_out / fname), out_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    coco["images"].append({
                        "id": image_id,
                        "file_name": f"{split_name}/images/{fname}",
                        "width": tile_size, "height": tile_size,
                        "source_key": r["source_key"],
                        "tile_x": int(x0), "tile_y": int(y0), "tile_size": tile_size,
                        "resize_ratio": 1.0, "orig_width": int(ow), "orig_height": int(oh),
                        "augmentation": tf,
                    })
                    for lab, obj in objs:
                        coco["annotations"].append({
                            "id": ann_id, "image_id": image_id,
                            "category_id": C.MAIN_ID[lab],
                            "segmentation": obj["segmentation"], "bbox": obj["bbox"],
                            "area": obj["area"], "iscrowd": 0,
                        })
                        s["class_counts"][lab] += 1
                        ann_id += 1
                        s["num_annotations"] += 1
                    image_id += 1
                    s["num_tiles"] += 1
                    s["num_base_tiles" if tf == "original" else "num_aug_tiles"] += 1

    ann_dir = out_dir / "annotations"
    ensure_dir(ann_dir)
    with open(ann_dir / f"instances_{split_name}.json", "w") as f:
        json.dump(coco, f)
    s["class_counts"] = {c: int(s["class_counts"].get(c, 0)) for c in C.MAIN_CLASSES}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", default=str(C.PREP_DIR / "02_e7_source_split.csv"))
    ap.add_argument("--out-dir", default=str(C.COCO_MAIN_DIR))
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--min-visible-ratio", type=float, default=C.MAIN_MIN_VISIBLE_RATIO)
    ap.add_argument("--min-visible-area", type=float, default=C.MAIN_MIN_VISIBLE_AREA_PX)
    ap.add_argument("--min-tile-cortex", type=float, default=0.0,
                    help="Cortex coverage gate (0 = off; needs cortex model for "
                         "inference parity before enabling).")
    ap.add_argument("--use-aug", action="store_true", default=C.USE_TARGETED_AUG)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))

    print("=" * 80)
    print("E7 Stage 3 - Build MAIN COCO tiles")
    print(f"Tile {args.tile_size}  stride {stride}  min_visible_ratio "
          f"{args.min_visible_ratio}  min_tile_cortex {args.min_tile_cortex}  "
          f"aug {args.use_aug}")
    print("=" * 80)

    df = pd.read_csv(args.split_csv)
    overall = OrderedDict()
    for split in ["train", "val", "test"]:
        rows = df[df["split"] == split]
        st = build_split(split, rows, out_dir, args.tile_size, stride,
                         args.min_visible_ratio, args.min_visible_area,
                         args.min_tile_cortex, args.use_aug)
        overall[split] = st
        print(f"\n[{split}] {json.dumps(st, indent=2)}")

    overall["params"] = {"tile_size": args.tile_size, "stride": stride,
                         "min_visible_ratio": args.min_visible_ratio,
                         "min_visible_area": args.min_visible_area,
                         "min_tile_cortex": args.min_tile_cortex,
                         "use_aug": bool(args.use_aug)}
    with open(out_dir / "03_e7_main_build_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSaved COCO + summary under {out_dir}")


if __name__ == "__main__":
    main()