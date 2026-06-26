"""
E7 Step 03 (CORTEX) - Build CORTEX COCO tiles.

Differs from the MAIN tiler in two essential ways (cortex = large region at low
magnification, not small structures at native res):

1. RESIZE-THEN-TILE. Large cortex source images are downscaled to a target long
   side (default 2048, matching the paper's ~x25 cortex view) BEFORE tiling at
   1024. The resize ratio and original dimensions are stored per tile so the
   cortex area can be mapped back to NATIVE pixels for morphometry.
   Set --target-long-side 0 to tile at native resolution.

2. TILE-COVERAGE RETENTION. The cortex/capsule/medullaire regions are huge; each
   tile only ever contains a small fraction of the whole region, so the MAIN
   "fraction-of-object" rule would drop almost every tile. Here a region mask is
   kept in a tile if it covers >= min_tile_coverage of the TILE area instead.

Classes: cortex, capsule, medullaire (3).
No targeted augmentation (region classes are abundant in area; not rare).

Input : 02_e7_cortex_source_split.csv
Output: data/coco/e7_clinical_pipeline/cortex/{train,val,test}/images + annotations
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import OrderedDict, defaultdict

import cv2
import numpy as np


ROOT = Path("/work/lead/mk657917/SKINET")
PREP = ROOT / "outputs/preprocessing/e7_clinical_pipeline"

DEFAULT_SPLIT_CSV = PREP / "02_e7_cortex_source_split.csv"
DEFAULT_OUT_DIR = ROOT / "data/coco/e7_clinical_pipeline/cortex"

CORTEX_CLASSES = ["cortex", "capsule", "medullaire"]
CLASS_TO_ID = {c: i + 1 for i, c in enumerate(CORTEX_CLASSES)}  # COCO ids start at 1

LABEL_NORMALIZATION = {
    "cortex": "cortex",
    "capsule": "capsule",
    "medullaire": "medullaire", "médullaire": "medullaire", "medulla": "medullaire",
}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def normalize_label(label):
    if label is None:
        return None
    s = str(label).strip().lower().replace("_", " ").replace("-", " ").strip()
    if s in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[s]
    return s.replace(" ", "_")


def parse_asap_xml(xml_path: Path):
    """Return list of {label, points (Nx2 float)} for cortex classes only."""
    out = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return out
    for ann in root.iter():
        if "annotation" not in ann.tag.lower():
            continue
        raw = (ann.attrib.get("PartOfGroup") or ann.attrib.get("partofgroup")
               or ann.attrib.get("Name") or ann.attrib.get("Group"))
        label = normalize_label(raw)
        if label not in CORTEX_CLASSES:
            continue
        pts = []
        for coord in ann.iter():
            if coord.tag.lower().endswith("coordinate"):
                try:
                    pts.append([float(coord.attrib["X"]), float(coord.attrib["Y"])])
                except (KeyError, ValueError):
                    pass
        if len(pts) >= 3:
            out.append({"label": label, "points": np.array(pts, dtype=np.float64)})
    return out


def compute_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
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


def mask_to_coco_segm(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    segm = []
    for c in contours:
        if cv2.contourArea(c.astype(np.float32)) <= 0 or len(c) < 3:
            continue
        segm.append(c.reshape(-1).astype(float).tolist())
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
    return {"segmentation": segm, "bbox": [x, y, bw, bh], "area": float(mask.sum())}


def build_split(split_name, rows, out_dir, tile_size, stride, target_long_side,
                min_tile_coverage):
    img_out = out_dir / split_name / "images"
    ensure_dir(img_out)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for c, i in CLASS_TO_ID.items()]}
    image_id, ann_id = 1, 1
    summary = {"num_tiles": 0, "num_annotations": 0, "num_skipped_empty": 0,
               "class_counts": defaultdict(int), "resize_ratios": []}

    tile_area = float(tile_size * tile_size)
    min_cov_px = min_tile_coverage * tile_area

    for _, r in rows.iterrows():
        src = r["source_key"]
        image = cv2.imread(str(r["image_path"]))
        if image is None:
            continue
        oh, ow = image.shape[:2]
        anns = parse_asap_xml(Path(r["xml_path"]))
        if not anns:
            continue

        # --- resize to low magnification ---
        ratio = 1.0
        if target_long_side and max(oh, ow) > target_long_side:
            ratio = target_long_side / float(max(oh, ow))
            image = cv2.resize(image, (round(ow * ratio), round(oh * ratio)),
                               interpolation=cv2.INTER_AREA)
            for a in anns:
                a["points"] = a["points"] * ratio
        summary["resize_ratios"].append(ratio)
        h, w = image.shape[:2]

        for y0 in compute_starts(h, tile_size, stride):
            for x0 in compute_starts(w, tile_size, stride):
                tile_img = make_tile_image(image, x0, y0, tile_size)
                tile_anns = []
                for a in anns:
                    mask = polygon_to_tile_mask(a["points"], x0, y0, tile_size)
                    vis = float(mask.sum())
                    if vis < min_cov_px:          # tile-coverage retention
                        continue
                    obj = mask_to_coco_segm(mask)
                    if obj is None or not obj["segmentation"]:
                        continue
                    tile_anns.append((a["label"], obj))
                if not tile_anns:
                    summary["num_skipped_empty"] += 1
                    continue

                fname = f"{src}_tile_{image_id:05d}_x{x0}_y{y0}.jpg"
                cv2.imwrite(str(img_out / fname), tile_img,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                coco["images"].append({
                    "id": image_id, "file_name": f"{split_name}/images/{fname}",
                    "width": tile_size, "height": tile_size,
                    "source_key": src, "tile_x": int(x0), "tile_y": int(y0),
                    "tile_size": tile_size,
                    "resize_ratio": float(ratio),      # tile/native scale
                    "orig_width": int(ow), "orig_height": int(oh),
                })
                for label, obj in tile_anns:
                    coco["annotations"].append({
                        "id": ann_id, "image_id": image_id,
                        "category_id": CLASS_TO_ID[label],
                        "segmentation": obj["segmentation"], "bbox": obj["bbox"],
                        "area": obj["area"], "iscrowd": 0,
                    })
                    summary["class_counts"][label] += 1
                    ann_id += 1
                    summary["num_annotations"] += 1
                image_id += 1
                summary["num_tiles"] += 1

    ann_dir = out_dir / "annotations"
    ensure_dir(ann_dir)
    with open(ann_dir / f"instances_{split_name}.json", "w") as f:
        json.dump(coco, f)
    summary["class_counts"] = dict(summary["class_counts"])
    summary["mean_resize_ratio"] = float(np.mean(summary["resize_ratios"])) if summary["resize_ratios"] else 1.0
    del summary["resize_ratios"]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", default=str(DEFAULT_SPLIT_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap-ratio", type=float, default=0.333)
    ap.add_argument("--target-long-side", type=int, default=2048,
                    help="Downscale source so its long side = this before tiling. 0 = native.")
    ap.add_argument("--min-tile-coverage", type=float, default=0.05,
                    help="Keep a region mask in a tile if it covers >= this fraction of the tile.")
    args = ap.parse_args()

    import pandas as pd
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))

    print("=" * 80)
    print("E7 Step 03 (CORTEX) - Build COCO tiles (resize + tile-coverage)")
    print(f"Tile {args.tile_size}  stride {stride}  target_long_side {args.target_long_side}"
          f"  min_tile_coverage {args.min_tile_coverage}")
    print("=" * 80)

    df = pd.read_csv(args.split_csv)
    overall = OrderedDict()
    for split in ["train", "val", "test"]:
        rows = df[df["split"] == split]
        s = build_split(split, rows, out_dir, args.tile_size, stride,
                        args.target_long_side, args.min_tile_coverage)
        overall[split] = s
        print(f"\n[{split}] {json.dumps(s, indent=2)}")

    overall["params"] = {"tile_size": args.tile_size, "stride": stride,
                         "target_long_side": args.target_long_side,
                         "min_tile_coverage": args.min_tile_coverage}
    overall["note"] = (
        "Cortex tiles use resize-then-tile (low magnification) and tile-coverage "
        "retention. resize_ratio per image lets morphometry convert cortex area "
        "back to native pixels (native_area = tile_area / resize_ratio^2)."
    )
    with open(out_dir / "03_e7_cortex_build_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\nSaved COCO + summary under {out_dir}")


if __name__ == "__main__":
    main()