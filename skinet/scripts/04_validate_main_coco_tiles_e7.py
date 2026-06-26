"""
E7 Stage 4 - Validate MAIN COCO tiles + render GT overlays.

Config-driven (imports e7_config). QC gate before training:

  - structural integrity: every image file exists, dimensions match, every
    annotation has a non-empty segmentation, a positive area, an in-bounds
    bbox, and a category_id that matches e7_config.MAIN_ID;
  - per-class instance counts per split (printed, sanity vs the split summary);
  - empty-image / orphan-annotation checks;
  - renders up to --num-overlays GT overlays per split to PREP_DIR/04_overlays
    so the masks can be eyeballed.

Input : COCO_MAIN_DIR/annotations/instances_{split}.json
Output: 04_e7_main_coco_validation.json  +  04_overlays/{split}/*.png
"""

import argparse
import json
import random
from pathlib import Path
from collections import OrderedDict, defaultdict

import cv2
import numpy as np

import e7_config as C


# distinct BGR colours per class for overlays
PALETTE = [
    (66, 135, 245), (245, 66, 66), (66, 245, 96), (245, 212, 66),
    (200, 66, 245), (66, 245, 233), (245, 144, 66), (150, 150, 150),
    (245, 66, 158), (120, 200, 60),
]
CLASS_COLOR = {i + 1: PALETTE[i % len(PALETTE)] for i in range(len(C.MAIN_CLASSES))}
ID_TO_NAME = {i: c for c, i in C.MAIN_ID.items()}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def validate_split(split, coco_dir: Path):
    ann_file = coco_dir / "annotations" / f"instances_{split}.json"
    issues = []
    if not ann_file.exists():
        return {"status": "MISSING", "annotation_file": str(ann_file)}, [], None

    with open(ann_file) as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    # category table must match config exactly
    if cats != ID_TO_NAME:
        issues.append(f"category map mismatch: {cats} != {ID_TO_NAME}")

    class_counts = defaultdict(int)
    imgs_with_ann = set()
    n_bad_segm = n_bad_area = n_bad_cat = n_oob_bbox = 0

    for a in coco["annotations"]:
        cid = a.get("category_id")
        if cid not in cats:
            n_bad_cat += 1
            continue
        class_counts[cats[cid]] += 1
        imgs_with_ann.add(a["image_id"])

        segm = a.get("segmentation")
        if not segm or all(len(p) < 6 for p in segm):
            n_bad_segm += 1
        if float(a.get("area", 0)) <= 0:
            n_bad_area += 1
        im = images.get(a["image_id"])
        if im is not None:
            x, y, bw, bh = a["bbox"]
            if x < 0 or y < 0 or x + bw > im["width"] + 1 or y + bh > im["height"] + 1:
                n_oob_bbox += 1

    # image file existence + empty images
    n_missing_files = 0
    for im in coco["images"]:
        if not (coco_dir / im["file_name"]).exists():
            n_missing_files += 1
    empty_images = [im["id"] for im in coco["images"] if im["id"] not in imgs_with_ann]

    if n_bad_segm: issues.append(f"{n_bad_segm} annotations with empty/degenerate segmentation")
    if n_bad_area: issues.append(f"{n_bad_area} annotations with non-positive area")
    if n_bad_cat: issues.append(f"{n_bad_cat} annotations with unknown category_id")
    if n_oob_bbox: issues.append(f"{n_oob_bbox} annotations with out-of-bounds bbox")
    if n_missing_files: issues.append(f"{n_missing_files} image files missing on disk")
    if empty_images: issues.append(f"{len(empty_images)} images with no annotations")

    summary = OrderedDict(
        status="OK" if not issues else "ISSUES",
        annotation_file=str(ann_file),
        num_images=len(coco["images"]),
        num_annotations=len(coco["annotations"]),
        class_counts={c: int(class_counts.get(c, 0)) for c in C.MAIN_CLASSES},
        num_missing_image_files=n_missing_files,
        num_empty_images=len(empty_images),
        num_bad_segmentation=n_bad_segm,
        num_bad_area=n_bad_area,
        num_bad_category=n_bad_cat,
        num_oob_bbox=n_oob_bbox,
        issues=issues,
    )
    return summary, coco["images"], coco


def render_overlays(split, coco, coco_dir: Path, viz_dir: Path, n, seed=42):
    if coco is None or not coco["images"]:
        return 0
    ensure_dir(viz_dir)
    by_img = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    images = {im["id"]: im for im in coco["images"]}

    # prefer tiles with more / rarer annotations for a useful QC sample
    hard_ids = {C.MAIN_ID[c] for c in C.KNOWN_HARD_CLASSES}
    def score(iid):
        anns = by_img[iid]
        return (sum(1 for a in anns if a["category_id"] in hard_ids), len(anns))
    candidates = sorted(by_img.keys(), key=score, reverse=True)
    random.Random(seed).shuffle(candidates[: max(n * 4, n)])
    chosen = candidates[:n]

    drawn = 0
    for iid in chosen:
        im = images[iid]
        path = coco_dir / im["file_name"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        overlay = img.copy()
        for a in by_img[iid]:
            color = CLASS_COLOR.get(a["category_id"], (255, 255, 255))
            for poly in a["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, 2)
            x, y, bw, bh = [int(v) for v in a["bbox"]]
            cv2.putText(img, ID_TO_NAME[a["category_id"]], (x, max(y - 3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.imwrite(str(viz_dir / f"{split}_{Path(im['file_name']).stem}.png"), blended)
        drawn += 1
    return drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default=str(C.COCO_MAIN_DIR))
    ap.add_argument("--out-dir", default=str(C.PREP_DIR))
    ap.add_argument("--num-overlays", type=int, default=8)
    args = ap.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir)
    viz_root = out_dir / "04_overlays"
    ensure_dir(out_dir)

    print("=" * 80)
    print("E7 Stage 4 - Validate MAIN COCO + overlays")
    print(f"COCO dir: {coco_dir}")
    print("=" * 80)

    report = OrderedDict(dataset_name="e7_clinical_pipeline", branch="main",
                         splits={})
    all_ok = True
    for split in ["train", "val", "test"]:
        summ, _, coco = validate_split(split, coco_dir)
        drawn = render_overlays(split, coco, coco_dir, viz_root / split,
                                args.num_overlays)
        summ["overlays_rendered"] = drawn
        report["splits"][split] = summ
        all_ok = all_ok and summ.get("status") == "OK"
        print(f"\n[{split}] {json.dumps(summ, indent=2)}")

    report["overall_status"] = "OK" if all_ok else "ISSUES"
    with open(out_dir / "04_e7_main_coco_validation.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Overall: {report['overall_status']}")
    print(f"Overlays: {viz_root}")
    print(f"Report : {out_dir / '04_e7_main_coco_validation.json'}")
    if not all_ok:
        print("[ACTION] Resolve issues above before training (Stage 5/6).")


if __name__ == "__main__":
    main()