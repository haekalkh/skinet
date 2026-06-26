"""
E7 Stage 5 (CORTEX) - Register CORTEX COCO datasets in Detectron2.

Config-driven (imports e7_config). Clone of the MAIN register for the 3 cortex
region classes. Verifies each split loads and that metadata matches
e7_config.CORTEX_CLASSES. No training here.

Output: 05_e7_register_cortex_coco_summary.json
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import e7_config as C


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def register_split(split, coco_dir: Path):
    name = C.DATASET_NAMES_CORTEX[split]
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    ann_file = coco_dir / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        raise RuntimeError(f"Missing annotation file: {ann_file}")
    register_coco_instances(
        name=name,
        metadata={"thing_classes": C.CORTEX_CLASSES},
        json_file=str(ann_file),
        image_root=str(coco_dir),
    )
    return name, ann_file


def summarize(name):
    dd = DatasetCatalog.get(name)
    md = MetadataCatalog.get(name)
    n_ann = sum(len(d.get("annotations", [])) for d in dd)
    sample = dd[0] if dd else None
    out = OrderedDict(
        dataset_name=name, num_images=len(dd), num_annotations=int(n_ann),
        thing_classes=list(md.thing_classes),
        classes_match_config=(list(md.thing_classes) == C.CORTEX_CLASSES),
    )
    if sample is not None:
        out["sample_file_name"] = sample.get("file_name")
        out["sample_hw"] = [int(sample.get("height")), int(sample.get("width"))]
        out["sample_num_annotations"] = len(sample.get("annotations", []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default=str(C.COCO_CORTEX_DIR))
    ap.add_argument("--out-dir", default=str(C.PREP_DIR))
    args = ap.parse_args()

    coco_dir = Path(args.coco_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("=" * 80)
    print("E7 Stage 5 (CORTEX) - Register CORTEX COCO")
    print(f"COCO dir: {coco_dir}")
    print("=" * 80)

    report = OrderedDict(branch="cortex", cortex_classes=C.CORTEX_CLASSES, datasets={})
    all_ok = True
    for split in ["train", "val", "test"]:
        name, ann_file = register_split(split, coco_dir)
        summ = summarize(name)
        summ["split"] = split
        summ["annotation_file"] = str(ann_file)
        report["datasets"][split] = summ
        all_ok = all_ok and summ["classes_match_config"]
        print(f"\n[{split}] {json.dumps(summ, indent=2)}")

    report["status"] = "OK" if all_ok else "CLASS_MISMATCH"
    with open(out_dir / "05_e7_register_cortex_coco_summary.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Status: {report['status']}")
    if not all_ok:
        print("[ACTION] thing_classes != e7_config.CORTEX_CLASSES — fix before training.")


if __name__ == "__main__":
    main()