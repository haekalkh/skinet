"""
E7 Stage 1 - Audit raw sources (MAIN + CORTEX).

Config-driven (imports e7_config). Data-integrity gate: verifies XML/image
pairing, parses ASAP labels, normalizes variants, counts per-class instances per
source, scores rare/hard-class presence, and separates MAIN vs CORTEX branches.

Outputs (in e7_config.PREP_DIR):
  01_e7_raw_main_audit.csv
  01_e7_raw_cortex_audit.csv
  01_e7_raw_source_audit_summary.json
  01_e7_parse_errors.json
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, OrderedDict

import pandas as pd

import e7_config as C


# Rare/hard scoring weights (heuristic, used to prioritise sources later).
RARE_WEIGHTS = {"pac": 5, "veine": 4, "media": 4, "intima": 4,
                "nsg_complet": 3, "nsg_partiel": 3, "vaisseau": 2}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_image_for_xml(xml_path: Path):
    stem = xml_path.stem
    d = xml_path.parent
    for ext in C.IMAGE_EXTS:
        cand = d / f"{stem}{ext}"
        if cand.exists():
            return cand
    for p in d.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in C.IMAGE_EXTS:
            return p
    return None


def extract_labels(xml_path: Path):
    labels = []
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as e:
        return labels, str(e)
    for elem in root.iter():
        if "annotation" not in elem.tag.lower():
            continue
        raw = (elem.attrib.get("PartOfGroup") or elem.attrib.get("partofgroup")
               or elem.attrib.get("Name") or elem.attrib.get("name")
               or elem.attrib.get("Label") or elem.attrib.get("Group"))
        lab = C.normalize_label(raw)
        if lab is not None:
            labels.append(lab)
    return labels, None


def audit_folder(raw_dir: Path, source_type: str):
    rows, glob_counter, unknown_counter, errors = [], Counter(), Counter(), []
    for xml_path in sorted(raw_dir.glob("*.xml")):
        img = find_image_for_xml(xml_path)
        labels, err = extract_labels(xml_path)
        if err is not None:
            errors.append({"source_key": xml_path.stem, "source_type": source_type,
                           "xml_path": str(xml_path), "error": err})
        lc = Counter(labels)
        glob_counter.update(lc)

        main_counts = {c: int(lc.get(c, 0)) for c in C.MAIN_CLASSES}
        cortex_counts = {c: int(lc.get(c, 0)) for c in C.CORTEX_CLASSES}
        rare_counts = {c: int(lc.get(c, 0)) for c in C.KNOWN_HARD_CLASSES}

        unknown = [l for l in lc
                   if l not in C.MAIN_CLASSES and l not in C.CORTEX_CLASSES
                   and l not in C.EXCLUDED_MAIN_LABELS]
        for l in unknown:
            unknown_counter[l] += lc[l]

        rare_score = int(sum(RARE_WEIGHTS.get(c, 0) * rare_counts.get(c, 0)
                             for c in RARE_WEIGHTS))

        row = OrderedDict(
            source_key=xml_path.stem, source_type=source_type,
            xml_path=str(xml_path), image_path="" if img is None else str(img),
            has_image=bool(img is not None), parse_ok=bool(err is None),
            num_annotations_total=int(sum(lc.values())),
            num_main_annotations=int(sum(main_counts.values())),
            num_cortex_annotations=int(sum(cortex_counts.values())),
            num_excluded_annotations=int(sum(lc.get(c, 0) for c in C.EXCLUDED_MAIN_LABELS)),
            num_unknown_annotations=int(sum(lc.get(l, 0) for l in unknown)),
            has_any_main=bool(sum(main_counts.values()) > 0),
            has_any_cortex=bool(sum(cortex_counts.values()) > 0),
            has_rare_hard=bool(sum(rare_counts.values()) > 0),
            rare_score=rare_score,
            unknown_labels=",".join(sorted(unknown)),
        )
        for c in C.MAIN_CLASSES:
            row[f"main_{c}"] = main_counts[c]
        for c in C.CORTEX_CLASSES:
            row[f"cortex_{c}"] = cortex_counts[c]
        for c in C.KNOWN_HARD_CLASSES:
            row[f"rare_{c}"] = rare_counts[c]
        rows.append(row)
    return rows, glob_counter, unknown_counter, errors


def sorted_counter(counter):
    return {k: int(v) for k, v in sorted(counter.items(), key=lambda x: (-x[1], x[0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-main-dir", default=str(C.RAW_MAIN_DIR))
    ap.add_argument("--raw-cortex-dir", default=str(C.RAW_CORTEX_DIR))
    ap.add_argument("--out-dir", default=str(C.PREP_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("=" * 80)
    print("E7 Stage 1 - Audit Raw Sources")
    print(f"MAIN  : {args.raw_main_dir}")
    print(f"CORTEX: {args.raw_cortex_dir}")
    print("=" * 80)

    m_rows, m_cnt, m_unk, m_err = audit_folder(Path(args.raw_main_dir), "main_raw")
    c_rows, c_cnt, c_unk, c_err = audit_folder(Path(args.raw_cortex_dir), "cortex_raw")

    m_df, c_df = pd.DataFrame(m_rows), pd.DataFrame(c_rows)
    m_df.to_csv(out_dir / "01_e7_raw_main_audit.csv", index=False)
    c_df.to_csv(out_dir / "01_e7_raw_cortex_audit.csv", index=False)

    all_cnt = Counter(); all_cnt.update(m_cnt); all_cnt.update(c_cnt)
    all_unk = Counter(); all_unk.update(m_unk); all_unk.update(c_unk)
    errors = m_err + c_err

    summary = OrderedDict(
        dataset_name="e7_clinical_pipeline",
        num_main_xml=len(m_rows), num_cortex_xml=len(c_rows),
        num_main_with_image=int(m_df["has_image"].sum()) if len(m_df) else 0,
        num_cortex_with_image=int(c_df["has_image"].sum()) if len(c_df) else 0,
        num_main_parse_ok=int(m_df["parse_ok"].sum()) if len(m_df) else 0,
        num_cortex_parse_ok=int(c_df["parse_ok"].sum()) if len(c_df) else 0,
        num_parse_errors=len(errors),
        main_class_counts={c: int(m_cnt.get(c, 0)) for c in C.MAIN_CLASSES},
        cortex_class_counts={c: int(c_cnt.get(c, 0)) for c in C.CORTEX_CLASSES},
        excluded_label_counts_in_main={c: int(m_cnt.get(c, 0)) for c in C.EXCLUDED_MAIN_LABELS},
        all_label_counts=sorted_counter(all_cnt),
        unknown_label_counts=sorted_counter(all_unk),
        rare_hard_presence_main={c: int((m_df[f"rare_{c}"] > 0).sum()) if len(m_df) else 0
                                 for c in C.KNOWN_HARD_CLASSES},
    )

    with open(out_dir / "01_e7_raw_source_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "01_e7_parse_errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print(json.dumps(summary, indent=2))
    if all_unk:
        print("\n[WARN] Unknown labels found (not in MAIN/CORTEX/excluded):",
              sorted_counter(all_unk))
    print(f"\nSaved audit CSVs + summary to {out_dir}")


if __name__ == "__main__":
    main()