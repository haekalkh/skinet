"""
E7 Step 02 (CORTEX) - Create CORTEX source-level split.

Mirrors the MAIN split but for the 3 cortex-branch classes, with one extra
safeguard: if a cortex source_key also exists in the MAIN split, it INHERITS the
MAIN split assignment. This keeps a given WSI in the same split across both
branches, so when the cortex + main models are combined at inference/test there
is no cross-branch leakage (a source used to train cortex is never a main test
source, and vice-versa). Cortex-only sources are split independently 70/15/15.

Input : 01_e7_raw_cortex_audit.csv  (from step 01)
Output: 02_e7_cortex_source_split.csv
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path("/work/lead/mk657917/SKINET")
PREP = ROOT / "outputs/preprocessing/e7_clinical_pipeline"

DEFAULT_AUDIT_CSV = PREP / "01_e7_raw_cortex_audit.csv"
DEFAULT_MAIN_SPLIT = PREP / "02_e7_source_split.csv"   # to inherit shared keys
OUT_DIR = PREP

CORTEX_CLASSES = ["cortex", "capsule", "medullaire"]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def class_counts(df):
    return {c: int(df[f"cortex_{c}"].sum()) for c in CORTEX_CLASSES}


def check_no_leakage(split_df):
    dup = split_df[split_df["source_key"].duplicated(keep=False)]
    if len(dup) > 0:
        raise RuntimeError(f"Source leakage: {sorted(dup['source_key'].unique())[:20]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV))
    ap.add_argument("--main-split-csv", default=str(DEFAULT_MAIN_SPLIT),
                    help="If present, shared source_keys inherit the MAIN split.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("=" * 80)
    print("E7 Step 02 (CORTEX) - Source-level Split")
    print("=" * 80)

    df = pd.read_csv(args.audit_csv)
    df = df[(df["has_image"] == True)
            & (df["parse_ok"] == True)
            & (df["has_any_cortex"] == True)].copy()
    df = df.sort_values("source_key").reset_index(drop=True)

    # Inherit MAIN split for shared keys
    inherited = {}
    main_path = Path(args.main_split_csv)
    if main_path.exists():
        main_split = pd.read_csv(main_path)[["source_key", "split"]]
        inherited = dict(zip(main_split["source_key"], main_split["split"]))
        print(f"Loaded MAIN split for inheritance: {len(inherited)} keys")

    df["split"] = df["source_key"].map(inherited)
    inherited_mask = df["split"].notna()
    n_inherited = int(inherited_mask.sum())

    # Split the remaining cortex-only sources independently 70/15/15
    rest = df[~inherited_mask].copy()
    if len(rest) > 0:
        tr, tmp = train_test_split(rest, train_size=0.70, random_state=args.seed, shuffle=True)
        va, te = train_test_split(tmp, test_size=0.50, random_state=args.seed, shuffle=True)
        df.loc[tr.index, "split"] = "train"
        df.loc[va.index, "split"] = "val"
        df.loc[te.index, "split"] = "test"

    check_no_leakage(df)

    order = {"train": 0, "val": 1, "test": 2}
    df["_o"] = df["split"].map(order)
    df = df.sort_values(["_o", "source_key"]).drop(columns=["_o"])

    summary = OrderedDict()
    summary["branch"] = "cortex"
    summary["cortex_classes"] = CORTEX_CLASSES
    summary["seed"] = args.seed
    summary["num_sources"] = int(len(df))
    summary["num_inherited_from_main"] = n_inherited
    summary["num_cortex_only_split"] = int(len(df) - n_inherited)
    summary["split_counts"] = {s: int((df["split"] == s).sum()) for s in ["train", "val", "test"]}
    summary["class_counts_by_split"] = {
        s: class_counts(df[df["split"] == s]) for s in ["train", "val", "test"]
    }
    summary["source_leakage_check"] = "passed"
    summary["note"] = (
        "Cortex source-level split. Shared keys inherit the MAIN split to avoid "
        "cross-branch leakage when cortex+main are combined. Cortex region classes "
        "(cortex/capsule/medullaire) are large; tiling uses resize + tile-coverage "
        "retention (see step 03 cortex), not the MAIN object-ratio retention."
    )

    split_csv = out_dir / "02_e7_cortex_source_split.csv"
    summary_json = out_dir / "02_e7_cortex_source_split_summary.json"
    df.to_csv(split_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved:\n  {split_csv}\n  {summary_json}")


if __name__ == "__main__":
    main()