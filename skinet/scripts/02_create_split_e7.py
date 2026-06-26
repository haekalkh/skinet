"""
E7 Stage 2 - MAIN source-level split.

Config-driven (imports e7_config). Splits MAIN sources train/val/test at the
SOURCE level (never at tile level) to prevent leakage: a WSI's tiles never
straddle two splits. Fractions and seed come from e7_config.

Input : 01_e7_raw_main_audit.csv  (Stage 1)
Output: 02_e7_source_split.csv  +  02_e7_source_split_summary.json
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

import e7_config as C


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def class_counts(df):
    return {c: int(df[f"main_{c}"].sum()) for c in C.MAIN_CLASSES}


def rare_presence(df):
    return {c: int((df[f"rare_{c}"] > 0).sum()) for c in C.KNOWN_HARD_CLASSES}


def check_no_leakage(df):
    dup = df[df["source_key"].duplicated(keep=False)]
    if len(dup) > 0:
        raise RuntimeError(f"Source leakage: {sorted(dup['source_key'].unique())[:20]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv",
                    default=str(C.PREP_DIR / "01_e7_raw_main_audit.csv"))
    ap.add_argument("--out-dir", default=str(C.PREP_DIR))
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    train_frac, val_frac, test_frac = C.SPLIT_FRACTIONS
    # fraction of the (val+test) remainder that goes to test
    test_within_temp = test_frac / (val_frac + test_frac)

    print("=" * 80)
    print("E7 Stage 2 - MAIN Source-level Split")
    print(f"Audit CSV : {args.audit_csv}")
    print(f"Fractions : train {train_frac} / val {val_frac} / test {test_frac}")
    print(f"Seed      : {args.seed}")
    print("=" * 80)

    df = pd.read_csv(args.audit_csv)
    df = df[(df["has_image"] == True)
            & (df["parse_ok"] == True)
            & (df["has_any_main"] == True)].copy()
    df = df.sort_values("source_key").reset_index(drop=True)

    if len(df) < 3:
        raise RuntimeError(f"Too few usable MAIN sources to split: {len(df)}")

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, random_state=args.seed, shuffle=True)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_within_temp, random_state=args.seed, shuffle=True)

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    check_no_leakage(split_df)

    order = {"train": 0, "val": 1, "test": 2}
    split_df["_o"] = split_df["split"].map(order)
    split_df = split_df.sort_values(["_o", "source_key"]).drop(columns=["_o"])

    summary = OrderedDict(
        dataset_name="e7_clinical_pipeline", branch="main",
        method="source_level_split", seed=int(args.seed),
        fractions={"train": train_frac, "val": val_frac, "test": test_frac},
        main_classes=C.MAIN_CLASSES,
        split_counts={s: int((split_df["split"] == s).sum())
                      for s in ["train", "val", "test"]},
        class_counts_by_split={s: class_counts(split_df[split_df["split"] == s])
                               for s in ["train", "val", "test"]},
        rare_presence_by_split={s: rare_presence(split_df[split_df["split"] == s])
                                for s in ["train", "val", "test"]},
        source_leakage_check="passed",
        note=("Source-level split for MAIN. No tiling/augmentation here. "
              "CORTEX branch inherits this split for shared source_keys."),
    )

    split_csv = out_dir / "02_e7_source_split.csv"
    summary_json = out_dir / "02_e7_source_split_summary.json"
    split_df.to_csv(split_csv, index=False)
    summary["outputs"] = {"split_csv": str(split_csv), "summary_json": str(summary_json)}
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    # Sanity warning: rare/hard class absent from a split makes its per-class
    # metrics undefined; flag it now rather than at eval time.
    for s in ["train", "val", "test"]:
        for c, n in summary["rare_presence_by_split"][s].items():
            if n == 0:
                print(f"[WARN] hard class '{c}' has 0 sources in '{s}' split")

    print(f"\nSaved:\n  {split_csv}\n  {summary_json}")


if __name__ == "__main__":
    main()