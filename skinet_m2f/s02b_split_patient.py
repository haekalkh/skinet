"""
Stage 02b - reassign the split so no patient appears on two sides.

The slide-level split leaks: 32 of 141 patients had slides in more than one
split, many of them across train and test. Sections from one biopsy share
staining, scanner settings, tissue and often the same structures, so a model
that saw EM0846N1 in training has an unfair advantage on EM0846N3 in test. The
legacy system guards against exactly this (separateByPatient=True).

This rewrites only the `split` column of the existing s02_split.csv, keeping
every other field, so the rest of the pipeline reads it unchanged. Splitting
happens over PATIENTS, then slides follow their patient.

Patient id is the source key with a short trailing section suffix removed:
EM0846N3 -> EM0846, ML7746B10 -> ML7746, HI15011016V1 -> HI15011016. Keys with
no such suffix are left alone. Always read the printed mapping before trusting
it -- these identifiers are a local convention, not a standard.

Usage:
  python s02b_split_patient.py --dry-run     # inspect the grouping first
  python s02b_split_patient.py
"""

import argparse
import csv
import random
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import config as C

SUFFIX = re.compile(r"(?<=\d)[A-Z]{1,3}\d{1,2}$")


def patient_of(source_key):
    p = SUFFIX.sub("", source_key)
    return p if p else source_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-csv", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--dry-run", action="store_true",
                    help="show the grouping and the proposed split, write nothing")
    args = ap.parse_args()

    work = Path(args.work_dir)
    in_csv = Path(args.in_csv) if args.in_csv else work / "s02_split" / "s02_split.csv"
    out_csv = Path(args.out_csv) if args.out_csv else work / "s02_split" / "s02_split_patient.csv"

    with open(str(in_csv)) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames
    if "split" not in fields or "source_key" not in fields:
        raise SystemExit("expected source_key and split columns in %s" % in_csv)

    by_patient = defaultdict(list)
    for r in rows:
        by_patient[patient_of(r["source_key"])].append(r)

    # how bad was the old one
    old_span = defaultdict(set)
    for p, rs in by_patient.items():
        for r in rs:
            old_span[p].add(r["split"])
    leaking = sorted(p for p, s in old_span.items() if len(s) > 1)

    print("=" * 74)
    print("stage 02b - patient-level split")
    print("=" * 74)
    print("slides %d   patients %d" % (len(rows), len(by_patient)))
    print("patients spanning more than one split in the OLD file: %d"
          % len(leaking))
    multi = sorted(((len(rs), p) for p, rs in by_patient.items()), reverse=True)
    print("largest patients: " + ", ".join(
        "%s(%d)" % (p, n) for n, p in multi[:8]))
    print("-" * 74)

    # split over patients, largest first, always into whichever split is
    # furthest below its target -- keeps slide counts close without splitting
    # any patient
    patients = sorted(by_patient.keys())
    random.Random(args.seed).shuffle(patients)
    patients.sort(key=lambda p: -len(by_patient[p]))

    total = len(rows)
    target = {"train": total * (1.0 - args.val_frac - args.test_frac),
              "val": total * args.val_frac,
              "test": total * args.test_frac}
    got = {"train": 0, "val": 0, "test": 0}
    assign = {}
    for p in patients:
        n = len(by_patient[p])
        pick = max(("train", "val", "test"),
                   key=lambda s: (target[s] - got[s]) / max(target[s], 1.0))
        assign[p] = pick
        got[pick] += n

    for r in rows:
        r["split"] = assign[patient_of(r["source_key"])]

    n_pat = defaultdict(int)
    for p, s in assign.items():
        n_pat[s] += 1
    print("%-8s%10s%10s%10s" % ("split", "patients", "slides", "target"))
    for s in ("train", "val", "test"):
        print("%-8s%10d%10d%10.0f" % (s, n_pat[s], got[s], target[s]))
    print("-" * 74)

    # verify
    check = defaultdict(set)
    for r in rows:
        check[patient_of(r["source_key"])].add(r["split"])
    bad = [p for p, s in check.items() if len(s) > 1]
    print("patients spanning more than one split now: %d" % len(bad))
    if bad:
        raise SystemExit("split still leaks: %s" % bad[:5])

    if args.dry_run:
        print("\ndry run, nothing written. Sample of the grouping:")
        for n, p in multi[:10]:
            keys = sorted(r["source_key"] for r in by_patient[p])
            print("   %-14s -> %s" % (p, ", ".join(keys)))
        print("\nCheck those groupings are right before writing.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_csv), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s" % out_csv)
    print("\nEverything downstream has to be rebuilt from here: stage 03 tiles,")
    print("training, then inference and evaluation. Numbers from the old split")
    print("are not comparable with numbers from this one.")
    C.write_run_meta(out_csv.parent, "s02b_split_patient", args,
                     extra=OrderedDict(n_slides=len(rows),
                                       n_patients=len(by_patient),
                                       old_leaking_patients=len(leaking),
                                       slides_per_split=dict(got),
                                       patients_per_split=dict(n_pat)))


if __name__ == "__main__":
    main()