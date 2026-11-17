"""
Confidence intervals over patients, by resampling the test set.

The seed-to-seed spread reported in Chapter 4 describes one source of
variability: random initialisation and data ordering. It says nothing about the
other, which is that the test set holds 23 patients and a different 23 would
have given a different figure. With veine at 13 instances and pac at 14, that
second source is likely the larger of the two.

Resampling has to be done over PATIENTS rather than regions, for the same reason
the partition was rebuilt at patient level: regions from one biopsy are not
independent. Scores are recomputed on each resample from the per-region counts,
since a macro average cannot be taken apart after the fact -- which is why
s08_eval writes s08_per_source.json alongside the summary.

Usage:
  python bootstrap_patients.py --runs patient_40k patient_40k_s43 patient_40k_s44
  python bootstrap_patients.py --runs <m2f runs> --against <mrcnn runs>
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as C

SUFFIX = re.compile(r"(?<=\d)[A-Z]{1,3}\d{1,2}$")


def patient_of(source_key):
    """Same rule as s02b_split_patient: EM0846N3 -> EM0846."""
    p = SUFFIX.sub("", source_key)
    return p if p else source_key


def load(path):
    with open(str(path)) as f:
        return json.load(f)


def macro_f1(sources, keys, classes):
    """Macro-averaged F1 over the given regions, computed from raw counts.

    Classes absent from the resample are skipped rather than scored as zero: a
    class with no annotations and no predictions is not a failure, and counting
    it as one would penalise resamples that happen to exclude the rare classes.
    """
    out = []
    for c in classes:
        tp = fp = fn = 0
        for k in keys:
            r = sources[k][c]
            tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        if tp + fn == 0 and tp + fp == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * p * r_ / (p + r_) if p + r_ else 0.0)
    return float(np.mean(out)) if out else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="runs to pool; usually the three seeds of one architecture")
    ap.add_argument("--against", nargs="+", default=None,
                    help="a second set of runs, to bootstrap the difference")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="_base050")
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    work = Path(args.work_dir)

    def read(run):
        p = work / "s08_eval" / run / (args.split + args.tag) / "s08_per_source.json"
        if not p.exists():
            raise SystemExit("missing %s -- rerun s08_eval for %s" % (p, run))
        return load(p)

    a = [read(r) for r in args.runs]
    b = [read(r) for r in args.against] if args.against else None
    classes = a[0]["classes"]

    # every run must cover the same regions, or the resamples are not comparable
    region_sets = [set(d["sources"]) for d in a] + \
                  ([set(d["sources"]) for d in b] if b else [])
    if len({frozenset(s) for s in region_sets}) != 1:
        raise SystemExit("the runs do not cover the same regions")
    regions = sorted(region_sets[0])

    by_patient = defaultdict(list)
    for r in regions:
        by_patient[patient_of(r)].append(r)
    patients = sorted(by_patient)

    print("=" * 72)
    print("bootstrap over patients | %d regions, %d patients, %d resamples"
          % (len(regions), len(patients), args.n))
    print("resampling %s" % ", ".join(args.runs))
    if b:
        print("      against %s" % ", ".join(args.against))
    print("=" * 72)

    # point estimate: the mean over seeds on the full test set
    point_a = float(np.mean([macro_f1(d["sources"], regions, classes) for d in a]))
    print("%-28s %.4f" % ("point estimate", point_a))
    if b:
        point_b = float(np.mean([macro_f1(d["sources"], regions, classes) for d in b]))
        print("%-28s %.4f" % ("point estimate, second set", point_b))
        print("%-28s %+.4f" % ("difference", point_a - point_b))
    print("-" * 72)

    rng = np.random.default_rng(args.seed)
    draws_a, draws_b, draws_d = [], [], []
    for _ in range(args.n):
        picked = rng.choice(len(patients), size=len(patients), replace=True)
        keys = [r for i in picked for r in by_patient[patients[i]]]
        va = float(np.mean([macro_f1(d["sources"], keys, classes) for d in a]))
        draws_a.append(va)
        if b:
            vb = float(np.mean([macro_f1(d["sources"], keys, classes) for d in b]))
            draws_b.append(vb)
            draws_d.append(va - vb)

    lo, hi = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)

    def report(name, draws, point):
        d = np.array(draws)
        l, h = np.percentile(d, [lo, hi])
        print("%-28s %.4f  [%.4f, %.4f]   SD %.4f"
              % (name, point, l, h, d.std(ddof=1)))
        return l, h

    report("macro F1", draws_a, point_a)
    if b:
        report("macro F1, second set", draws_b, point_b)
        l, h = report("difference", draws_d, point_a - point_b)
        print("-" * 72)
        crosses = l <= 0.0 <= h
        frac = float((np.array(draws_d) <= 0).mean())
        print("The %d%% interval on the difference %s zero."
              % (int(100 * (1 - args.alpha)),
                 "includes" if crosses else "excludes"))
        print("The difference is negative in %.1f%% of resamples." % (100 * frac))
        if crosses:
            print("Resampling the patients is therefore enough to reverse the")
            print("ordering, and the difference should not be reported as")
            print("established.")

    print("=" * 72)
    print("Patients are resampled with replacement and all regions of a patient")
    print("move together. Seeds are averaged within each resample rather than")
    print("resampled, so the interval describes variability from the choice of")
    print("patients alone; the seed spread is reported separately.")


if __name__ == "__main__":
    main()