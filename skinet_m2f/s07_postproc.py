"""
Stage 07 - the seven-stage post-processing chain, ported from the legacy system.

Order matters and is not negotiable; it is the legacy order:

  0. confidence   drop anything below min_confidence
  1. fusion       merge same-class duplicates created by tile overlap
  2. border       drop objects cut by the edge of the field (four classes only)
  3. orphan       a child must sit inside a parent; nsg must hold a child
  4. priority     when two overlapping instances cannot both be right, one wins
  5. orphan       run again — stage 4 can create new orphans by removing parents
  6. class fusion collapse nsg_complet with nsg_partiel into one instance
  7. small        drop debris

Every stage removes or merges whole instances. None of them edits pixels: an
instance is never trimmed to make room for another, which is what keeps the
count meaningful.

Each stage reports how many instances it removed, so the chain can be read as a
funnel rather than a black box.

Outputs (into --out-dir):
  filtered/<source_key>.json
  s07_summary.json
  run_meta.json

Usage:
  python s07_postproc.py --run baseline_seed42 --split test
  python s07_postproc.py --run baseline_seed42 --split test --stop-after 2
"""

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils

import config as C

P = C.POSTPROC


# ------------------------------------------------------------- helpers ----
def rle_in(r):
    return {"size": r["size"], "counts": r["counts"].encode("ascii")}


def rle_out(r):
    return {"size": [int(r["size"][0]), int(r["size"][1])],
            "counts": r["counts"].decode("ascii")}


def cortex_rle(xml_path, h, w):
    """Binary mask of the annotated cortex, as an RLE."""
    polys = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return None
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
            polys.append(np.array(pts, dtype=np.float64))
    if not polys:
        return None
    m = np.zeros((h, w), dtype=np.uint8)
    for p in polys:
        cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    if m.sum() == 0:
        return None
    return maskUtils.encode(np.asfortranarray(m))


def bbox_overlap(a, b):
    """Intersection of two boxes over the smaller one, so containment counts."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def mask_inter(ra, rb):
    return float(maskUtils.area(maskUtils.merge([ra, rb], intersect=True)))


def part_of(ra, rb):
    """Fraction of b that lies inside a."""
    area_b = float(maskUtils.area(rb))
    if area_b <= 0:
        return 0.0
    return mask_inter(ra, rb) / area_b


def union(ra, rb):
    return maskUtils.merge([ra, rb], intersect=False)


def bbox_of(r):
    return [float(v) for v in maskUtils.toBbox(r)]


def border_contact_ratio(r, H, W):
    """How much of the object sits against the edge, relative to its own size.

    The legacy rule measured the share of the mask falling on the blacked-out
    region of its cleaned images. These ROIs have no blacked-out region, so the
    equivalent question is how much of the object runs along the image edge.
    Contact length is compared against the perimeter the object would have if it
    were a circle of the same area, which makes the number size-independent: a
    circle cut in half by the edge scores about 0.32.
    """
    m = maskUtils.decode(r)
    area = float(m.sum())
    if area <= 0:
        return 0.0
    contact = int(m[0, :].sum() + m[H - 1, :].sum() + m[:, 0].sum() + m[:, W - 1].sum())
    if contact == 0:
        return 0.0
    equiv_perimeter = 2.0 * math.sqrt(math.pi * area)
    return contact / equiv_perimeter if equiv_perimeter > 0 else 0.0


# -------------------------------------------------------------- stages ----
def stage_confidence(inst, thr, per_class=None):
    """Drop low-confidence instances. per_class overrides thr where given."""
    if not per_class:
        return [i for i in inst if i["score"] >= thr]
    return [i for i in inst
            if i["score"] >= per_class.get(i["class_name"], thr)]


def stage_cortex(inst, cortex, thr):
    """Keep only instances lying mostly inside the cortex.

    The model was trained on tiles that still held non-cortex tissue, unlabelled,
    so it learned to call that region empty while it plainly is not. At inference
    it predicts there anyway, and every such prediction counts against precision
    because the ground truth holds nothing outside the annotated cortex.
    """
    if cortex is None or thr <= 0:
        return inst
    keep = []
    for it in inst:
        a = float(maskUtils.area(it["_rle"]))
        if a <= 0:
            continue
        inside = mask_inter(cortex, it["_rle"]) / a
        if inside >= thr:
            keep.append(it)
    return keep


def stage_fuse_same_class(inst, bb_thr, mask_thr):
    """Merge same-class instances that overlap. Repeats until nothing merges."""
    items = [dict(i) for i in inst]
    for it in items:
        it["_n"] = 1
        it["whole_view"] = bool(it.get("whole_view", False))
    changed = True
    while changed:
        changed = False
        by_class = defaultdict(list)
        for k, it in enumerate(items):
            if it is not None:
                by_class[it["class_name"]].append(k)
        for _, idxs in by_class.items():
            for a in range(len(idxs)):
                i = idxs[a]
                if items[i] is None:
                    continue
                for b in range(a + 1, len(idxs)):
                    j = idxs[b]
                    if items[j] is None or items[i] is None:
                        continue
                    if bbox_overlap(items[i]["bbox"], items[j]["bbox"]) <= bb_thr:
                        continue
                    ri, rj = items[i]["_rle"], items[j]["_rle"]
                    inter = mask_inter(ri, rj)
                    ai = float(maskUtils.area(ri))
                    aj = float(maskUtils.area(rj))
                    smaller = min(ai, aj)
                    if smaller <= 0 or inter / smaller <= mask_thr:
                        continue
                    u = union(ri, rj)
                    ni, nj = items[i]["_n"], items[j]["_n"]
                    items[i]["_rle"] = u
                    items[i]["bbox"] = bbox_of(u)
                    items[i]["area"] = float(maskUtils.area(u))
                    items[i]["score"] = (items[i]["score"] * ni +
                                         items[j]["score"] * nj) / (ni + nj)
                    items[i]["_n"] = ni + nj
                    # One tile seeing it whole is enough for the pair to count
                    # as wholly observed.
                    items[i]["whole_view"] = (items[i]["whole_view"]
                                              or items[j]["whole_view"])
                    items[j] = None
                    changed = True
    return [it for it in items if it is not None]


def stage_border(inst, thr, classes, H, W):
    keep = []
    for it in inst:
        if it["class_name"] in classes:
            if border_contact_ratio(it["_rle"], H, W) > thr:
                continue
        keep.append(it)
    return keep


def stage_orphan(inst, bb_thr, mask_thr, hierarchy):
    """Children must sit inside a parent; some parents must hold a child."""
    child_of = {}
    for parent, spec in hierarchy.items():
        for ch in spec["contains"]:
            child_of[ch] = parent

    drop = set()
    parent_has_child = defaultdict(bool)

    for ci, child in enumerate(inst):
        cname = child["class_name"]
        pname = child_of.get(cname)
        if pname is None:
            continue
        adopted = False
        for pi, parent in enumerate(inst):
            if pi == ci or parent["class_name"] != pname:
                continue
            if bbox_overlap(parent["bbox"], child["bbox"]) <= bb_thr:
                continue
            if part_of(parent["_rle"], child["_rle"]) > mask_thr:
                adopted = True
                parent_has_child[pi] = True
        if not adopted:
            drop.add(ci)

    for pi, it in enumerate(inst):
        spec = hierarchy.get(it["class_name"])
        if spec is None or spec.get("keep_if_no_child", True):
            continue
        if not parent_has_child[pi]:
            drop.add(pi)

    return [it for k, it in enumerate(inst) if k not in drop]


def stage_score_resolve(inst, pairs, bb_thr, mask_thr):
    """Resolve an overlap between two classes by confidence, keeping one.

    The priority table settles such overlaps by precedence: tubule_sain removes
    tubule_atrophique wherever they meet, whatever the model thought. Removing
    that rule instead lets both survive, which doubles the object where the two
    predictions describe the same tubule. Neither is satisfactory -- one ignores
    the model, the other ignores the anatomy.

    This keeps whichever the model scored higher and discards the other, so one
    tubule stays one tubule and an atrophic call survives where the model is
    confident of it. Masks are not merged: a union of two neighbouring tubules
    is the merging failure this is meant to avoid.
    """
    if not pairs:
        return inst
    lookup = set()
    for a, b in pairs:
        lookup.add((a, b))
        lookup.add((b, a))

    drop = set()
    n = len(inst)
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            ci, cj = inst[i]["class_name"], inst[j]["class_name"]
            if (ci, cj) not in lookup:
                continue
            if bbox_overlap(inst[i]["bbox"], inst[j]["bbox"]) <= bb_thr:
                continue
            ri, rj = inst[i]["_rle"], inst[j]["_rle"]
            inter = mask_inter(ri, rj)
            smaller = min(float(maskUtils.area(ri)), float(maskUtils.area(rj)))
            if smaller <= 0 or inter / smaller <= mask_thr:
                continue
            loser = j if inst[i]["score"] >= inst[j]["score"] else i
            drop.add(loser)
            if loser == i:
                break
    return [it for k, it in enumerate(inst) if k not in drop]


def stage_priority(inst, bb_thr, mask_thr, erases):
    drop = set()
    n = len(inst)
    for i in range(n):
        if i in drop:
            continue
        victims = erases.get(inst[i]["class_name"], [])
        if not victims:
            continue
        for j in range(n):
            if j == i or j in drop:
                continue
            if inst[j]["class_name"] not in victims:
                continue
            if bbox_overlap(inst[i]["bbox"], inst[j]["bbox"]) <= bb_thr:
                continue
            if part_of(inst[i]["_rle"], inst[j]["_rle"]) > mask_thr:
                drop.add(j)
    return [it for k, it in enumerate(inst) if k not in drop]


def pick_winner(a, b, prefer):
    """Which of two fused members gives the surviving label."""
    if prefer == "whole_view":
        aw, bw = bool(a.get("whole_view")), bool(b.get("whole_view"))
        if aw != bw:
            return a if aw else b
    return a if a["score"] >= b["score"] else b


def stage_class_fusion(inst, bb_thr, mask_thr, groups, prefer="score"):
    """Merge overlapping instances inside a compatibility group.

    Which label survives is set by `prefer`. With "whole_view" a member that
    some tile saw whole beats one that was cut, and score only breaks ties.
    """
    partner = {}
    for g in groups:
        for a in g:
            partner[a] = set(g)

    items = [dict(i) for i in inst]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            if items[i] is None:
                continue
            allowed = partner.get(items[i]["class_name"])
            if not allowed:
                continue
            for j in range(i + 1, len(items)):
                if items[j] is None or items[i] is None:
                    continue
                if items[j]["class_name"] not in allowed:
                    continue
                if bbox_overlap(items[i]["bbox"], items[j]["bbox"]) <= bb_thr:
                    continue
                ri, rj = items[i]["_rle"], items[j]["_rle"]
                inter = mask_inter(ri, rj)
                smaller = min(float(maskUtils.area(ri)), float(maskUtils.area(rj)))
                if smaller <= 0 or inter / smaller <= mask_thr:
                    continue
                winner = pick_winner(items[i], items[j], prefer)
                u = union(ri, rj)
                items[i]["class_name"] = winner["class_name"]
                items[i]["category_id"] = winner["category_id"]
                items[i]["score"] = max(items[i]["score"], items[j]["score"])
                items[i]["whole_view"] = (bool(items[i].get("whole_view"))
                                          or bool(items[j].get("whole_view")))
                items[i]["_rle"] = u
                items[i]["bbox"] = bbox_of(u)
                items[i]["area"] = float(maskUtils.area(u))
                items[j] = None
                changed = True
    return [it for it in items if it is not None]


def stage_small(inst, min_area):
    return [it for it in inst if float(maskUtils.area(it["_rle"])) >= min_area]


# ---------------------------------------------------------------- main ----
STAGE_NAMES = ["0_confidence", "0b_cortex", "1_fusion", "2_border", "3_orphan",
               "4_priority", "5_orphan", "6_class_fusion", "7_small"]


def process(rec, args, cortex=None):
    H, W = rec["height"], rec["width"]
    inst = [dict(i) for i in rec["instances"]]
    counts = OrderedDict(input=len(inst))

    inst = stage_confidence(inst, args.min_confidence, args.per_class_thr)
    counts["0_confidence"] = len(inst)
    for it in inst:
        it["_rle"] = rle_in(it["mask_rle"])
    if args.stop_after == 0:
        return inst, counts

    inst = stage_cortex(inst, cortex, args.cortex_min_inside)
    counts["0b_cortex"] = len(inst)

    inst = stage_fuse_same_class(
        inst,
        P["fuse_bb"] if args.fuse_bb is None else args.fuse_bb,
        P["fuse_mask"] if args.fuse_mask is None else args.fuse_mask)
    counts["1_fusion"] = len(inst)
    if args.stop_after == 1:
        return inst, counts

    inst = stage_border(inst, args.border_threshold,
                        set(P["border_classes"]), H, W)
    counts["2_border"] = len(inst)
    if args.stop_after == 2:
        return inst, counts

    inst = stage_orphan(inst, P["orphan_bb"], P["orphan_mask"], P["hierarchy"])
    counts["3_orphan"] = len(inst)
    if args.stop_after == 3:
        return inst, counts

    inst = stage_score_resolve(inst, args.pairs,
                               P["priority_bb"], P["priority_mask"])
    counts["3b_score_resolve"] = len(inst)

    inst = stage_priority(inst, P["priority_bb"], P["priority_mask"], args.erases)
    counts["4_priority"] = len(inst)
    if args.stop_after == 4:
        return inst, counts

    inst = stage_orphan(inst, P["orphan_bb"], P["orphan_mask"], P["hierarchy"])
    counts["5_orphan"] = len(inst)
    if args.stop_after == 5:
        return inst, counts

    inst = stage_class_fusion(inst, P["class_fusion_bb"], P["class_fusion_mask"],
                              P["compatible"], args.class_fusion_prefer)
    counts["6_class_fusion"] = len(inst)
    if args.stop_after == 6:
        return inst, counts

    inst = stage_small(inst, P["min_area_px"])
    counts["7_small"] = len(inst)
    return inst, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-confidence", type=float, default=P["min_confidence"])
    ap.add_argument("--stop-after", type=int, default=7,
                    help="run only up to this stage (0-7); useful for seeing "
                         "what each one costs")
    ap.add_argument("--thresholds", default=None,
                    help="json file of per-class score thresholds, as produced "
                         "by s09_tune.py. Classes not listed keep "
                         "--min-confidence.")
    ap.add_argument("--fuse-mask", type=float, default=None,
                    help="overlap of the SMALLER mask above which two same-class "
                         "predictions are merged. The inherited value is 0.1, "
                         "which merges neighbouring tubules that touch; raising "
                         "it makes fusion less willing to join.")
    ap.add_argument("--fuse-bb", type=float, default=None,
                    help="the same for bounding boxes, applied first as a cheap "
                         "filter")
    ap.add_argument("--score-resolve", nargs=2, action="append",
                    metavar=("A", "B"), default=None,
                    help="where classes A and B overlap, keep whichever scored "
                         "higher and drop the other, instead of letting the "
                         "priority table decide. Implies --unerase for the pair.")
    ap.add_argument("--unerase", nargs=2, action="append", metavar=("A", "B"),
                    default=None,
                    help="stop class A from erasing class B in the priority "
                         "table. Repeatable. Recorded in run_meta, unlike an "
                         "edit to config.py. Example: --unerase tubule_sain "
                         "tubule_atrophique")
    ap.add_argument("--border-threshold", type=float,
                    default=P["border_threshold"],
                    help="drop veine/nsg/intima/media touching the image edge by "
                         "more than this. Set 1.0 to disable the stage. On a "
                         "perfect input this filter is the ONLY thing the "
                         "pipeline loses, so it is worth measuring both ways.")
    ap.add_argument("--cortex-min-inside", type=float,
                    default=P["cortex_min_inside"],
                    help="drop instances with less than this fraction of "
                         "themselves inside the annotated cortex. 0 disables it.")
    ap.add_argument("--class-fusion-prefer", default=P["class_fusion_prefer"],
                    choices=["score", "whole_view"],
                    help="which member of a fused nsg_complet/nsg_partiel pair "
                         "gives the surviving label")
    ap.add_argument("--tag", default="",
                    help="suffix for the output folder, so one configuration "
                         "never overwrites another. e.g. --tag _base030")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.erases = dict((k, list(v)) for k, v in P["erases"].items())
    args.pairs = [tuple(p) for p in (args.score_resolve or [])]
    for a, b in args.pairs:
        # a precedence rule and a score resolution for the same pair would
        # fight, and precedence runs first, so the rule goes
        for x, y in ((a, b), (b, a)):
            if x in args.erases and y in args.erases[x]:
                args.erases[x] = [c for c in args.erases[x] if c != y]
                print("priority table: %s no longer erases %s" % (x, y))
        print("score resolution: %s against %s, higher score kept" % (a, b))
    for a, b in (args.unerase or []):
        if a not in args.erases:
            raise SystemExit("unknown class %s" % a)
        if b not in args.erases[a]:
            print("note: %s did not erase %s to begin with" % (a, b))
        args.erases[a] = [c for c in args.erases[a] if c != b]
        print("priority table: %s no longer erases %s" % (a, b))

    args.per_class_thr = {}
    if args.thresholds:
        with open(args.thresholds) as f:
            args.per_class_thr = json.load(f).get("thresholds", {})
        print("per-class thresholds loaded from %s" % args.thresholds)

    work = Path(args.work_dir)
    if args.in_dir is None:
        args.in_dir = str(work / "s06_stitch" / args.run / args.split / "stitched")
    if args.out_dir is None:
        args.out_dir = str(work / "s07_postproc" / args.run /
                           (args.split + args.tag))

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    filt_dir = out_dir / "filtered"
    filt_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise SystemExit("no stitched files in %s" % in_dir)

    print("=" * 76)
    print("stage 07 - post-processing | run %s  split %s  sources %d"
          % (args.run, args.split, len(files)))
    print("min_confidence %.2f   stop after stage %d" %
          (args.min_confidence, args.stop_after))
    if args.fuse_mask is not None or args.fuse_bb is not None:
        print("fusion thresholds: bb %.2f  mask %.2f  (inherited %.2f / %.2f)"
              % (P["fuse_bb"] if args.fuse_bb is None else args.fuse_bb,
                 P["fuse_mask"] if args.fuse_mask is None else args.fuse_mask,
                 P["fuse_bb"], P["fuse_mask"]))
    print("class fusion prefers: %s   border threshold %.2f%s"
          % (args.class_fusion_prefer, args.border_threshold,
             " (disabled)" if args.border_threshold >= 1.0 else ""))
    if args.per_class_thr:
        print("overrides: " + ", ".join(
            "%s %.2f" % (k, v) for k, v in sorted(args.per_class_thr.items())))
    print("=" * 76)

    totals = OrderedDict()
    per_class = defaultdict(int)
    n_sources = 0

    xml_of = {}
    if args.cortex_min_inside > 0:
        with open(str(work / "s02_split" / "s02_split.csv")) as f:
            xml_of = dict((r["source_key"], r["xml_path"])
                          for r in csv.DictReader(f) if r["split"] == args.split)
        print("cortex filter ON, keeping instances >= %.2f inside"
              % args.cortex_min_inside)

    for i, pf in enumerate(files, 1):
        with open(str(pf)) as f:
            rec = json.load(f)
        cortex = None
        if args.cortex_min_inside > 0:
            xp = xml_of.get(rec["source_key"])
            if xp:
                cortex = cortex_rle(Path(xp), rec["height"], rec["width"])
        inst, counts = process(rec, args, cortex)
        n_sources += 1

        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v

        out = []
        for it in inst:
            per_class[it["class_name"]] += 1
            out.append({
                "class_name": it["class_name"],
                "category_id": int(it["category_id"]),
                "score": float(it["score"]),
                "bbox": [float(v) for v in it["bbox"]],
                "area": float(maskUtils.area(it["_rle"])),
                "mask_rle": rle_out(it["_rle"]),
            })
        with open(str(filt_dir / ("%s.json" % rec["source_key"])), "w") as f:
            json.dump(OrderedDict(source_key=rec["source_key"],
                                  height=rec["height"], width=rec["width"],
                                  n_instances=len(out), instances=out), f)
        print("[%d/%d] %-16s %5d -> %4d" % (i, len(files), rec["source_key"],
                                            counts["input"], len(out)))

    summary = OrderedDict(n_sources=n_sources, funnel=totals,
                          instances_per_class=OrderedDict(
                              (c, int(per_class.get(c, 0)))
                              for c in C.STRUCTURE_CLASSES))
    with open(str(out_dir / "s07_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 76)
    print("%-18s%10s%10s" % ("stage", "kept", "removed"))
    print("-" * 76)
    prev = totals.get("input", 0)
    print("%-18s%10d" % ("input", prev))
    for name in STAGE_NAMES:
        if name not in totals:
            break
        cur = totals[name]
        print("%-18s%10d%10d" % (name, cur, prev - cur))
        prev = cur
    print("-" * 76)
    for c in C.STRUCTURE_CLASSES:
        print("   %-20s%7d" % (c, per_class.get(c, 0)))
    print("=" * 76)
    C.write_run_meta(out_dir, "s07_postproc", args, extra=summary)


if __name__ == "__main__":
    main()