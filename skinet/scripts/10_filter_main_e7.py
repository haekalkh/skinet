"""
E7 Stage 10 - Priority filter + Orphan/hierarchy filter (MAIN).

Config-driven (imports e7_config). Final cleanup of the fused MAIN instances from
Stage 9. Order: cleanup -> priority -> orphan.

  (0) CLEANUP  (POSTFUSE_CLEANUP):
      Drop fused instances below a confidence floor (inference ran at score_thr
      0.30 to feed fusion; 0.5 is re-applied here) and below a per-class minimum
      mask area. This removes the scattered low-score / tiny fragments.

  (1) PRIORITY FILTER  (PRIORITY_FILTER):
      For two instances of DIFFERENT classes that strongly co-locate
      (bbox IoU >= bb_threshold AND mask IoMin >= mask_threshold), keep the
      higher-score one and drop the other. This is a hierarchy-aware cross-class
      NMS: it does NOT touch
        - same-class instances (already unioned in Stage 9),
        - compatibility-group pairs (already grade-resolved in Stage 9),
        - legal parent-child pairs (intima/media inside vaisseau IS expected).
      The high bbox gate (0.5) protects adjacent-but-distinct structures, which
      is why this succeeds where the earlier blanket class-agnostic NMS failed.

  (2) ORPHAN / HIERARCHY FILTER  (ORPHAN_FILTER + CLASSES_HIERARCHY):
      A child (intima/media) is kept only if it is contained inside a parent
      (vaisseau): containment = intersection / child_area >= mask_threshold, with
      a bbox gate. Children inside no parent are orphans -> removed (the vessel
      false-positive fix). Parents are kept regardless when keep_if_no_child.

Input : OUT_MAIN/postproc/{split}/09_fused/{source}.json
Output: OUT_MAIN/postproc/{split}/10_filtered/{source}.json + summary
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

from pycocotools import mask as mask_util

import e7_config as C


OVERLAP_METRIC = "iomin"   # match Stage 9


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rle_in(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def area(rle):
    return float(mask_util.area(rle))


def inter_area(a, b):
    return float(mask_util.area(mask_util.merge([a, b], intersect=True)))


def iomin(a, b):
    i = inter_area(a, b)
    m = min(area(a), area(b))
    return i / m if m > 0 else 0.0


def containment(child, parent):
    """Fraction of the CHILD that lies inside the PARENT."""
    ca = area(child)
    return inter_area(child, parent) / ca if ca > 0 else 0.0


def bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


# --- hierarchy / compatibility lookups (by category_id) ---
def build_lookups():
    parent_to_children, child_to_parents = {}, defaultdict(set)
    keep_if_no_child = {}
    for parent, spec in C.CLASSES_HIERARCHY.items():
        pid = C.MAIN_ID[parent]
        kids = {C.MAIN_ID[c] for c in spec["contains"]}
        parent_to_children[pid] = kids
        keep_if_no_child[pid] = bool(spec.get("keep_if_no_child", True))
        for k in kids:
            child_to_parents[k].add(pid)
    compat = []
    for group in C.CLASSES_COMPATIBILITY:
        compat.append({C.MAIN_ID[c] for c in group})
    return parent_to_children, child_to_parents, keep_if_no_child, compat


def same_compat_group(a, b, compat):
    return any(a in g and b in g for g in compat)


def parent_child_related(a, b, parent_to_children):
    return (b in parent_to_children.get(a, set())
            or a in parent_to_children.get(b, set()))


def cleanup_filter(instances):
    """Drop low-confidence and tiny fragments BEFORE priority/orphan. This is the
    main fix for scattered low-score fragments that fusion kept."""
    cfg = C.POSTFUSE_CLEANUP
    by_class = cfg.get("min_area_px_by_class", {})
    kept, n_score, n_area = [], 0, 0
    for it in instances:
        if it["score"] < cfg["min_confidence"]:
            n_score += 1
            continue
        a = area(rle_in(it["mask_rle"]))
        thr = by_class.get(it["class_name"], cfg["min_area_px"])
        if a < thr:
            n_area += 1
            continue
        kept.append(it)
    return kept, n_score, n_area


def priority_filter(instances, parent_to_children, compat):
    cfg = C.PRIORITY_FILTER
    n = len(instances)
    rles = [rle_in(it["mask_rle"]) for it in instances]
    removed = [False] * n
    n_removed = 0
    for i in range(n):
        if removed[i]:
            continue
        for j in range(i + 1, n):
            if removed[j]:
                continue
            ci, cj = instances[i]["category_id"], instances[j]["category_id"]
            if ci == cj:
                continue                                    # same class: skip
            if same_compat_group(ci, cj, compat):
                continue                                    # grade pair: skip
            if parent_child_related(ci, cj, parent_to_children):
                continue                                    # legal nesting: skip
            if bbox_iou(instances[i]["bbox_global"],
                        instances[j]["bbox_global"]) < cfg["bb_threshold"]:
                continue
            if iomin(rles[i], rles[j]) < cfg["mask_threshold"]:
                continue
            # conflict: drop the lower-score instance
            if instances[i]["score"] >= instances[j]["score"]:
                removed[j] = True
            else:
                removed[i] = True
                break
            n_removed += 1
    kept = [instances[k] for k in range(n) if not removed[k]]
    return kept, n_removed


def orphan_filter(instances, parent_to_children, child_to_parents, keep_if_no_child):
    cfg = C.ORPHAN_FILTER
    rles = [rle_in(it["mask_rle"]) for it in instances]
    parent_ids = set(parent_to_children.keys())
    child_ids = set(child_to_parents.keys())

    parent_idx_by_class = defaultdict(list)
    for i, it in enumerate(instances):
        if it["category_id"] in parent_ids:
            parent_idx_by_class[it["category_id"]].append(i)

    kept_flags = [True] * len(instances)
    n_orphans = 0
    for i, it in enumerate(instances):
        cid = it["category_id"]
        if cid not in child_ids:
            continue                                   # not a hierarchy child
        allowed_parents = child_to_parents[cid]
        inside = False
        for pid in allowed_parents:
            for pj in parent_idx_by_class.get(pid, []):
                if bbox_iou(it["bbox_global"],
                            instances[pj]["bbox_global"]) < cfg["bb_threshold"]:
                    continue
                if containment(rles[i], rles[pj]) >= cfg["mask_threshold"]:
                    inside = True
                    break
            if inside:
                break
        if not inside:
            if it["score"] >= cfg.get("keep_if_score_ge", 1.0):
                continue                    # confident wall: keep despite no parent
            kept_flags[i] = False
            n_orphans += 1

    # parents: keep_if_no_child handling (default keep)
    n_childless_parents_removed = 0
    kept = [instances[k] for k in range(len(instances)) if kept_flags[k]]
    return kept, n_orphans, n_childless_parents_removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "postproc" / args.split / "09_fused")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "postproc" / args.split / "10_filtered")
    ensure_dir(out_dir)

    files = sorted(in_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("09_")]
    if not files:
        raise FileNotFoundError(f"No Stage 9 fused files in {in_dir}")

    p2c, c2p, keep_nc, compat = build_lookups()

    print("=" * 80)
    print(f"E7 Stage 10 - Cleanup + Priority + Orphan filter | split={args.split}")
    print(f"  cleanup: score>={C.POSTFUSE_CLEANUP['min_confidence']} "
          f"area>={C.POSTFUSE_CLEANUP['min_area_px']}px "
          f"(per-class: {C.POSTFUSE_CLEANUP.get('min_area_px_by_class', {})})")
    print(f"  priority: bb>={C.PRIORITY_FILTER['bb_threshold']} "
          f"iomin>={C.PRIORITY_FILTER['mask_threshold']}")
    print(f"  orphan: bb>={C.ORPHAN_FILTER['bb_threshold']} "
          f"contain>={C.ORPHAN_FILTER['mask_threshold']} "
          f"keep_if_score>={C.ORPHAN_FILTER.get('keep_if_score_ge')}")
    print(f"  hierarchy: {C.CLASSES_HIERARCHY}")
    print("=" * 80)

    summary = OrderedDict(split=args.split, num_sources=0,
                          total_in=0, total_after_cleanup=0,
                          total_cleanup_score_removed=0, total_cleanup_area_removed=0,
                          total_after_priority=0, total_after_orphan=0,
                          total_priority_removed=0, total_orphans_removed=0,
                          per_class_final=defaultdict(int), sources=[])

    for pf in files:
        with open(pf) as f:
            rec = json.load(f)
        insts = rec["instances"]
        after_clean, n_score, n_area = cleanup_filter(insts)
        after_prio, n_prio = priority_filter(after_clean, p2c, compat)
        after_orph, n_orph, _ = orphan_filter(after_prio, p2c, c2p, keep_nc)

        per_class = defaultdict(int)
        for it in after_orph:
            per_class[it["class_name"]] += 1
            summary["per_class_final"][it["class_name"]] += 1

        out = OrderedDict(source_key=rec["source_key"], height=rec["height"],
                          width=rec["width"],
                          num_in=len(insts),
                          num_after_cleanup=len(after_clean),
                          num_cleanup_score_removed=n_score,
                          num_cleanup_area_removed=n_area,
                          num_after_priority=len(after_prio),
                          num_priority_removed=n_prio,
                          num_after_orphan=len(after_orph),
                          num_orphans_removed=n_orph,
                          per_class=dict(per_class), instances=after_orph)
        with open(out_dir / f"{rec['source_key']}.json", "w") as f:
            json.dump(out, f)

        summary["sources"].append({
            "source_key": rec["source_key"], "in": len(insts),
            "after_cleanup": len(after_clean),
            "cleanup_score_removed": n_score, "cleanup_area_removed": n_area,
            "after_priority": len(after_prio), "priority_removed": n_prio,
            "after_orphan": len(after_orph), "orphans_removed": n_orph})
        summary["total_in"] += len(insts)
        summary["total_after_cleanup"] += len(after_clean)
        summary["total_cleanup_score_removed"] += n_score
        summary["total_cleanup_area_removed"] += n_area
        summary["total_after_priority"] += len(after_prio)
        summary["total_after_orphan"] += len(after_orph)
        summary["total_priority_removed"] += n_prio
        summary["total_orphans_removed"] += n_orph
        summary["num_sources"] += 1
        print(f"  {rec['source_key']}: {len(insts)} -> {len(after_clean)} "
              f"(clean -{n_score + n_area}) -> {len(after_prio)} (prio -{n_prio}) "
              f"-> {len(after_orph)} (orphan -{n_orph})")

    summary["per_class_final"] = {c: int(summary["per_class_final"].get(c, 0))
                                  for c in C.MAIN_CLASSES}
    with open(out_dir / "10_filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"In {summary['total_in']} -> cleanup {summary['total_after_cleanup']} "
          f"(-{summary['total_cleanup_score_removed']} score "
          f"-{summary['total_cleanup_area_removed']} area) -> priority "
          f"{summary['total_after_priority']} (-{summary['total_priority_removed']}) "
          f"-> orphan {summary['total_after_orphan']} "
          f"(-{summary['total_orphans_removed']})")
    print(f"Per-class final: {summary['per_class_final']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()