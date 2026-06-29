"""
E7 Stage 9 - Fusion (same-class union + cross-class grade resolution).

Config-driven (imports e7_config). Collapses the redundant global instances from
Stage 8 using the MRCNN post-processing rules:

  (a) SAME-CLASS UNION  (FUSE_SAME_CLASS):
      Within each class, instances that overlap (tile-overlap duplicates / split
      fragments) are clustered and merged by mask UNION. Overlap is measured by
      intersection-over-min-area (IoMin), which is lenient enough to merge
      adjacent fragments of one object split across tiles. Score = max of cluster.

  (b) CROSS-CLASS GRADE RESOLUTION  (CROSS_CLASS_FUSION + CLASSES_COMPATIBILITY):
      Within a compatibility group ([nsg, nsg_partiel, nsg_complet] and
      [tubule_sain, tubule_atrophique]), instances of DIFFERENT classes that
      overlap are the SAME physical object predicted with different grades. Keep
      ONE -- the higher-score label wins (its class + mask). This kills the
      grade-duplicate over-prediction seen in the confusion analysis.

Order matters (POSTPROC_CHAIN): same-class union first, then cross-class.
No priority/orphan filtering here -- that is Stage 10.

Input : OUT_MAIN/postproc/{split}/08_stitched/{source}.json
Output: OUT_MAIN/postproc/{split}/09_fused/{source}.json + summary
"""

import argparse
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

from pycocotools import mask as mask_util

import e7_config as C


# overlap metric for fusion: "iomin" (intersection/min area) or "iou"
OVERLAP_METRIC = "iomin"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rle_in(rle):
    return {"size": rle["size"], "counts": rle["counts"].encode("ascii")}


def rle_out(rle):
    return {"size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": rle["counts"].decode("ascii")}


def area(rle):
    return float(mask_util.area(rle))


def overlap(a, b, metric=OVERLAP_METRIC):
    inter = float(mask_util.area(mask_util.merge([a, b], intersect=True)))
    if inter <= 0:
        return 0.0
    aa, ab = area(a), area(b)
    if metric == "iou":
        union = aa + ab - inter
        return inter / union if union > 0 else 0.0
    return inter / min(aa, ab) if min(aa, ab) > 0 else 0.0  # iomin


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


def bbox_overlaps(a, b):
    """Cheap gate: do the two bboxes intersect at all?"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _gate_pass(a, b, mode, thr):
    if mode == "intersect":
        return bbox_overlaps(a, b)
    return bbox_iou(a, b) >= thr            # "iou"


def connected_components(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i, j in edges:
        union(i, j)
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


def cluster_indices(instances, idxs, bb_thr, mask_thr, gate_mode="iou"):
    """Edges where the bbox gate passes (cheap) AND mask overlap >= mask_thr."""
    edges = []
    rles = [rle_in(instances[i]["mask_rle"]) for i in idxs]
    bxs = [instances[i]["bbox_global"] for i in idxs]
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            if not _gate_pass(bxs[a], bxs[b], gate_mode, bb_thr):
                continue
            if overlap(rles[a], rles[b]) >= mask_thr:
                edges.append((a, b))
    comps = connected_components(len(idxs), edges)
    return [[idxs[k] for k in comp] for comp in comps], rles


def fuse_same_class(instances):
    cfg = C.FUSE_SAME_CLASS
    by_class = defaultdict(list)
    for i, it in enumerate(instances):
        by_class[it["category_id"]].append(i)

    fused = []
    for cid, idxs in by_class.items():
        comps, _ = cluster_indices(instances, idxs,
                                   cfg["bb_threshold"], cfg["mask_threshold"],
                                   gate_mode=cfg.get("bbox_gate", "iou"))
        for comp in comps:
            members = [instances[i] for i in comp]
            best = max(members, key=lambda m: m["score"])
            union_rle = mask_util.merge(
                [rle_in(m["mask_rle"]) for m in members], intersect=False)
            x, y, w, h = [float(v) for v in mask_util.toBbox(union_rle)]
            fused.append({
                "category_id": best["category_id"],
                "class_name": best["class_name"],
                "score": float(max(m["score"] for m in members)),
                "bbox_global": [x, y, w, h],
                "mask_rle": rle_out(union_rle),
                "_n_merged": len(members),
            })
    return fused


def pick_winner(members, resolution):
    """Pick the surviving instance for a cross-class cluster.
      "score"                              -> highest score
      [class, ...]                         -> earliest-listed class present
      {"priority":[...], "score_margin":m,
       "score_offset":{cls:off}}           -> compare CALIBRATED scores
                                              (score - per-class offset); among
                                              members within m of the best
                                              calibrated score, earliest-listed
                                              class wins; else highest calibrated.
    A positive offset pulls a class down (it must out-score others by `offset`
    to win) -- used to undo the inflated nsg_complet confidence after the
    class-weighted retrain, without another retrain."""
    if isinstance(resolution, dict):
        prio = resolution["priority"]
        margin = float(resolution.get("score_margin", 0.0))
        offset = resolution.get("score_offset", {}) or {}

        def eff(m):
            return m["score"] - float(offset.get(m["class_name"], 0.0))

        best = max(eff(m) for m in members)
        eligible = [m for m in members if eff(m) >= best - margin]
        for cls in prio:
            cands = [m for m in eligible if m["class_name"] == cls]
            if cands:
                return max(cands, key=eff)
        return max(members, key=eff)
    if isinstance(resolution, list):
        for cls in resolution:
            cands = [m for m in members if m["class_name"] == cls]
            if cands:
                return max(cands, key=lambda m: m["score"])
        return max(members, key=lambda m: m["score"])
    return max(members, key=lambda m: m["score"])     # "score"


def fuse_cross_class(instances):
    cfg = C.CROSS_CLASS_FUSION
    resolutions = getattr(C, "CROSS_CLASS_RESOLUTION", None)
    # map each class to its compatibility group id (classes not in any group are
    # left untouched)
    group_of = {}
    for gi, group in enumerate(C.CLASSES_COMPATIBILITY):
        for cls in group:
            group_of[C.MAIN_ID[cls]] = gi

    grouped = defaultdict(list)   # group_id -> instance indices
    untouched = []
    for i, it in enumerate(instances):
        gid = group_of.get(it["category_id"])
        if gid is None:
            untouched.append(i)
        else:
            grouped[gid].append(i)

    resolved = [instances[i] for i in untouched]
    for gid, idxs in grouped.items():
        comps, _ = cluster_indices(instances, idxs,
                                   cfg["bb_threshold"], cfg["mask_threshold"],
                                   gate_mode=cfg.get("bbox_gate", "iou"))
        res = (resolutions[gid] if (resolutions and gid < len(resolutions))
               else "score")
        for comp in comps:
            members = [instances[i] for i in comp]
            resolved.append(pick_winner(members, res))   # grade-priority or score
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--glom-score-margin", type=float, default=None,
                    help="override glomerulus grade score_margin (for val sweeps)")
    ap.add_argument("--complet-offset", type=float, default=None,
                    help="override nsg_complet score offset (pull complet down)")
    ap.add_argument("--partiel-offset", type=float, default=None,
                    help="override nsg_partiel score offset")
    args = ap.parse_args()

    if args.glom_score_margin is not None:
        for r in C.CROSS_CLASS_RESOLUTION:
            if isinstance(r, dict):
                r["score_margin"] = args.glom_score_margin
        print(f"[override] glomerulus score_margin = {args.glom_score_margin}")
    if args.complet_offset is not None or args.partiel_offset is not None:
        for r in C.CROSS_CLASS_RESOLUTION:
            if isinstance(r, dict):
                off = r.setdefault("score_offset", {})
                if args.complet_offset is not None:
                    off["nsg_complet"] = args.complet_offset
                if args.partiel_offset is not None:
                    off["nsg_partiel"] = args.partiel_offset
        print(f"[override] score_offset = "
              f"complet {args.complet_offset} partiel {args.partiel_offset}")

    in_dir = Path(args.in_dir) if args.in_dir else (C.OUT_MAIN / "postproc" / args.split / "08_stitched")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "postproc" / args.split / "09_fused")
    ensure_dir(out_dir)

    files = sorted(in_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("08_")]
    if not files:
        raise FileNotFoundError(f"No Stage 8 stitched files in {in_dir}")

    print("=" * 80)
    print(f"E7 Stage 9 - Fusion | split={args.split}  sources={len(files)}")
    print(f"  same-class: bb>={C.FUSE_SAME_CLASS['bb_threshold']} "
          f"mask{OVERLAP_METRIC}>={C.FUSE_SAME_CLASS['mask_threshold']}")
    print(f"  cross-class groups: {C.CLASSES_COMPATIBILITY}")
    print("=" * 80)

    summary = OrderedDict(split=args.split, overlap_metric=OVERLAP_METRIC,
                          num_sources=0, total_pre=0, total_after_same=0,
                          total_after_cross=0,
                          per_class_final=defaultdict(int), sources=[])

    for pf in files:
        with open(pf) as f:
            rec = json.load(f)
        pre = rec["instances"]
        after_same = fuse_same_class(pre)
        after_cross = fuse_cross_class(after_same)

        per_class = defaultdict(int)
        for it in after_cross:
            per_class[it["class_name"]] += 1
            summary["per_class_final"][it["class_name"]] += 1

        out = OrderedDict(source_key=rec["source_key"], height=rec["height"],
                          width=rec["width"],
                          num_pre_fusion=len(pre),
                          num_after_same_class=len(after_same),
                          num_after_cross_class=len(after_cross),
                          per_class=dict(per_class), instances=after_cross)
        with open(out_dir / f"{rec['source_key']}.json", "w") as f:
            json.dump(out, f)

        summary["sources"].append({
            "source_key": rec["source_key"], "pre": len(pre),
            "after_same": len(after_same), "after_cross": len(after_cross)})
        summary["total_pre"] += len(pre)
        summary["total_after_same"] += len(after_same)
        summary["total_after_cross"] += len(after_cross)
        summary["num_sources"] += 1
        print(f"  {rec['source_key']}: {len(pre)} -> {len(after_same)} "
              f"(same) -> {len(after_cross)} (cross)")

    summary["per_class_final"] = {c: int(summary["per_class_final"].get(c, 0))
                                  for c in C.MAIN_CLASSES}
    with open(out_dir / "09_fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Pre {summary['total_pre']} -> same {summary['total_after_same']} "
          f"-> cross {summary['total_after_cross']}")
    print(f"Per-class final: {summary['per_class_final']}")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()