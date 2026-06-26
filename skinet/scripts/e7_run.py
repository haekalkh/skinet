#!/usr/bin/env python
"""
E7 end-to-end single-process runner.

One model load, one image in -> final reconciled instance masks out, in ONE
process with in-memory hand-off between the validated stages:

    infer (07) -> stitch (08) -> fuse same/cross (09)
        -> cleanup/priority/orphan (10)  [-> SAM-Med2D refine|replace (12)]
        -> tissue filter (NEW)  -> write JSON [+ overlay]

It does NOT reimplement any stage. It imports the real stage modules from
--scripts-dir and calls their per-source functions, so the output is identical to
the staged 07->10(->12) pipeline that Stage 14 already evaluates.

SAM has two modes (--sam-mode):
  refine  (default) : do-no-harm gated refinement; SAM only replaces a mask when
                      it passes the IoU/trim/single-component gate, else the M2F
                      mask is kept. This is the validated Stage-12 behaviour.
  replace (EXPERIMENT): use M2F purely as a box detector. Every eligible
                      instance's mask is REPLACED by SAM's mask (no gate). If SAM
                      returns an empty/None mask, fall back to the M2F mask so no
                      instance is lost. Answers: "is SAM-Med2D + Kidney Adapter a
                      good enough stand-alone segmenter, given only the box?"

Timing + peak GPU (nvidia-smi, this PID) are built in.
"""
import os
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import argparse
import importlib.util
import json
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
class GPUSampler(threading.Thread):
    """Polls nvidia-smi for THIS process's GPU memory; records peak (MB)."""
    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mb = 0
        self._stop = False
        self._pid = os.getpid()

    def _sample(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode()
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == self._pid:
                    self.peak_mb = max(self.peak_mb, int(parts[1]))
                    return
        except Exception:
            pass

    def run(self):
        while not self._stop:
            self._sample()
            time.sleep(self.interval)

    def stop(self):
        self._stop = True


def load_module(path, name):
    path = str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- a small palette for the optional overlay (decode-only convenience) ----
PALETTE = {
    "tubule_sain": (230, 126, 60), "tubule_atrophique": (123, 90, 200),
    "vaisseau": (40, 160, 70), "nsg": (40, 200, 210), "intima": (235, 30, 110),
    "nsg_partiel": (240, 200, 30), "media": (70, 130, 235), "veine": (150, 150, 150),
    "nsg_complet": (180, 40, 200), "pac": (20, 160, 140),
}


def render_overlay(image_bgr, instances, out_path, alpha=0.45):
    import cv2
    from pycocotools import mask as mask_util
    over = image_bgr.copy()
    for it in instances:
        rle = {"size": it["mask_rle"]["size"],
               "counts": it["mask_rle"]["counts"].encode("ascii")}
        m = mask_util.decode(rle).astype(bool)
        if m.sum() == 0:
            continue
        r, g, b = PALETTE.get(it["class_name"], (255, 0, 0))
        over[m] = ((1 - alpha) * over[m] + alpha * np.array([b, g, r])).astype(np.uint8)
    cv2.imwrite(str(out_path), over)


# ---------------------------------------------------------------------------
def build_sam(s12, args, device):
    """Reconstruct the Stage-12 SAM model + Kidney Adapter exactly as Stage 12 does."""
    ns = argparse.Namespace(image_size=args.sam_target, sam_checkpoint=args.sam_checkpoint,
                            encoder_adapter=False)
    model = s12.sam_model_registry[args.sam_model_type](ns).to(device)
    adapter = s12.build_kidney_adapter(variant="B", d_model=256,
                                       bottleneck=64, dropout=0.1).to(device)
    ck = torch.load(args.sam_best, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    adapter.load_state_dict(ck["adapter"])
    model.eval(); adapter.eval()
    return model, adapter


def _refine_set(args):
    return (set(__import__("e7_config").MAIN_CLASSES)
            if args.refine_classes.strip().lower() == "all"
            else {c.strip() for c in args.refine_classes.split(",") if c.strip()})


def sam_refine(s12, model, adapter, image_bgr, instances, args, device,
               progress_every=20):
    """Stage-12 do-no-harm GATED refinement (default mode). Unchanged behaviour:
    SAM mask is accepted only if single-component, area ratio within trim band,
    and IoU >= floor; otherwise the original M2F mask is kept."""
    refine_set = _refine_set(args)
    total = sum(1 for it in instances if it["class_name"] in refine_set)
    print(f"    [12] SAM refine (gated): {total} eligible instances "
          f"(classes: {sorted(refine_set)})", flush=True)
    n_done, n_acc = 0, 0
    for it in instances:
        it["refined"] = False
        if it["class_name"] not in refine_set:
            continue
        m2f = s12.rle_decode(it["mask_rle"])
        sam = s12.refine_instance(model, adapter, image_bgr, it["bbox_global"],
                                  args.sam_roi_pad, args.sam_target, device)
        n_done += 1
        if not (sam is None or sam.sum() == 0):
            a_s, a_m = float(sam.sum()), float(m2f.sum())
            if not (a_m == 0 or a_s < args.sam_min_area):
                j = s12.iou(sam, m2f)
                ncc = s12.n_components(m2f)
                ratio = a_s / a_m
                if ncc <= 1 and args.sam_trim_lo <= ratio <= args.sam_trim_hi and j >= args.sam_iou_lo:
                    ys, xs = np.where(sam)
                    x0, y0 = int(xs.min()), int(ys.min())
                    it["mask_rle"] = s12.rle_encode(sam)
                    it["bbox_global"] = [x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)]
                    it["refined"] = True
                    n_acc += 1
        if progress_every and (n_done % progress_every == 0 or n_done == total):
            print(f"      [12] SAM {n_done}/{total} "
                  f"({100 * n_done / max(1, total):.0f}%)  accepted: {n_acc}",
                  flush=True)
    return instances


def sam_replace(s12, model, adapter, image_bgr, instances, args, device,
                progress_every=20):
    """EXPERIMENT mode: M2F is a box-only detector; SAM is the sole segmenter.

    For every eligible instance, REPLACE its mask with SAM's mask (NO gate).
    If SAM returns empty/None, keep the M2F mask (fallback) so no instance is
    lost. Records mask_source so you can audit how often SAM actually produced
    the mask vs fell back."""
    refine_set = _refine_set(args)
    total = sum(1 for it in instances if it["class_name"] in refine_set)
    print(f"    [12] SAM replace (box->SAM, no gate): {total} eligible instances "
          f"(classes: {sorted(refine_set)})", flush=True)
    n_done = n_sam = n_fb = 0
    for it in instances:
        it["refined"] = False
        it["mask_source"] = "m2f"          # default for non-eligible classes
        if it["class_name"] not in refine_set:
            continue
        sam = s12.refine_instance(model, adapter, image_bgr, it["bbox_global"],
                                  args.sam_roi_pad, args.sam_target, device)
        n_done += 1
        if sam is not None and sam.sum() >= args.sam_min_area:
            # use SAM's mask outright
            ys, xs = np.where(sam)
            x0, y0 = int(xs.min()), int(ys.min())
            it["mask_rle"] = s12.rle_encode(sam)
            it["bbox_global"] = [x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1)]
            it["refined"] = True
            it["mask_source"] = "sam"
            n_sam += 1
        else:
            # SAM failed -> keep the M2F mask (instance not lost)
            it["mask_source"] = "m2f_fallback"
            n_fb += 1
        if progress_every and (n_done % progress_every == 0 or n_done == total):
            print(f"      [12] SAM {n_done}/{total} "
                  f"({100 * n_done / max(1, total):.0f}%)  "
                  f"sam: {n_sam}  fallback: {n_fb}", flush=True)
    print(f"    [12] replace summary: {n_sam} from SAM, {n_fb} fell back to M2F "
          f"(of {total} eligible)", flush=True)
    return instances


# ---------------------------------------------------------------------------
def run_one(modules, predictor, image, src, tile, stride, score_thr,
            sam_ctx=None, tissue_cfg=None, tissue_debug_path=None):
    """The full in-memory chain for ONE source. Returns (final_instances, H, W, n_tiles)."""
    s07, s08, s09, s10 = modules["s07"], modules["s08"], modules["s09"], modules["s10"]
    oh, ow = image.shape[:2]
    t_src = time.time()
    print(f"  === {src} ({ow}x{oh}) ===", flush=True)

    t = time.time()
    instances, n_tiles = s07.infer_source(predictor, image, tile, stride,
                                          score_thr, src=src)
    print(f"    [07] infer done: {n_tiles} tiles -> {len(instances)} raw "
          f"instances ({time.time() - t:.1f}s)", flush=True)

    rec = OrderedDict(source_key=src, orig_width=int(ow), orig_height=int(oh),
                      tile_size=tile, stride=stride, num_tiles=n_tiles,
                      num_instances=len(instances), instances=instances)

    t = time.time()
    stitched, H, W = s08.stitch_source(rec)
    print(f"    [08] stitch done: {len(stitched)} instances on {W}x{H} "
          f"({time.time() - t:.1f}s)", flush=True)

    t = time.time()
    fused = s09.fuse_cross_class(s09.fuse_same_class(stitched))
    print(f"    [09] fuse done: {len(stitched)} -> {len(fused)} instances "
          f"({time.time() - t:.1f}s)", flush=True)

    p2c, c2p, keep_nc, compat = s10.build_lookups()
    t = time.time()
    clean, _, _ = s10.cleanup_filter(fused)
    print(f"    [10a] cleanup: {len(fused)} -> {len(clean)} "
          f"({time.time() - t:.1f}s)", flush=True)
    t = time.time()
    prio, _ = s10.priority_filter(clean, p2c, compat)
    print(f"    [10b] priority: {len(clean)} -> {len(prio)} "
          f"({time.time() - t:.1f}s)", flush=True)
    t = time.time()
    final, _, _ = s10.orphan_filter(prio, p2c, c2p, keep_nc)
    print(f"    [10c] orphan: {len(prio)} -> {len(final)} "
          f"({time.time() - t:.1f}s)", flush=True)

    if sam_ctx is not None:
        s12, model, adapter, args, device = sam_ctx
        t = time.time()
        if args.sam_mode == "replace":
            final = sam_replace(s12, model, adapter, image, final, args, device)
            n_sam = sum(1 for it in final if it.get("mask_source") == "sam")
            print(f"    [12] SAM done (replace): {n_sam}/{len(final)} masks from SAM "
                  f"({time.time() - t:.1f}s)", flush=True)
        else:
            final = sam_refine(s12, model, adapter, image, final, args, device)
            n_ref = sum(1 for it in final if it.get("refined"))
            print(f"    [12] SAM done (refine): {n_ref}/{len(final)} refined "
                  f"({time.time() - t:.1f}s)", flush=True)

    # --- spatial guard: drop biopsy-edge "flat blob" masks that sit off-tissue ---
    if tissue_cfg is not None and tissue_cfg.get("enable", True):
        import tissue_filter
        t = time.time()
        n_before = len(final)
        final = tissue_filter.filter_outside_tissue(
            image, final, tissue_cfg, debug_overlay_path=tissue_debug_path)
        print(f"    [tissue] {n_before} -> {len(final)} "
              f"(dropped {n_before - len(final)} off-tissue) "
              f"({time.time() - t:.1f}s)", flush=True)

    print(f"  === {src} complete: {len(final)} final instances "
          f"(total {time.time() - t_src:.1f}s) ===", flush=True)
    return final, H, W, n_tiles


def main():
    ap = argparse.ArgumentParser("E7 end-to-end single-process runner")
    ap.add_argument("--image", default=None, help="path to a source image (overrides lookup)")
    ap.add_argument("--source", default=None, help="source_key, or comma-separated keys")
    ap.add_argument("--all", action="store_true", help="run every source of --split")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent),
                    help="dir with 7_/8_/9_/10_/12_ stage scripts + e7_config.py")
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--overlap-ratio", type=float, default=None)
    ap.add_argument("--score-thr", type=float, default=0.30)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the untimed warm-up source (benchmark accuracy suffers)")
    # SAM (Stage 12) options -- mirror Stage 12 defaults
    ap.add_argument("--sam", action="store_true", help="apply Stage-12 SAM-Med2D")
    ap.add_argument("--sam-mode", default="refine", choices=["refine", "replace"],
                    help="refine = gated do-no-harm (default); "
                         "replace = box-only detector, SAM is the sole segmenter (EXPERIMENT)")
    ap.add_argument("--sam-best", default=None)
    ap.add_argument("--sam-checkpoint", default=None)
    ap.add_argument("--sam-model-type", default="vit_b")
    ap.add_argument("--sam-target", type=int, default=256)
    ap.add_argument("--sam-roi-pad", type=float, default=0.30)
    ap.add_argument("--refine-classes", default="all")
    ap.add_argument("--sam-trim-lo", type=float, default=0.60)
    ap.add_argument("--sam-trim-hi", type=float, default=0.95)
    ap.add_argument("--sam-iou-lo", type=float, default=0.60)
    ap.add_argument("--sam-min-area", type=int, default=16)
    # Tissue filter (NEW) -- defaults pulled from e7_config.TISSUE_FILTER
    ap.add_argument("--no-tissue-filter", action="store_true",
                    help="disable the off-tissue spatial guard")
    ap.add_argument("--tissue-min-ratio", type=float, default=None,
                    help="override min fraction of a mask that must lie on tissue")
    ap.add_argument("--tissue-debug", action="store_true",
                    help="write a kept(class colour)-vs-dropped(red) overlay per source")
    args = ap.parse_args()

    sdir = Path(args.scripts_dir)
    sys.path.insert(0, str(sdir))          # so the stages' `import e7_config` works
    import e7_config as C

    def find_stage(pattern):
        cands = sorted(sdir.glob(pattern))
        if not cands:
            raise SystemExit(f"no stage file matching '{pattern}' in {sdir}")
        return cands[0]   # prefix-agnostic: matches 7_/07_/etc.

    modules = {
        "s07": load_module(find_stage("*infer_tiles_main_e7.py"), "s07_infer"),
        "s08": load_module(find_stage("*stitch_main_e7.py"), "s08_stitch"),
        "s09": load_module(find_stage("*fuse_main_e7.py"), "s09_fuse"),
        "s10": load_module(find_stage("*filter_main_e7.py"), "s10_filter"),
    }
    import cv2

    tile = args.tile_size or C.TILE_SIZE
    overlap = args.overlap_ratio if args.overlap_ratio is not None else C.OVERLAP_RATIO
    stride = int(round(tile * (1.0 - overlap)))
    weights = args.weights or str(C.OUT_MAIN / "checkpoints" / "model_best.pth")
    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "end2end" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- tissue filter config (e7_config default, then CLI overrides) ----
    tissue_cfg = dict(getattr(C, "TISSUE_FILTER",
                              dict(enable=True, min_tissue_ratio=0.15,
                                   otsu_blur=5, downsample=4)))
    if args.no_tissue_filter:
        tissue_cfg["enable"] = False
    if args.tissue_min_ratio is not None:
        tissue_cfg["min_tissue_ratio"] = args.tissue_min_ratio

    # ---- resolve the list of (source_key, image_path) to process ----
    work = []
    if args.image:
        work.append((args.source or Path(args.image).stem, args.image))
    else:
        import pandas as pd
        csv = args.split_csv or str(C.PREP_DIR / "02_e7_source_split.csv")
        df = pd.read_csv(csv)
        df = df[(df["split"] == args.split) & (df.get("has_image", True) == True)]
        if "parse_ok" in df.columns:
            df = df[df["parse_ok"] == True]
        if "has_any_main" in df.columns:
            df = df[df["has_any_main"] == True]
        src2path = {r["source_key"]: r["image_path"] for _, r in df.iterrows()}
        if args.all:
            keys = sorted(src2path)
        elif args.source:
            keys = [k.strip() for k in args.source.split(",") if k.strip()]
        else:
            raise SystemExit("give --image, --source KEY[,KEY...], or --all")
        for k in keys:
            if k not in src2path:
                print(f"[WARN] {k} not in split '{args.split}', skipping")
                continue
            work.append((k, src2path[k]))
    if not work:
        raise SystemExit("no sources to process")

    sampler = GPUSampler(); sampler.start()

    # ---- model load (timed once) ----
    t0 = time.time()
    cfg = modules["s07"].build_inference_cfg(weights)
    predictor = modules["s07"].DefaultPredictor(cfg)
    sam_ctx = None
    if args.sam:
        s12 = load_module(find_stage("*refine_sam_e7.py"), "s12_sam")
        device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
        args.sam_best = args.sam_best or str(s12.DEFAULT_BEST)
        args.sam_checkpoint = args.sam_checkpoint or str(s12.DEFAULT_PRETRAINED)
        model, adapter = build_sam(s12, args, device)
        sam_ctx = (s12, model, adapter, args, device)
    t_load = time.time() - t0

    print("=" * 78)
    print(f"E7 end-to-end | sources={len(work)} split={args.split} tile={tile} "
          f"stride={stride} SAM={'on' if args.sam else 'off'}"
          f"{(' mode=' + args.sam_mode) if args.sam else ''} "
          f"tissue_filter={'on' if tissue_cfg.get('enable', True) else 'off'}")
    print(f"  weights: {weights}")
    print("=" * 78)
    print(f"  model loaded in {t_load:.1f} s")

    # ---- warm-up (excluded from timing) ----
    if len(work) > 1 and not args.no_warmup:
        wsrc, wpath = work[0]
        wimg = cv2.imread(str(wpath))
        if wimg is not None:
            run_one(modules, predictor, wimg, wsrc, tile, stride, args.score_thr,
                    sam_ctx, tissue_cfg)
            print("  warm-up done (first source, untimed)")

    # ---- timed end-to-end over all sources ----
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    per_source, total_tiles = [], 0
    t1 = time.time()
    n_work = len(work)
    for wi, (src, path) in enumerate(work, 1):
        print(f"\n[{wi}/{n_work}] processing {src} ...", flush=True)
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [skip] {src}: cannot read {path}", flush=True); continue
        dbg_path = (out_dir / f"{src}_tissuedbg.png") if args.tissue_debug else None
        ts = time.time()
        final, H, W, n_tiles = run_one(modules, predictor, image, src, tile,
                                       stride, args.score_thr, sam_ctx,
                                       tissue_cfg, dbg_path)
        dt = time.time() - ts
        out = OrderedDict(source_key=src, height=H, width=W,
                          num_instances=len(final), instances=final)
        (out_dir / f"{src}.json").write_text(json.dumps(out))
        if args.overlay:
            render_overlay(image, final, out_dir / f"{src}_overlay.png")
        per_source.append((src, dt, n_tiles, len(final)))
        total_tiles += n_tiles
        print(f"  {src}: {n_tiles} tiles -> {len(final)} instances  ({dt:.1f} s)")
    t_e2e = time.time() - t1
    sampler.stop(); time.sleep(0.3)

    n = len(per_source)
    peak_torch = (torch.cuda.max_memory_allocated() / 1e9) if (HAS_TORCH and torch.cuda.is_available()) else 0.0
    print("-" * 60)
    print(f"TIMING (M2F end-to-end{(' + SAM ' + args.sam_mode) if args.sam else ''})")
    print(f"  model load        : {t_load:6.1f} s")
    print(f"  sources           : {n}")
    print(f"  tiles (total)     : {total_tiles}")
    if n > 0:
        print(f"  end-to-end wall   : {t_e2e:6.1f} s  "
              f"({t_e2e / n:.2f} s/source, "
              f"{1000 * t_e2e / max(1, total_tiles):.1f} ms/tile-incl-postproc)")
    print(f"  peak GPU memory   : {sampler.peak_mb / 1024:6.2f} GB  (nvidia-smi, this PID)"
          + (f"  | torch tensor {peak_torch:.2f} GB" if peak_torch else ""))
    print("-" * 60)
    print(f"  wrote -> {out_dir}")
    print("  NOTE: end-to-end scope (tile->infer->stitch->fuse->filter[->SAM][->tissue]) matches")
    print("        MRCNN's bundled InferenceTool call -> per-source time is comparable.")


if __name__ == "__main__":
    main()
