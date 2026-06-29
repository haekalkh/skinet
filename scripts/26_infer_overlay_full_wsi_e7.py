"""
E7 Stage 26 - Full-WSI MAIN inference inside the cortex, with a combined overlay.

Runs the trained MAIN Mask2Former model directly on a FULL WSI (the same images
the cortex branch uses, e.g. data/raw/cortex/<src>.jpg), tiling natively at the
training TILE_SIZE/stride but ONLY over tiles that overlap the cortex region
(from the already-computed cortex masks). Produces a single combined picture:

    cortex region  -> outline (+ light fill)
    main structures (tubules / glomeruli / vessels) -> per-class colour fill,
                      clipped to the cortex.

This is the clinical view "full slide, cortex + main together". GPU stage
(reuses the Stage-7 model setup). Run via qsub.

  qsub -q gpu -v SRC=ML0029C2,SPLIT=test jobs/submit_e7_full_wsi.sh
  # or directly on a GPU node:
  python 26_infer_overlay_full_wsi_e7.py --source ML0029C2 --split test

Output: OUT_MAIN/full_wsi/{split}/{src}.png  [+ {src}_main_wsi.json with --save-json]
"""

import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util

import e7_config as C


# ----------------------------------------------------------------------------
# Palette (RGB) - mirrors the existing overlay style
# ----------------------------------------------------------------------------
MAIN_PALETTE = {
    "tubule_sain": (237, 125, 49), "tubule_atrophique": (91, 60, 196),
    "vaisseau": (76, 175, 80), "nsg": (0, 200, 220), "intima": (233, 30, 99),
    "nsg_partiel": (235, 205, 30), "media": (66, 135, 245), "veine": (120, 120, 120),
    "nsg_complet": (156, 39, 176), "pac": (0, 150, 136),
}
CORTEX_COLOR = (40, 40, 40)        # cortex outline


def draw_legend_panel(out, entries, title, pos="tl", cortex_color=None):
    """Compact semi-transparent legend panel. entries: list of (bgr|None, text);
    bgr None + cortex_color -> draw an outline swatch. pos in tl/tr/bl/br/none."""
    if pos == "none":
        return
    H, W = out.shape[:2]
    pad, sw, gap, lh = 10, 16, 7, 22
    texts = [title] + [t for _, t in entries]
    fs = [0.6] + [0.5] * len(entries)
    widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, f, 1)[0][0]
              for t, f in zip(texts, fs)]
    box_w = sw + gap + max(widths) + 2 * pad
    box_h = pad * 2 + 24 + lh * len(entries)
    x0 = (W - box_w - 12) if "r" in pos else 12
    y0 = (H - box_h - 12) if "b" in pos else 12
    panel = out.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + box_w, y0 + box_h), (35, 35, 35), -1)
    cv2.addWeighted(panel, 0.55, out, 0.45, 0, out)
    cv2.rectangle(out, (x0, y0), (x0 + box_w, y0 + box_h), (210, 210, 210), 1)
    yy = y0 + pad + 16
    cv2.putText(out, title, (x0 + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    yy += lh + 2
    for bgr, txt in entries:
        if bgr is None and cortex_color is not None:
            cv2.rectangle(out, (x0 + pad, yy - 12), (x0 + pad + sw, yy + 2), cortex_color, 2)
        else:
            cv2.rectangle(out, (x0 + pad, yy - 12), (x0 + pad + sw, yy + 2), bgr, -1)
        cv2.putText(out, txt, (x0 + pad + sw + gap, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (245, 245, 245), 1, cv2.LINE_AA)
        yy += lh


def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)


def resolve_wsi(source, explicit):
    """Find the full WSI image: explicit path, else data/raw/cortex/<src>.<ext>
    trying all known extensions (.jpg, .jp2, .tif, ...)."""
    if explicit:
        return Path(explicit)
    for ext in C.IMAGE_EXTS:
        cand = C.RAW_CORTEX_DIR / f"{source}{ext}"
        if cand.exists():
            return cand
    return C.RAW_CORTEX_DIR / f"{source}.jpg"  # default (will error if missing)


def read_wsi(path):
    """Robust full-res read. cv2 first; PIL fallback for JPEG2000 (.jp2) when
    OpenCV lacks OpenJPEG. Returns BGR uint8 or None."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        arr = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # RGB->BGR
        return np.ascontiguousarray(arr)
    except Exception as e:
        print(f"[read_wsi] PIL fallback failed: {e}")
        return None


def rle_in(rle):
    c = rle["counts"]
    return {"size": rle["size"], "counts": c.encode("ascii") if isinstance(c, str) else c}


# ---- tiling helpers (same logic as Stage 7) --------------------------------
def compute_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def make_tile_image(image, x0, y0, tile_size):
    h, w = image.shape[:2]
    x1, y1 = min(x0 + tile_size, w), min(y0 + tile_size, h)
    tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    tile[: y1 - y0, : x1 - x0] = image[y0:y1, x0:x1]
    return tile


# ---- model (reuses the Stage-7 config; detectron2 imported lazily) ----------
def build_predictor(weights):
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(C.BASE_CONFIG))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(C.MAIN_CLASSES)
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.INPUT.MIN_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return DefaultPredictor(cfg)


def predict_tile(predictor, tile):
    """-> (masks [N,h,w] bool, scores [N], class_idx [N])."""
    out = predictor(tile)["instances"].to("cpu")
    if len(out) == 0:
        return np.zeros((0,) + tile.shape[:2], bool), np.zeros((0,)), np.zeros((0,), int)
    return out.pred_masks.numpy().astype(bool), out.scores.numpy(), out.pred_classes.numpy()


# ---- cortex gate (display resolution) --------------------------------------
def load_cortex_gate(cortex_json, dh, dw, gate_classes=("cortex",)):
    """Return (gate_disp[dh,dw] uint8, regions{class:disp_mask}). gate = cortex
    region used to restrict MAIN; falls back to all cortex-branch regions."""
    d = json.load(open(cortex_json))
    insts = d.get("instances", [])
    if not insts:
        raise SystemExit(f"cortex json has no 'instances': {cortex_json}\n"
                         f"keys present: {list(d.keys())}")
    gate = np.zeros((dh, dw), np.uint8)
    allm = np.zeros((dh, dw), np.uint8)
    regions = defaultdict(lambda: np.zeros((dh, dw), np.uint8))
    for it in insts:
        rle = it.get("mask_rle") or it.get("rle")
        cn = it.get("class_name", "cortex")
        if not rle:
            continue
        m = mask_util.decode(rle_in(rle))
        ms = cv2.resize(m, (dw, dh), interpolation=cv2.INTER_NEAREST)
        allm[ms > 0] = 1
        regions[cn][ms > 0] = 1
        if cn in gate_classes:
            gate[ms > 0] = 1
    if gate.sum() == 0:
        print("[gate] no 'cortex' class found; gating on ALL cortex-branch regions")
        gate = allm
    return gate, dict(regions)


def crop_full(image, cx, cy, size):
    """size x size full-res crop centered at (cx,cy), white-padded at edges."""
    h, w = image.shape[:2]
    half = size // 2
    x0, y0 = cx - half, cy - half
    out = np.full((size, size, 3), 255, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return out


def select_inset_centers(mask_disp, k, min_area=20):
    """Top-k connected components by area in a display-res binary mask ->
    list of (cx, cy) display-coord centroids."""
    if mask_disp.sum() == 0:
        return []
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask_disp.astype(np.uint8), 8)
    comps = [(int(stats[i, cv2.CC_STAT_AREA]), int(cents[i][0]), int(cents[i][1]))
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    comps.sort(reverse=True)
    return [(cx, cy) for _, cx, cy in comps[:k]]


def render_patch(crop_bgr, masks, scores, classes, score_thr, alpha=0.45):
    """High-mag inset: overlay per-instance masks (colored) on a full-res crop."""
    layer = crop_bgr.copy()
    for i in range(len(scores)):
        if scores[i] < score_thr or masks[i].sum() == 0:
            continue
        cn = C.MAIN_CLASSES[int(classes[i])]
        rgb = MAIN_PALETTE.get(cn, (255, 255, 255))
        layer[masks[i]] = (rgb[2], rgb[1], rgb[0])
    return cv2.addWeighted(layer, alpha, crop_bgr, 1 - alpha, 0)


def compute_insets(args, image, canvas, ratio, dh, dw, predictor):
    """Auto-pick up to args.insets components of args.inset_class and render a
    full-res high-mag patch for each (re-segmented for crisp masks)."""
    insets = []
    if args.insets <= 0:
        return insets
    isize = args.inset_size or args.tile_size
    cls_canvas = canvas.get(args.inset_class, np.zeros((dh, dw), np.uint8))
    for (cxd, cyd) in select_inset_centers(cls_canvas, args.insets):
        cx, cy = int(cxd / ratio), int(cyd / ratio)
        crop = crop_full(image, cx, cy, isize)
        if isize != args.tile_size:
            cm, cs, cc = predict_tile(predictor, cv2.resize(crop, (args.tile_size, args.tile_size)))
            if len(cm):
                cm = np.stack([cv2.resize(m.astype(np.uint8), (isize, isize),
                               interpolation=cv2.INTER_NEAREST).astype(bool) for m in cm])
        else:
            cm, cs, cc = predict_tile(predictor, crop)
        patch = render_patch(crop, cm, cs, cc, args.score_thr)
        half = int((isize * ratio) / 2)
        insets.append({"img": patch, "box": (cxd - half, cyd - half, cxd + half, cyd + half)})
    print(f"  insets: {len(insets)} high-mag patch(es) centered on '{args.inset_class}'")
    return insets


def compose_figure(overview, insets):
    """overview (BGR, with numbered boxes already drawn) + a right column of
    numbered high-mag insets."""
    if not insets:
        return overview
    oh, ow = overview.shape[:2]
    col_w = max(260, ow // 4)
    gap, labelh = 10, 22
    tiles = []
    for k, ins in enumerate(insets, 1):
        pim = ins["img"]
        ph, pw = pim.shape[:2]
        rim = cv2.resize(pim, (col_w, max(1, int(ph * col_w / pw))),
                         interpolation=cv2.INTER_AREA)
        cv2.rectangle(rim, (0, 0), (col_w - 1, rim.shape[0] - 1), (0, 200, 255), 3)
        band = np.full((labelh, col_w, 3), 255, np.uint8)
        cv2.putText(band, f"#{k}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([band, rim]))
    parts = []
    for i, t in enumerate(tiles):
        parts.append(t)
        if i < len(tiles) - 1:
            parts.append(np.full((gap, col_w, 3), 255, np.uint8))
    col = np.vstack(parts)
    ch = col.shape[0]
    fig = np.full((max(oh, ch), ow + gap + col_w, 3), 255, np.uint8)
    fig[:oh, :ow] = overview
    fig[:ch, ow + gap:ow + gap + col_w] = col
    return fig


def render_and_save(disp, canvas, gate_disp, counts, draw_classes, whole_wsi,
                    alpha, legend_pos, source, out_png, insets=None):
    layer = disp.copy()
    for c in draw_classes:
        rgb = MAIN_PALETTE.get(c, (255, 255, 255))
        layer[canvas[c] > 0] = (rgb[2], rgb[1], rgb[0])
    out = cv2.addWeighted(layer, alpha, disp, 1 - alpha, 0)
    if not whole_wsi:
        cnts, _ = cv2.findContours(gate_disp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, CORTEX_COLOR, 2)
    entries = []
    for c in draw_classes:
        rgb = MAIN_PALETTE.get(c, (255, 255, 255))
        entries.append(((rgb[2], rgb[1], rgb[0]), f"{c} ({counts.get(c, 0)})"))
    if not whole_wsi:
        entries.append((None, "cortex (outline)"))
    title = f"{source}  ({'whole WSI' if whole_wsi else 'main inside cortex'})"
    draw_legend_panel(out, entries, title, pos=legend_pos, cortex_color=CORTEX_COLOR)
    if insets:
        for k, ins in enumerate(insets, 1):
            x0, y0, x1, y1 = ins["box"]
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(out, f"#{k}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2, cv2.LINE_AA)
        out = compose_figure(out, insets)
    cv2.imwrite(str(out_png), out)
    return out


def save_cache(path, disp, gate_disp, canvas, counts, whole_wsi, insets=None):
    classes = list(canvas.keys())
    insets = insets or []
    data = dict(disp=disp, gate=gate_disp, classes=np.array(classes),
                stack=np.stack([canvas[c] for c in classes]),
                counts=np.array([counts.get(c, 0) for c in classes]),
                whole_wsi=np.array(1 if whole_wsi else 0),
                n_insets=np.array(len(insets)))
    if insets:
        data["inset_imgs"] = np.stack([i["img"] for i in insets])
        data["inset_boxes"] = np.array([i["box"] for i in insets])
    np.savez_compressed(path, **data)


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    classes = [str(c) for c in z["classes"]]
    canvas = {c: z["stack"][i] for i, c in enumerate(classes)}
    counts = {c: int(z["counts"][i]) for i, c in enumerate(classes)}
    insets = []
    if "n_insets" in z and int(z["n_insets"]) > 0:
        imgs, boxes = z["inset_imgs"], z["inset_boxes"]
        insets = [{"img": imgs[i], "box": tuple(int(v) for v in boxes[i])}
                  for i in range(len(imgs))]
    return z["disp"], z["gate"], canvas, counts, bool(int(z["whole_wsi"])), insets


def infer_full(args):
    """Run MAIN inference over the WSI (gated to cortex unless --whole-wsi).
    Returns disp, canvas, gate_disp, counts, saved, H, W, n_used."""
    wsi_path = resolve_wsi(args.source, args.wsi_image)
    cortex_json = (Path(args.cortex_json) if args.cortex_json
                   else (C.OUT_CORTEX / "cortex_masks" / args.split / f"{args.source}.json"))
    if not wsi_path.exists():
        raise SystemExit(f"WSI image not found: {wsi_path}")
    if not args.whole_wsi and not cortex_json.exists():
        raise SystemExit(f"cortex json not found: {cortex_json}\n"
                         f"(run the cortex branch first, or use --whole-wsi)")
    print("=" * 78)
    print(f"E7 Stage 26 - full-WSI MAIN | source={args.source} split={args.split}")
    print(f"  WSI   : {wsi_path}")
    print(f"  cortex: {'(disabled, whole WSI)' if args.whole_wsi else cortex_json}")
    print("=" * 78)

    print("  reading WSI (may be large; .jp2 decode can be slow)...")
    image = read_wsi(wsi_path)
    if image is None:
        raise SystemExit(
            f"cannot decode WSI {wsi_path}.\n"
            f"If it is .jp2, your OpenCV/PIL may lack OpenJPEG. Options: install a\n"
            f"jp2-capable reader, or convert once to .tif/.jpg and pass --wsi-image.")
    H, W = image.shape[:2]
    ratio = args.long_side / float(max(H, W)) if max(H, W) > args.long_side else 1.0
    dh, dw = max(1, round(H * ratio)), max(1, round(W * ratio))
    print(f"  WSI {W}x{H} -> display {dw}x{dh} (ratio {ratio:.4f})")

    if args.whole_wsi:
        gate_disp = np.ones((dh, dw), np.uint8)
    else:
        gate_disp, _ = load_cortex_gate(cortex_json, dh, dw)
        print(f"  cortex covers {100.0 * gate_disp.mean():.1f}% of the WSI")

    stride = int(round(args.tile_size * (1.0 - args.overlap_ratio)))
    t = time.time()
    predictor = build_predictor(Path(args.weights))
    load_s = time.time() - t
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    canvas = {c: np.zeros((dh, dw), np.uint8) for c in C.MAIN_CLASSES}
    counts = defaultdict(int)
    n_tiles = n_used = 0
    infer_s = 0.0
    saved = []
    ys = compute_starts(H, args.tile_size, stride)
    xs = compute_starts(W, args.tile_size, stride)
    for y0 in ys:
        for x0 in xs:
            n_tiles += 1
            if not args.whole_wsi:
                xd0, yd0 = int(x0 * ratio), int(y0 * ratio)
                xd1, yd1 = int((x0 + args.tile_size) * ratio), int((y0 + args.tile_size) * ratio)
                if gate_disp[yd0:yd1, xd0:xd1].sum() == 0:
                    continue
            n_used += 1
            tile = make_tile_image(image, x0, y0, args.tile_size)
            _ti = time.time()
            masks, scores, classes = predict_tile(predictor, tile)
            infer_s += time.time() - _ti
            for i in range(len(scores)):
                if scores[i] < args.score_thr or masks[i].sum() == 0:
                    continue
                cn = C.MAIN_CLASSES[int(classes[i])]
                tdw = max(1, round(args.tile_size * ratio))
                md = cv2.resize(masks[i].astype(np.uint8), (tdw, tdw),
                                interpolation=cv2.INTER_NEAREST)
                xd0, yd0 = int(x0 * ratio), int(y0 * ratio)
                yd1, xd1 = min(yd0 + tdw, dh), min(xd0 + tdw, dw)
                sub = canvas[cn][yd0:yd1, xd0:xd1]
                sub[md[: yd1 - yd0, : xd1 - xd0] > 0] = 1
                counts[cn] += 1
                if args.save_json:
                    saved.append({"class_name": cn, "score": float(scores[i]),
                                  "tile_x": int(x0), "tile_y": int(y0)})
        print(f"  ...tiles {n_used} used / {n_tiles} scanned", end="\r")
    print(f"\n  tiles used (in cortex): {n_used}/{n_tiles}")

    gpu_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    ms_tile = (infer_s / n_used * 1000) if n_used else 0.0
    print("-" * 60)
    print("TIMING")
    print(f"  model load        : {load_s:6.1f} s")
    print(f"  tiles inferred    : {n_used}")
    print(f"  pure inference    : {infer_s:6.1f} s  ({ms_tile:.1f} ms/tile, "
          f"{n_used / infer_s if infer_s else 0:.1f} tiles/s)")
    if gpu_gb is not None:
        print(f"  peak GPU memory   : {gpu_gb:5.2f} GB")
    print("-" * 60)

    if not args.whole_wsi:
        for c in canvas:
            canvas[c][gate_disp == 0] = 0

    insets = compute_insets(args, image, canvas, ratio, dh, dw, predictor)

    disp = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_AREA)
    return disp, canvas, gate_disp, dict(counts), saved, H, W, n_used, insets


def main():
    ap = argparse.ArgumentParser("E7 full-WSI MAIN inference + combined overlay")
    ap.add_argument("--source", required=True, help="source_key, e.g. ML0029C2")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--wsi-image", default=None,
                    help="full WSI path (default: data/raw/cortex/<src>.<ext>)")
    ap.add_argument("--cortex-json", default=None,
                    help="cortex mask json (default: cortex_masks/<split>/<src>.json)")
    ap.add_argument("--weights", default=str(C.OUT_MAIN / "checkpoints" / "model_best.pth"))
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--overlap-ratio", type=float, default=C.OVERLAP_RATIO)
    ap.add_argument("--score-thr", type=float, default=0.50)
    ap.add_argument("--long-side", type=int, default=2048, help="display resolution")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--whole-wsi", action="store_true",
                    help="do NOT restrict to cortex (tile the whole WSI)")
    ap.add_argument("--classes", default=None,
                    help="comma-separated MAIN classes to draw (default: all)")
    ap.add_argument("--legend-pos", default="tl", choices=["tl", "tr", "bl", "br", "none"],
                    help="legend position (tl/tr/bl/br) or none")
    ap.add_argument("--render-only", action="store_true",
                    help="skip GPU inference; re-render from the cached result "
                         "(fast, CPU-only; change --legend-pos/--classes/--alpha)")
    ap.add_argument("--insets", type=int, default=0,
                    help="add N high-mag inset patches (auto-picked, full-res)")
    ap.add_argument("--inset-class", default="nsg",
                    help="class to center insets on (default: nsg)")
    ap.add_argument("--inset-size", type=int, default=0,
                    help="full-res inset window in px (0 = tile size)")
    ap.add_argument("--insets-only", action="store_true",
                    help="reuse cached overview; infer ONLY the inset crops (fast)")
    ap.add_argument("--no-cache", action="store_true", help="do not write the render cache")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-json", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (C.OUT_MAIN / "full_wsi" / args.split)
    ensure_dir(out_dir)
    draw_classes = args.classes.split(",") if args.classes else list(C.MAIN_CLASSES)
    bad = [c for c in draw_classes if c not in C.MAIN_CLASSES]
    if bad:
        raise SystemExit(f"unknown --classes: {bad}\nvalid: {C.MAIN_CLASSES}")
    cache_path = out_dir / f"{args.source}_cache.npz"
    t_wall0 = time.time()

    if args.render_only:
        if not cache_path.exists():
            raise SystemExit(f"no cache for --render-only: {cache_path}\n"
                             f"run once without --render-only first.")
        print(f"[render-only] loading cache {cache_path}")
        disp, gate_disp, canvas, counts, whole_wsi, insets = load_cache(cache_path)
    elif args.insets_only:
        if not cache_path.exists():
            raise SystemExit(f"no cache for --insets-only: {cache_path}\n"
                             f"run a full pass first.")
        if args.insets <= 0:
            raise SystemExit("--insets-only needs --insets N (N>0)")
        print(f"[insets-only] reusing cached overview; inferring only {args.insets} crop(s)")
        disp, gate_disp, canvas, counts, whole_wsi, _old = load_cache(cache_path)
        wsi_path = resolve_wsi(args.source, args.wsi_image)
        image = read_wsi(wsi_path)
        if image is None:
            raise SystemExit(f"cannot read WSI {wsi_path}")
        H, W = image.shape[:2]
        dh, dw = disp.shape[:2]
        ratio = dw / float(W)
        predictor = build_predictor(Path(args.weights))
        insets = compute_insets(args, image, canvas, ratio, dh, dw, predictor)
        if not args.no_cache:
            save_cache(cache_path, disp, gate_disp, canvas, counts, whole_wsi, insets)
    else:
        disp, canvas, gate_disp, counts, saved, H, W, n_used, insets = infer_full(args)
        whole_wsi = args.whole_wsi
        if not args.no_cache:
            save_cache(cache_path, disp, gate_disp, canvas, counts, whole_wsi, insets)
            print(f"[cache] saved {cache_path}  (re-render fast: --render-only)")
        if args.save_json:
            jp = out_dir / f"{args.source}_main_wsi.json"
            json.dump({"source_key": args.source, "wsi_hw": [H, W],
                       "tiles_used": n_used, "instances": saved}, open(jp, "w"))
            print(f"Saved json    -> {jp}")

    out_png = out_dir / f"{args.source}.png"
    render_and_save(disp, canvas, gate_disp, counts, draw_classes, whole_wsi,
                    args.alpha, args.legend_pos, args.source, out_png, insets=insets)
    print(f"\nSaved overlay -> {out_png}")
    print("  per-class instances: "
          + ", ".join(f"{c}:{counts.get(c, 0)}" for c in draw_classes if counts.get(c, 0)))
    print(f"\nTOTAL wall time   : {time.time() - t_wall0:6.1f} s "
          f"({(time.time() - t_wall0) / 60:.1f} min)")


if __name__ == "__main__":
    main()