#!/usr/bin/env python
"""
tissue_filter.py  --  spatial post-processing guard for the E7 pipeline.

Drops final instances whose mask lies mostly OFF the tissue (i.e. floating on the
white WSI background). These appear when a tile at the biopsy edge is mostly
background: M2F still fires (score > floor) and the stitched mask spills over the
white area as a flat, non-biological blob.

This is the cheap stand-in for the cortex-aware tiling (MAIN_MIN_TILE_CORTEX),
which is disabled until cortex masks exist. It needs no cortex branch: tissue vs
white background is a trivial Otsu split on the clean source image.

Runs LAST in the chain (after orphan_filter), in global WSI coordinates, so each
instance mask is checked against one tissue mask. Imports nothing from the stages.

Public API
----------
    build_tissue_mask(image_bgr, blur=5, downsample=4) -> (mask_bool_small, ds)
    filter_outside_tissue(image_bgr, instances, cfg, debug_overlay_path=None)
        -> kept_instances        # also annotates it["tissue_ratio"] on every input
    render_filter_debug(image_bgr, kept, dropped, out_path, alpha=0.45)
"""
from pathlib import Path

import numpy as np


# --- same palette as e7_run.render_overlay, kept local so this module stands alone
_PALETTE = {
    "tubule_sain": (230, 126, 60), "tubule_atrophique": (123, 90, 200),
    "vaisseau": (40, 160, 70), "nsg": (40, 200, 210), "intima": (235, 30, 110),
    "nsg_partiel": (240, 200, 30), "media": (70, 130, 235), "veine": (150, 150, 150),
    "nsg_complet": (180, 40, 200), "pac": (20, 160, 140),
}

_DEFAULT_CFG = dict(enable=True, min_tissue_ratio=0.15, otsu_blur=5, downsample=4)


def _decode_mask(it):
    """Decode an instance RLE (counts as str) to a full-resolution bool mask."""
    from pycocotools import mask as mask_util
    rle = {"size": it["mask_rle"]["size"],
           "counts": it["mask_rle"]["counts"].encode("ascii")}
    return mask_util.decode(rle).astype(bool)


def build_tissue_mask(image_bgr, blur=5, downsample=4):
    """Otsu tissue mask. True = tissue (darker/coloured), False = white background.

    Computed at 1/downsample resolution to stay cheap on whole-slide images
    (a native 30k x 17k slide is ~0.5 GB per channel). A light morphological close
    fills pin holes inside tissue so interior objects are not wrongly flagged.

    Returns (tissue_small_bool, downsample) -- the mask is at reduced resolution;
    callers map instance bboxes into it with // downsample.
    """
    import cv2
    h, w = image_bgr.shape[:2]
    ds = max(1, int(downsample))
    if ds > 1:
        small = cv2.resize(image_bgr, (max(1, w // ds), max(1, h // ds)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = int(blur)
    if k >= 3:
        k = k if k % 2 == 1 else k + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    # tissue is darker than the white background -> inverse-binary Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return th.astype(bool), ds


def filter_outside_tissue(image_bgr, instances, cfg=None, debug_overlay_path=None):
    """Keep only instances whose mask is >= min_tissue_ratio on tissue.

    Every input instance is annotated in place with it["tissue_ratio"] (float in
    [0, 1]) so the decision is auditable from the JSON. Returns the kept list; if
    debug_overlay_path is given, writes an overlay (kept = class colour,
    dropped = solid red) for visual validation before hand-off.
    """
    cfg = {**_DEFAULT_CFG, **(cfg or {})}
    if not cfg["enable"] or not instances:
        return instances

    min_ratio = float(cfg["min_tissue_ratio"])
    tissue, ds = build_tissue_mask(image_bgr, blur=cfg["otsu_blur"],
                                   downsample=cfg["downsample"])
    th, tw = tissue.shape
    import cv2

    kept, dropped = [], []
    for it in instances:
        m = _decode_mask(it)
        area = int(m.sum())
        if area == 0:
            it["tissue_ratio"] = 0.0
            dropped.append(it)
            continue
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        m_crop = m[y0:y1, x0:x1]
        # map the bbox into downsampled tissue coords (floor start, ceil end)
        sy0, sx0 = y0 // ds, x0 // ds
        sy1 = min(th, (y1 + ds - 1) // ds)
        sx1 = min(tw, (x1 + ds - 1) // ds)
        sy1 = max(sy1, sy0 + 1)
        sx1 = max(sx1, sx0 + 1)
        t_small = tissue[sy0:sy1, sx0:sx1].astype(np.uint8)
        # upsample tissue patch to the instance crop size (NEAREST: keep it binary)
        t_up = cv2.resize(t_small, (m_crop.shape[1], m_crop.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        on_tissue = int((m_crop & t_up).sum())
        ratio = on_tissue / area
        it["tissue_ratio"] = round(float(ratio), 4)
        (kept if ratio >= min_ratio else dropped).append(it)

    if debug_overlay_path is not None:
        render_filter_debug(image_bgr, kept, dropped, debug_overlay_path)
    return kept


def render_filter_debug(image_bgr, kept, dropped, out_path, alpha=0.45):
    """Overlay: kept instances in their class colour, dropped ones in solid red.
    Lets you eyeball exactly what the tissue filter removed before trusting it."""
    import cv2
    over = image_bgr.copy()
    for it in kept:
        m = _decode_mask(it)
        if m.sum() == 0:
            continue
        r, g, b = _PALETTE.get(it["class_name"], (255, 255, 0))
        over[m] = ((1 - alpha) * over[m] + alpha * np.array([b, g, r])).astype(np.uint8)
    for it in dropped:
        m = _decode_mask(it)
        if m.sum() == 0:
            continue
        over[m] = ((1 - alpha) * over[m] + alpha * np.array([0, 0, 255])).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), over)
    return out_path