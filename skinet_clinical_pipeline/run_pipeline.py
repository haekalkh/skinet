#!/usr/bin/env python
"""
SKINET CLINICAL PIPELINE (Cara B) - orchestrator modular.

Reproduksi sistem klinis MRCNN dengan Mask2Former, memakai cortex-gating di
POST-PROCESSING (bukan input cropping). Setia pada fungsi klinis (analisis
dibatasi ke cortex, per Marechal et al. 2022) tapi kompatibel dengan M2F.

URUTAN STEP (tiap script bisa dijalankan sendiri):
  step 1  01_gen_cortex_mask.py       WSI -> cortex mask PNG            [GPU]
  step 2  (08_stitch, sudah ada)      main M2F normal -> stitched      [GPU]
  step 3  03_apply_threshold.py       filter per-class score threshold [CPU]
  step 4  04_cortex_gating_postproc.py 7-stage postproc + cortex gating[CPU]
  step 5  05_compute_stats.py         count/area/density vs MRCNN      [CPU]
  step 6  06_export_results.py        LabelMe+ASAP+overlay per source  [CPU]

CATATAN: main M2F jalan pada WSI NORMAL (background putih, sesuai training),
BUKAN cleaned/cortex-cropped image (yg menyebabkan domain shift + prediksi
kacau). Cortex-gating diterapkan sebagai border_filter di step 4.

Contoh:
  # satu source, step 3-6 (cortex+main sudah ada):
  python run_pipeline.py --source ML9360N14 --split test --from-step 3 --skip-cortex
  # semua 44 WSI:
  python run_pipeline.py --all --split test --from-step 3 --skip-cortex
  # full dari awal (butuh GPU step 1):
  python run_pipeline.py --all --split test

Sesuaikan PATH di bawah dengan struktur cluster-mu.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# ============================================================================
# PATH - SESUAIKAN dengan struktur cluster
# ============================================================================
SKINET   = Path("/work/lead/mk657917/SKINET")
SCRIPTS  = Path(__file__).resolve().parent / "scripts"
PY       = sys.executable

CORTEX_CKPT = SKINET / "outputs/e7_clinical_pipeline/cortex_m2f_swin_b/checkpoints/model_best.pth"
CORTEX_MASK = SKINET / "outputs/e7_clinical_pipeline/cortex_m2f_swin_b/cortex_masks_main"
WSI_DIR     = SKINET / "outputs/mrcnn_cmp/work/results/test_main_2026-06-16_20-16-13"
AUG         = SKINET / "outputs/main/mask2former_skinet_aug"
THR_JSON    = AUG / "postproc/val/postfusion_threshold.json"
MRCNN_DIR   = WSI_DIR   # MRCNN _stats.json utk banding
# ============================================================================


def run(cmd, label):
    print(f"\n{'='*64}\n[{label}]\n{'='*64}", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        print(f"[FAIL] {label} (rc={r.returncode})")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="satu source; kosong -> semua")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--from-step", type=int, default=1)
    ap.add_argument("--to-step", type=int, default=6)
    ap.add_argument("--skip-cortex", action="store_true",
                    help="lewati step 1 (cortex mask sudah ada)")
    ap.add_argument("--mpp", type=float, default=0.0,
                    help="microns-per-pixel utk density mm^2 (0=per-px)")
    args = ap.parse_args()

    split = args.split
    src   = args.source
    lo, hi = args.from_step, args.to_step

    post        = AUG / "postproc" / split
    stitched    = post / "08_stitched"
    thresholded = post / "08b_thresholded"
    gated       = post / "10b_cortex_gated"
    cortex_dir  = CORTEX_MASK / split
    export_dir  = AUG / "export" / split

    def with_src(cmd):
        if src and not args.all:
            cmd += ["--sources", src]
        return cmd

    # --- step 1: cortex mask (GPU) ---
    if lo <= 1 <= hi and not args.skip_cortex:
        cmd = [PY, SCRIPTS/"01_gen_cortex_mask.py",
               "--weights", CORTEX_CKPT, "--wsi-dir", WSI_DIR,
               "--out-dir", cortex_dir, "--score-thr", "0.5"]
        run(with_src(cmd), "STEP 1 cortex mask")
    elif lo <= 1:
        print("[STEP 1] skip cortex (--skip-cortex)")

    # --- step 2: main stitch (diasumsikan sudah ada) ---
    if lo <= 2 <= hi:
        if stitched.exists():
            print(f"[STEP 2] pakai main stitched yang ada: {stitched}")
        else:
            print(f"[STEP 2][WARN] {stitched} tak ada. Jalankan 07_infer + 08_stitch dulu (GPU).")

    # --- step 3: apply per-class threshold (CPU) ---
    if lo <= 3 <= hi:
        cmd = [PY, SCRIPTS/"03_apply_threshold.py",
               "--in-dir", stitched, "--out-dir", thresholded,
               "--thr-json", THR_JSON]
        run(with_src(cmd), "STEP 3 apply threshold")

    # --- step 4: cortex gating post-processing (CPU) ---
    if lo <= 4 <= hi:
        cmd = [PY, SCRIPTS/"04_cortex_gating_postproc.py",
               "--in-dir", thresholded, "--out-dir", gated,
               "--cortex-dir", cortex_dir]
        run(cmd, "STEP 4 cortex gating postproc")

    # --- step 5: stats + density (CPU) ---
    if lo <= 5 <= hi:
        cmd = [PY, SCRIPTS/"05_compute_stats.py",
               "--gated-dir", gated, "--cortex-dir", cortex_dir,
               "--mrcnn-dir", MRCNN_DIR,
               "--out", post / f"stats_m2f_{split}.json",
               "--mpp", str(args.mpp)]
        run(cmd, "STEP 5 stats")

    # --- step 6: export LabelMe+ASAP+overlay (CPU) ---
    if lo <= 6 <= hi:
        cmd = [PY, SCRIPTS/"06_export_results.py",
               "--gated-dir", gated, "--wsi-dir", WSI_DIR,
               "--out-dir", export_dir]
        run(with_src(cmd), "STEP 6 export")

    print(f"\n[PIPELINE DONE] split={split} "
          f"source={'ALL' if args.all or not src else src} steps {lo}-{hi}")


if __name__ == "__main__":
    main()
