# SKINET Clinical Pipeline (Cara B)

Reproduksi sistem klinis MRCNN (Marechal et al. 2022) dengan **Mask2Former
(Swin-B)**, memakai **cortex-gating di post-processing** (bukan input cropping).

## Kenapa Cara B (post-proc gating, bukan cleaned-image crop)

MRCNN asli meng-crop WSI ke cortex (area luar cortex jadi hitam) sebelum main
model. Mask R-CNN bisa karena **dilatih dengan cleaned image**. M2F **tidak** —
memberi cleaned image (area hitam besar) ke M2F menyebabkan domain shift:
prediksi kacau (halusinasi vaisseau, tubule_atrophique hilang). Terbukti di
ML9360N14: cleaned crop -> tubule_sain=15, vaisseau=34 (buruk); WSI normal +
post-proc cortex gating -> tubule_sain=67, vaisseau=2 (baik).

Cara B: main M2F jalan pada **WSI normal** (background putih, sesuai training),
lalu struktur di luar cortex dibuang di **post-processing** (border_filter
dengan cortex mask). Setia pada fungsi klinis (analisis dibatasi ke cortex),
kompatibel dengan M2F.

## Struktur

```
skinet_clinical_pipeline/
├── run_pipeline.py                    # orchestrator modular (step 1-6)
├── config/
│   └── postfusion_threshold.json      # threshold per-kelas val-tuned
└── scripts/
    ├── 01_gen_cortex_mask.py          # [GPU] WSI -> cortex mask PNG
    ├── 03_apply_threshold.py          # [CPU] filter per-class score threshold
    ├── 04_cortex_gating_postproc.py   # [CPU] 7-stage postproc + cortex gating
    ├── 05_compute_stats.py            # [CPU] count/area/density vs MRCNN
    ├── 06_export_results.py           # [CPU] LabelMe+ASAP+overlay per source
    └── verify_cortex_vs_gt.py         # [CPU] validasi cortex mask vs GT (IoU/Dice)
```

Step 2 (main M2F inference + 08_stitch) TIDAK disertakan — itu pipeline M2F
utama yang sudah ada (07_infer_tiles + 08_stitch). Pipeline ini mulai dari
output stitched-nya.

## Alur (per WSI)

```
1. cortex M2F        -> cortex mask PNG (label 1=cortex)
2. main M2F normal   -> stitched (SUDAH ADA, pipeline utama)
3. apply threshold   -> buang instance < threshold per-kelas
4. cortex gating     -> 7-stage postproc, buang struktur luar cortex
5. stats             -> count/area/density + banding MRCNN
6. export            -> {src}.json (LabelMe), {src}.xml (ASAP),
                        {src}_Predicted.jpg (overlay), _Predicted_clean.jpg
```

## Cara pakai

```bash
# satu WSI, step 3-6 (cortex + main sudah ada):
python run_pipeline.py --source ML9360N14 --split test --from-step 3 --skip-cortex

# semua WSI test, step 3-6:
python run_pipeline.py --all --split test --from-step 3 --skip-cortex

# full dari awal (butuh GPU utk step 1):
python run_pipeline.py --all --split test

# density per mm^2 (kalau tahu microns-per-pixel):
python run_pipeline.py --all --split test --from-step 3 --skip-cortex --mpp 0.5
```

Tiap script bisa dijalankan sendiri — lihat `--help` masing-masing.

## PENTING - sesuaikan sebelum jalan

1. **PATH di `run_pipeline.py`** (atas file): SKINET root, checkpoint cortex,
   WSI dir, dst. Sesuaikan dengan cluster.
2. **Import M2F di `01_gen_cortex_mask.py`**: butuh env M2F (detectron2 +
   Mask2Former) untuk inference cortex model. Step lain (3-6) CPU murni.
3. **cortex mask dir**: step 4 baca cortex PNG via `--cortex-dir` atau env
   `SKINET_CORTEX_DIR`. Kalau kosong -> fallback border tepi-image.

## Hasil (AUG, 44 WSI test, via 14_eval --mrcnn-match)

```
MRCNN (referensi)*              macro-F1 0.957
M2F Cara B (threshold+gating)   macro-F1 0.700   <- pipeline ini
M2F cortex-gating (no thresh)   macro-F1 0.640
M2F tanpa gating                macro-F1 0.346
```

*MRCNN 0.957: reference GT diturunkan dari MRCNN (IoU mask MRCNN-vs-GT = 0.95),
jadi MRCNN dievaluasi terhadap turunannya sendiri. Bukan perbandingan setara.

Cara B dipilih sbg pipeline utama untuk **validitas klinis** (cortex gating
anatomis, per Marechal), **density denominator** (cortex area, per mm^2), dan
**output interoperability** (format setara MRCNN utk tool klinis) — trade-off
F1 minimal demi sistem yg benar secara klinis.

## Catatan / TODO

- **veine** recall rendah (0.167) di test — threshold 0.85 + gating terlalu
  agresif (veine sering di perbatasan cortex-medulla). Perbaiki: turunkan
  threshold veine atau kecualikan dari gating agresif.
- **EM0812N5** cortex 0% (cortex M2F salah label medullaire; stain berbeda).
  Fallback: pakai cortex+medullaire kalau cortex <5%.
- density saat ini per-px; butuh mpp WSI untuk per-mm^2 (ala Marechal).
