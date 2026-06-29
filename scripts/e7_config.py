"""
E7 Clinical Pipeline - single source of truth.

Every stage (audit, split, tiling, train, inference, fusion, filtering,
morphometry, eval) imports from here. Changing a rule = changing one place.

Encodes:
  - the class definitions for both branches,
  - the tiling rules ported from the MRCNN SKINET project,
  - the post-processing rules (fusion / priority / hierarchy) from MRCNN,
  - the training decisions we established empirically from the diagnostics.

Lines marked  # DECISION  need confirmation (anatomical/clinical -> ask Mathieu,
or data-dependent -> measure first). Sensible defaults are set so the pipeline
runs today.
"""

from pathlib import Path
import os

# ============================================================================
# Paths
# ============================================================================
# Override with environment variables for portability (defaults preserve the
# original CCuB layout, so existing runs are unaffected):
#   SKINET_ROOT      -> ROOT       (new clinical-pipeline tree)
#   SKINET_OLD_ROOT  -> OLD_ROOT   (legacy tree with pretrained weights/configs)
ROOT = Path(os.environ.get("SKINET_ROOT", "/work/lead/mk657917/SKINET"))
OLD_ROOT = Path(os.environ.get("SKINET_OLD_ROOT",
                               str(ROOT).replace("SKINET", "skinet")))

RAW_MAIN_DIR = ROOT / "data/raw/main"
RAW_CORTEX_DIR = ROOT / "data/raw/cortex"

COCO_MAIN_DIR = ROOT / "data/coco/e7_clinical_pipeline/main"
COCO_CORTEX_DIR = ROOT / "data/coco/e7_clinical_pipeline/cortex"

PREP_DIR = ROOT / "outputs/preprocessing/e7_clinical_pipeline"
OUT_MAIN = ROOT / "outputs/e7_clinical_pipeline/main_m2f_swin_b"
OUT_CORTEX = ROOT / "outputs/e7_clinical_pipeline/cortex_m2f_swin_b"

PRETRAINED = OLD_ROOT / "pretrained/mask2former_swin_base.pkl"
BASE_CONFIG = OLD_ROOT / "configs/mask2former/main_instance_swin_base.yaml"

# ============================================================================
# Classes
# ============================================================================
MAIN_CLASSES = [
    "tubule_sain", "tubule_atrophique", "vaisseau", "nsg", "intima",
    "nsg_partiel", "media", "veine", "nsg_complet", "pac",
]
CORTEX_CLASSES = ["cortex", "capsule", "medullaire"]

# COCO category ids start at 1, in list order.
MAIN_ID = {c: i + 1 for i, c in enumerate(MAIN_CLASSES)}
CORTEX_ID = {c: i + 1 for i, c in enumerate(CORTEX_CLASSES)}

EXCLUDED_MAIN_LABELS = ["cortex", "capsule", "medullaire", "artefact", "none"]

LABEL_NORMALIZATION = {
    "tubule sain": "tubule_sain", "tubule saine": "tubule_sain", "tubule sains": "tubule_sain",
    "tubule atrophique": "tubule_atrophique", "tubule atrophie": "tubule_atrophique",
    "tubule atrophié": "tubule_atrophique",
    "vaisseau": "vaisseau", "vaisseaux": "vaisseau",
    "veine": "veine", "veines": "veine",
    "nsg": "nsg", "nsg partiel": "nsg_partiel", "nsg complet": "nsg_complet",
    "intima": "intima", "media": "media", "média": "media", "pac": "pac",
    "cortex": "cortex", "capsule": "capsule",
    "medullaire": "medullaire", "médullaire": "medullaire", "medulla": "medullaire",
    "artefact": "artefact", "artifact": "artefact",
}

DATASET_NAMES_MAIN = {s: f"skinet_e7_main_{s}" for s in ["train", "val", "test"]}
DATASET_NAMES_CORTEX = {s: f"skinet_e7_cortex_{s}" for s in ["train", "val", "test"]}

# ============================================================================
# Split
# ============================================================================
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test

# ============================================================================
# Tiling rules  (ported from MRCNN datasetDivider, with diagnostic fixes)
# ============================================================================
TILE_SIZE = 1024
OVERLAP_RATIO = 0.333  # -> stride = round(TILE_SIZE * (1 - OVERLAP_RATIO))

# MAIN: native-resolution tiling. Keep a fragment if >= this fraction of the
# WHOLE object is visible in the tile. MRCNN used 0.10; our E7 used 0.03, which
# leaked boundary slivers into the noisy COCO "small" bucket. Back to 0.10.
MAIN_MIN_VISIBLE_RATIO = 0.10
MAIN_MIN_VISIBLE_AREA_PX = 100  # absolute floor

# MAIN tiles: drop a tile if < this fraction of it is inside the cortex mask
# (cortex-aware tiling, as in MRCNN). Requires the cortex masks; set to 0.0 to
# disable until the cortex branch exists.
MAIN_MIN_TILE_CORTEX = 0.10  # DECISION: enable once cortex masks are available

# CORTEX: resize-then-tile (low magnification, ~x25 paper view), and keep a
# region mask if it covers >= this fraction of the TILE (region classes are huge,
# so object-ratio retention would wrongly drop almost every tile).
CORTEX_TARGET_LONG_SIDE = 2048  # DECISION: confirm vs native cortex resolution
CORTEX_MIN_TILE_COVERAGE = 0.05

# Targeted offline augmentation: our diagnostics showed frequency is NOT the
# bottleneck (pac, rarest, scored well), so this is OFF by default. Online LSJ in
# the M2F mapper provides augmentation. Enable only if a future experiment shows
# a frequency-driven gap.
USE_TARGETED_AUG = False
TARGETED_AUG_CLASSES = ["pac", "veine", "media", "intima",
                        "nsg_complet", "nsg_partiel", "vaisseau"]
AUG_TRANSFORMS = ["flip_h", "flip_v", "rot90", "rot180", "rot270"]

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".jp2", ".tif", ".tiff"]

# ============================================================================
# Training  (M2F SWIN-B, established empirically)
# ============================================================================
# Val AP plateaus ~30k at batch 1 on ~200 images; 120k was 3x over-trained.
TRAIN = dict(
    input_size=1024,
    min_scale=0.5,           # 0.1 default destroyed small structures
    max_scale=2.0,
    test_size=1024,
    ims_per_batch=1,
    base_lr=2.5e-5,          # tuned for batch 1; scale ~linearly if batch grows
    main=dict(max_iter=40000, steps=(28000, 36000)),
    cortex=dict(max_iter=25000, steps=(17000, 22000)),  # 3 large classes, faster
    eval_period=2500,
    checkpoint_period=2500,
    best_metric="segm/AP",
)

# --- Class-imbalance handling for the glomerulus grade entanglement -----------
# Diagnosis: the model scored nsg highest even on sclerotic glomeruli, so NO
# post-processing operating point recovered all grades (margin 0.0 -> nsg perfect
# but complet/partiel = 0; margin 0.05 -> the reverse). This is a model-level
# score-calibration problem. These two mechanisms (single-model retrain, no extra
# model) recalibrate so that plain "highest score" works for every grade.
USE_REPEAT_SAMPLER = True
REPEAT_THRESHOLD = 0.5      # oversample tiles whose rarest class is below this
                            # image-frequency; raise = more aggressive. Tune on val.
# Per-class classification-loss weights, injected into the M2F criterion's
# empty_weight (the per-class weight tensor in its cross-entropy). >1 penalises
# misclassifying that class harder -> pushes its score up. Focused on the
# entangled glomerulus grades; the rest stay 1.0. Tune on val.
CLASS_LOSS_WEIGHTS = {
    "nsg_complet": 4.0,
    "nsg_partiel": 3.0,
    "intima": 2.0, "media": 2.0, "veine": 2.0,   # mild help for other rare/hard
}

# ============================================================================
# Inference + post-processing rules  (ported from MRCNN post_processing.py)
# ============================================================================
MIN_CONFIDENCE = 0.5  # score floor on raw predictions before fusion

# Post-fusion cleanup (Stage 10, BEFORE priority/orphan): drop low-confidence and
# tiny fragments that fusion consolidated but did not remove. Main fix for the
# scattered tubule_atrophique fragments (precision 0.49): inference ran at
# score_thr 0.30 to feed fusion, but the 0.5 floor was never re-applied, so
# low-score FPs survived. A fused instance's score = max over its merged members,
# so a real object keeps its confident detection; pure low-score junk is dropped.
POSTFUSE_CLEANUP = dict(
    min_confidence=0.50,        # drop fused instances scored below this
    min_area_px=300,            # DECISION: tune on val; tiny-fragment floor
    min_area_px_by_class={      # thin structures need a lower floor
        "intima": 80, "media": 150,
    },
)

# Same-class duplicate fusion (tile-overlap duplicates -> UNION).
# bbox_gate="intersect": admit any pair whose bboxes overlap at all, then let mask
# IoMin decide. (The old "iou" gate wrongly blocked small edge fragments whose
# bbox-IoU with the big parent instance is tiny even though the fragment lies
# inside it -> those fragments survived as the scattered anomalies we saw. Fixed.)
FUSE_SAME_CLASS = dict(bbox_gate="intersect", bb_threshold=0.0, mask_threshold=0.1)

# Cross-class fusion: only WITHIN a compatibility group. A single physical object
# predicted under several grades -> keep ONE. The surviving label is chosen per
# group by CROSS_CLASS_RESOLUTION (aligned to CLASSES_COMPATIBILITY order).
CROSS_CLASS_FUSION = dict(bbox_gate="intersect", bb_threshold=0.0, mask_threshold=0.1)
CLASSES_COMPATIBILITY = [
    # glomerulus REMOVED: nsg is the PARENT of nsg_partiel/nsg_complet (nested
    # sclerosis sub-regions, confirmed 99.7%/100% spatial containment in GT and
    # 96-99.7% in raw predictions), NOT a competing grade. Collapsing them with
    # pick_winner destroyed nsg (335->9). Handled via CLASSES_HIERARCHY instead;
    # the clinical grade is derived from which child a glomerulus contains.
    ["tubule_sain", "tubule_atrophique"],      # tubule grade -> one instance (truly exclusive)
    # DECISION (Mathieu): should vaisseau/veine ever be fused? default: no.
]
# How the surviving label is picked in each compatibility group (same index order).
# Entry forms:
#   "score"                              -> plain highest score
#   [class, ...]                         -> earliest-listed class present wins (no
#                                           score gate)
#   {"priority":[...], "score_margin":m} -> SCORE-AWARE priority: among members
#                                           scoring within m of the cluster's best,
#                                           the earliest-listed class wins; else
#                                           highest score.
# Glomerulus uses score-aware priority: prefer the SCLEROTIC grade ONLY when its
# score is competitive with the best in the cluster. (Unconditional priority made
# a single low-score nsg_complet relabel even normal glomeruli -> nsg collapsed to
# 0 and complet was massively over-called. The margin balances both collapses.)
# score_margin is the knob to tune on VAL.
CROSS_CLASS_RESOLUTION = [
    # glomerulus resolution REMOVED with its compatibility group (see above).
    "score",                                    # group 0 tubule: highest score
]

# Priority filtering for overlapping mutually-exclusive predictions.
PRIORITY_FILTER = dict(bb_threshold=0.5, mask_threshold=0.2)

# Orphan / containment hierarchy. A child kept only if inside a parent by
# >= mask_threshold; parent kept regardless if keep_if_no_child True.
# This is the principled fix for the vessel-class false positives (intima/media
# predicted outside any vessel get removed).
# Relaxed from (0.5, 0.5): intima is a thin ring -- requiring 50% of it inside the
# predicted vaisseau mask plus bbox-IoU 0.5 removed almost all TRUE intima
# (recall 0.03). Lower containment to 0.2 and bbox gate to 0.2, and KEEP a
# high-confidence child even with no detected parent (the parent vessel may simply
# have been missed). Removes low-score walls outside vessels (real FPs) while
# sparing confident wall detections.
ORPHAN_FILTER = dict(bb_threshold=0.2, mask_threshold=0.2, keep_if_score_ge=0.50)
CLASSES_HIERARCHY = {
    # parent : {"contains": [children], "keep_if_no_child": bool}
    "vaisseau": {"contains": ["intima", "media"], "keep_if_no_child": True},
    # glomerulus hierarchy: nsg is the whole glomerulus; partiel/complet are
    # nested sclerosis sub-regions (confirmed by GT + raw-pred containment).
    # Keep the glomerulus always; keep a lesion child if it sits inside one.
    "nsg": {"contains": ["nsg_partiel", "nsg_complet"], "keep_if_no_child": True},
    # DECISION (Mathieu): does 'veine' carry annotated intima/media? default no.
    # DECISION (Mathieu): may a vaisseau be deleted if no wall layer? default keep.
}

# Post-processing chain order (do not reorder: union before filtering).
POSTPROC_CHAIN = [
    "fuse_results",        # stitch tiles -> global coords
    "fuse_same_class",     # union duplicates
    "fuse_cross_class",    # grade groups -> one label
    "filter_priority",     # resolve mutually-exclusive overlaps
    "filter_orphan",       # hierarchy / containment
]

# ============================================================================
# SAM-Med2D refinement  (refinement only, never detection)
# ============================================================================
SAM = dict(
    prompt="bbox",                 # bbox prompt from stitched global instance
    apply_after="fusion_and_filter",
    # Quality gate: keep M2F mask unless SAM is plausibly better.
    min_area_ratio=0.5,            # if SAM area < 0.5x M2F -> keep M2F
    max_area_ratio=2.0,            # if SAM area > 2.0x M2F -> keep M2F
    min_iou_with_m2f=0.5,          # if SAM vs M2F IoU < 0.5 -> flag for review
    refine_classes="all",          # DECISION: maybe only vessel-wall classes
)

# ============================================================================
# Morphometry
# ============================================================================
# Count a child inside a base region only if >= this fraction is included.
MORPHO_INCLUSION_THRESHOLD = 0.90
# Cortex denominator: native_area = tile_area / resize_ratio**2 (ratio per tile).
CORTEX_RELEVANT_CLASSES = [  # classes whose density/ratio is normalized by cortex
    "tubule_sain", "tubule_atrophique", "nsg", "nsg_partiel", "nsg_complet",
]

# ============================================================================
# Evaluation
# ============================================================================
EVAL = dict(
    count_iou=0.30,        # detection counting (clinical "detection rate")
    ap_iou=0.50,           # COCO AP operating point for P/R/F1
    confusion_iou=0.50,
    score_thr=0.50,
)

# Classes that are intrinsically hard (confirmed by Marechal et al. 2022 with a
# different model) - report these explicitly, don't treat low AP as a model bug.
KNOWN_HARD_CLASSES = ["intima", "media", "veine", "vaisseau",
                      "nsg_partiel", "nsg_complet", "tubule_atrophique"]


def stride():
    return int(round(TILE_SIZE * (1.0 - OVERLAP_RATIO)))


def normalize_label(label):
    if label is None:
        return None
    s = str(label).strip().lower().replace("_", " ").replace("-", " ").strip()
    if s == "":
        return None
    return LABEL_NORMALIZATION.get(s, s.replace(" ", "_"))