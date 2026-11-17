"""
Single visible place for every setting in the pipeline.

Nothing is hidden and nothing is imported from the old code. Every stage
imports this file, and every stage writes a run_meta.json next to its output
recording exactly which settings produced it.

Paths here are defaults only -- every stage accepts command-line overrides,
and every path can also be set through an environment variable (SKINET_*), so
nothing in this file has to be edited to run on a new machine. Relative
defaults point inside the repository (data/, work/, models/).
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Repository root: the folder that holds skinet_m2f/, configs/ and models/.
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- paths ----
# Slide images and annotation XMLs live in different trees. Override on the
# command line if these move.
RAW_IMG_DIR = os.environ.get("SKINET_RAW_DIR", str(REPO_ROOT / "data" / "raw"))
RAW_XML_DIR = os.environ.get("SKINET_XML_DIR", RAW_IMG_DIR)   # ASAP XMLs, same folder by default

# One switch for the whole pipeline. Set M2F_WORK_DIR to run every
# stage against a different tree -- the patient-level split lives in
# work_patient, and the old slide-level results stay untouched in work.
WORK_DIR = os.environ.get("M2F_WORK_DIR", str(REPO_ROOT / "work"))

IMAGE_EXTS = [".tif", ".tiff", ".jpg", ".jpeg", ".png"]

# -------------------------------------------------------------- classes ----
# Order is fixed. COCO category ids are the index in this list plus one.
# Anything added or reordered here invalidates every trained checkpoint.
STRUCTURE_CLASSES = [
    "tubule_sain",
    "tubule_atrophique",
    "vaisseau",
    "nsg",
    "intima",
    "nsg_partiel",
    "media",
    "veine",
    "nsg_complet",
    "pac",
]

# Present in the same XML files but not predicted by the structure model.
REGION_CLASSES = ["cortex", "capsule", "medullaire"]

# Annotated but excluded from the structure model.
OTHER_CLASSES = ["artefact"]

# Raw XML label -> canonical name. Anything not listed and not already
# canonical is reported as unknown by stage 01 rather than silently dropped.
LABEL_ALIASES = {
    "tubule sain": "tubule_sain",
    "tubule saine": "tubule_sain",
    "tubule sains": "tubule_sain",
    "tubule atrophique": "tubule_atrophique",
    "tubule atrophie": "tubule_atrophique",
    "tubule atrophie\u0301": "tubule_atrophique",
    "vaisseaux": "vaisseau",
    "veines": "veine",
    "nsg partiel": "nsg_partiel",
    "nsg complet": "nsg_complet",
    "me\u0301dia": "media",
    "me\u0301dullaire": "medullaire",
    "medulla": "medullaire",
    "artefacts": "artefact",
}


def canonical_label(raw):
    """Normalise one raw XML label. Returns None for blanks."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if s == "" or s == "none":
        return None
    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]
    return s.replace(" ", "_")


ALL_KNOWN = STRUCTURE_CLASSES + REGION_CLASSES + OTHER_CLASSES

# ---------------------------------------------------------------- split ----
SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / val / test, by source slide

# --------------------------------------------------------------- tiling ----
TILE_SIZE = 1024
OVERLAP_RATIO = 0.333                  # stride = TILE_SIZE * (1 - this)

# Keep an object in a tile only if at least this fraction of the WHOLE object
# is visible there. Legacy MRCNN used 0.10 for every class; kept identical so
# this stays a faithful reproduction until we deliberately change it.
MIN_VISIBLE_RATIO = 0.10
MIN_VISIBLE_AREA_PX = 100

# Drop a tile whose cortex coverage falls below this. The legacy main model
# did exactly this at data-build time. Requires cortex polygons in the XML.
MIN_TILE_CORTEX = 0.10

# Write tiles that contain no annotation at all, per split. The legacy system
# dropped them everywhere; keeping them in val/test makes checkpoint selection
# and evaluation see the false positives that inference actually produces.
KEEP_EMPTY_TILES = {"train": False, "val": True, "test": True}


def stride():
    return int(round(TILE_SIZE * (1.0 - OVERLAP_RATIO)))


# ------------------------------------------------------------- run meta ----
def _git_hash():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.STDOUT)
        return out.decode().strip()
    except Exception:
        return None


def settings_snapshot():
    """Every tunable in this file, as plain data."""
    return {
        "raw_img_dir": RAW_IMG_DIR,
        "raw_xml_dir": RAW_XML_DIR,
        "work_dir": WORK_DIR,
        "structure_classes": STRUCTURE_CLASSES,
        "region_classes": REGION_CLASSES,
        "other_classes": OTHER_CLASSES,
        "split_seed": SPLIT_SEED,
        "split_fractions": list(SPLIT_FRACTIONS),
        "tile_size": TILE_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "stride": stride(),
        "min_visible_ratio": MIN_VISIBLE_RATIO,
        "min_visible_area_px": MIN_VISIBLE_AREA_PX,
        "min_tile_cortex": MIN_TILE_CORTEX,
        "keep_empty_tiles": KEEP_EMPTY_TILES,
        "arch": dict((k, dict(v)) for k, v in ARCH.items()),
        "train": dict(TRAIN),
        "infer": dict(INFER),
        "postproc": dict(POSTPROC),
    }


def write_run_meta(out_dir, stage, args=None, extra=None):
    """Record what produced the files in out_dir. Call at the end of every stage."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stage": stage,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "host": platform.node(),
        "git_hash": _git_hash(),
        "settings": settings_snapshot(),
    }
    if args is not None:
        meta["args"] = dict(vars(args))
    if extra is not None:
        meta["result"] = extra
    path = out_dir / "run_meta.json"
    with open(str(path), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("wrote {}".format(path))
    return path

# ----------------------------------------------------------------- arch ----
# Two architectures share every other stage of this pipeline, so a comparison
# between them has exactly one variable. Mask R-CNN is the family the legacy
# system used, which is what makes it the comparison worth running.
#
# ResNet-101 rather than ResNet-50: closer in capacity to Swin-B, so the
# difference read off the table is architectural rather than a size gap.
#
# The yaml ships inside detectron2. The weights do not — fetch them once on a
# node that has network access:
#   wget -P models/pretrained \
#     https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl
ARCH = {
    "mask2former": {
        "yaml": None,          # falls back to BASE_YAML below
        "weights": None,       # falls back to PRETRAINED below
        "from_model_zoo": False,
    },
    "maskrcnn": {
        "yaml": "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        "weights": os.environ.get("SKINET_MRCNN_PRETRAINED",
                                  str(REPO_ROOT / "models" / "pretrained" / "model_final_a3ec72.pkl")),
        "from_model_zoo": True,
    },
}


# ------------------------------------------------------------- training ----
# Mask2Former Swin-B. Absolute paths on purpose: these yaml files inherit from
# others through _BASE_, so copying one out of its config tree breaks the
# chain. A relative value here is resolved against the pipeline folder instead.
BASE_YAML = os.environ.get("SKINET_BASE_YAML",
                           str(REPO_ROOT / "configs" / "main_instance_swin_base.yaml"))
# COCO-pretrained Mask2Former Swin-B (IN21k, 384) instance checkpoint from the
# Mask2Former model zoo. Only needed to train from scratch, not for inference.
PRETRAINED = os.environ.get("SKINET_M2F_PRETRAINED",
                            str(REPO_ROOT / "models" / "pretrained" / "mask2former_swin_base.pkl"))

# The baseline is deliberately plain: no repeat sampler and no per-class loss
# weights. The legacy Mask R-CNN had neither, so this keeps the comparison
# fair, and it leaves the rare-class experiment something clean to move.
TRAIN = {
    "image_size": 1024,        # LSJ target size
    "min_scale": 0.5,          # 0.1 destroyed small structures
    "max_scale": 2.0,
    "test_size": 1024,
    "ims_per_batch": 1,
    "base_lr": 2.5e-5,         # tuned for batch 1; scale with batch size
    "max_iter": 40000,
    "steps": (28000, 36000),
    "eval_period": 2500,
    "checkpoint_period": 2500,
    "best_metric": "segm/AP",
    "seed": 42,

    # "step"   -> the default schedule, LR dropped at the steps above
    # "cosine" -> a single smooth decay across the whole run, as the legacy
    #             system used. With step decay the model flattens once the LR
    #             is cut; cosine keeps it alive far longer, so a long run is
    #             only a fair test of length under this schedule.
    "lr_schedule": "step",

    # Rotation by multiples of 90 degrees. The legacy system had this and
    # nothing else. Kidney tissue has no canonical orientation, so it costs
    # nothing and the model sees four times the arrangements.
    "rot90": False,
}


# ------------------------------------------------------------- inference ----
# Keep the floor low here. Post-processing does the real filtering, and it can
# only work with candidates it actually receives -- anything dropped at this
# point is gone for good.
INFER = {
    "score_floor": 0.05,
}


# ------------------------------------------------------ post-processing ----
# Ported stage by stage from the legacy system, in its execution order. Every
# stage deletes or merges whole instances; none of them ever edits pixels.
POSTPROC = {
    # Applied first. The model returns everything above the inference floor;
    # this is where the real confidence cut happens, as in the legacy system.
    "min_confidence": 0.50,

    # 0.5 cortex filter: drop instances lying mostly outside the annotated
    #     cortex. The legacy system achieves this by blanking those pixels before
    #     the model ever sees them; filtering the output instead reaches the same
    #     place with no train/test mismatch, since the model runs exactly as it
    #     was trained. An instance is kept when at least this fraction of it sits
    #     inside the cortex. 0 disables the stage.
    "cortex_min_inside": 0.0,

    # 1. same-class fusion: tiles overlap, so one object arrives several times.
    "fuse_bb": 0.1,
    "fuse_mask": 0.1,

    # 2. border filter: an object cut by the edge of the field has the wrong
    #    shape and the wrong area. The legacy rule measured how much of the mask
    #    fell on the blacked-out region of its cleaned images; these ROIs have no
    #    such region, so the equivalent here is contact with the image edge,
    #    measured against the object's own size.
    "border_threshold": 0.25,
    "border_classes": ["veine", "nsg", "intima", "media"],

    # 3 and 5. containment. A child only counts if it sits inside a parent.
    #    nsg is dropped when it holds neither child — every glomerulus carries
    #    exactly one in this data. vaisseau is kept regardless, since its wall
    #    layers are often missed.
    "orphan_bb": 0.3,
    "orphan_mask": 0.3,
    "hierarchy": {
        "nsg": {"contains": ["nsg_complet", "nsg_partiel"], "keep_if_no_child": False},
        "vaisseau": {"contains": ["intima", "media"], "keep_if_no_child": True},
    },

    # 4. priority. When two overlapping instances cannot both be right, the
    #    listed class wins and the other is removed. Read each line as
    #    "this class erases these". pac, intima and media erase nothing.
    "priority_bb": 0.3,
    "priority_mask": 0.3,
    "erases": {
        "tubule_sain":       ["tubule_atrophique", "pac", "veine", "intima", "media"],
        "tubule_atrophique": ["pac", "veine", "intima", "media"],
        "nsg_complet":       ["tubule_sain", "tubule_atrophique", "pac", "vaisseau", "veine"],
        "nsg_partiel":       ["tubule_sain", "tubule_atrophique", "pac", "vaisseau", "veine"],
        "nsg":               ["tubule_sain", "tubule_atrophique", "pac", "vaisseau", "veine"],
        "vaisseau":          ["tubule_sain", "tubule_atrophique", "pac", "veine"],
        "veine":             ["tubule_sain", "tubule_atrophique"],
        "pac":               [],
        "intima":            [],
        "media":             [],
    },

    # 6. cross-class fusion, only within a listed group. One physical object
    #    predicted under two grades becomes one instance, labelled by the
    #    higher-scoring member. nsg is deliberately NOT in any group: it is the
    #    parent of the other two, not a competing grade.
    "class_fusion_bb": 0.1,
    "class_fusion_mask": 0.1,
    "compatible": [["nsg_complet", "nsg_partiel"]],

    # Which member of a fused pair gives the surviving label.
    #   "score"      the higher-scoring one, as the legacy system did
    #   "whole_view" prefer a member that some tile saw whole, score breaking
    #                ties. Complete and partial differ by whether the section
    #                passes through the vascular pole, so a tile that cut the
    #                glomerulus may have cut away the very feature the call
    #                depends on. Falls back to score when the flag is absent.
    "class_fusion_prefer": "whole_view",

    # 7. anything this small is debris.
    "min_area_px": 50,
}

# Released structure model and its validation-tuned per-class thresholds. These
# are what run_inference.sh uses; a model trained with s04 lives under WORK_DIR.
RELEASED_WEIGHTS = os.environ.get("SKINET_WEIGHTS",
                                  str(REPO_ROOT / "models" / "skinet_m2f_structures.pth"))
RELEASED_THRESHOLDS = str(REPO_ROOT / "configs" / "thresholds" / "patient_100k.json")

# The pixel size of the exported regions, confirmed against the scale bar burnt
# into them. The figure printed in the original description of the cohort is
# wrong; this is the one to use for any conversion to physical units.
MPP = 0.227

# ------------------------------------------------------------ labelling ----
# Taken verbatim from the clinical configuration (skinet_intra_glom.json), so
# that a figure produced here can be laid beside one produced by the deployed
# system without a reader having to translate.
#
# Two of these names carry information we did not have at first. intima and
# media are named for the LAMINAE that bound them, internal and external, which
# is exactly why they are siblings inside the artery rather than nested one
# inside the other.
#
# One name is actively misleading and should NOT be reused in the thesis text.
# "Complete" and "Partial Glomeruli" suggest a profile cut short, and that
# reading cost two days. The distinction is whether the plane of section passes
# through the vascular pole (M. Legendre, personal communication). Keep these
# strings for figures that must match the clinical system; describe the classes
# properly in prose. "Globally Sclerosis Glomeruli" is also ungrammatical and is
# reproduced here only for that same consistency.
DISPLAY_NAME = {
    "tubule_sain":       "Non Atrophic Tubule",
    "tubule_atrophique": "Atrophic Tubule",
    "nsg":               "NSG",
    "nsg_complet":       "Complete Glomeruli",
    "nsg_partiel":       "Partial Glomeruli",
    "pac":               "Globally Sclerosis Glomeruli",
    "veine":             "Vein",
    "vaisseau":          "Artery",
    "media":             "External Elastic Lamina",
    "intima":            "Internal Elastic Lamina",
    "cortex":            "Cortex",
    "medullaire":        "Medulla",
    "capsule":           "Fibrous Capsule",
    "artefact":          "Artefact",
}

# Short forms for axis ticks and other places a full name will not fit.
SHORT_NAME = {
    "tubule_sain":       "Non atrophic tubule",
    "tubule_atrophique": "Atrophic tubule",
    "nsg":               "NSG",
    "nsg_complet":       "Complete glomeruli",
    "nsg_partiel":       "Partial glomeruli",
    "pac":               "Sclerotic glomeruli",
    "veine":             "Vein",
    "vaisseau":          "Artery",
    "media":             "External elastic lamina",
    "intima":            "Internal elastic lamina",
    "cortex":            "Cortex",
    "medullaire":        "Medulla",
    "capsule":           "Fibrous capsule",
}

# Taken from the same clinical configuration, converted from its RGB to the BGR
# that OpenCV expects. Matching the deployed system means a pathologist reads a
# figure from this work without relearning the colours.
#
# One caveat: nsg_complet and nsg_partiel are both dark there and are hard to
# separate at whole-slide scale. LEGIBLE_COLOR below keeps every other class
# unchanged and lightens only those two, for figures where the two subtypes
# have to be told apart at a glance.
CLASS_COLOR = {
    "tubule_sain":       (0, 0, 255),
    "tubule_atrophique": (0, 153, 255),
    "nsg":               (0, 255, 204),
    "nsg_complet":       (60, 120, 0),
    "nsg_partiel":       (15, 30, 120),
    "pac":               (255, 255, 0),
    "veine":             (255, 102, 0),
    "vaisseau":          (255, 0, 51),
    "media":             (195, 10, 108),
    "intima":            (235, 35, 255),
    "cortex":            (0, 0, 255),
    "medullaire":        (0, 255, 0),
    "capsule":           (255, 0, 0),
}

LEGIBLE_COLOR = dict(CLASS_COLOR)
LEGIBLE_COLOR["nsg_complet"] = (80, 200, 80)
LEGIBLE_COLOR["nsg_partiel"] = (120, 230, 160)


def display(name, short=False):
    d = SHORT_NAME if short else DISPLAY_NAME
    return d.get(name, name.replace("_", " "))


# The cortex model works on a different set of regions, at a coarser scale.
# REGION_CLASSES already exists above and carries the right order and the
# right spelling ("medullaire", not "medulla"); getting that wrong would
# silently mislabel every prediction the cortex model makes.
CLASS_SETS = {
    "main": STRUCTURE_CLASSES,
    "cortex": REGION_CLASSES,
}


# ------------------------------------------------------- cortex gating -----
# The cortex model does not work at the same magnification as the main one. Its
# training data was resized so the long side became 2048 -- a ratio of about
# 0.067 for a whole slide -- and then tiled at 2048 with stride 1366, so a
# single tile covers the entire biopsy. Feeding it full-resolution tiles would
# show it something it has never seen.
#
# The build summary records mean_resize_ratio, which is also what converts
# cortex area back to native pixels: native_area = tile_area / ratio^2.
CORTEX_MODEL = {
    "weights": os.environ.get("SKINET_CORTEX_WEIGHTS",
                              str(REPO_ROOT / "models" / "skinet_m2f_cortex.pth")),
    "target_long_side": 2048,
    "tile_size": 2048,
    "stride": 1366,
    "score": 0.50,
}