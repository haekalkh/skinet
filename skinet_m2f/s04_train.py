"""
Stage 04 - train Mask2Former on the tiles built by stage 03.

Registers the three COCO splits, builds the config with every override taken
from config.py (nothing hidden in the yaml), trains, and evaluates on val every
eval_period iterations, keeping the checkpoint with the best segm/AP.

The baseline here is plain Mask2Former: no repeat sampler, no per-class loss
weights. The legacy Mask R-CNN had neither, so the comparison stays fair and
the rare-class experiment has something clean to move.

Outputs (into --out-dir):
  checkpoints/      periodic and best checkpoints
  metrics.json      Detectron2's training log
  run_meta.json

Usage:
  python s04_train.py --run baseline_seed42
  python s04_train.py --run smoke --max-iter 100 --eval-period 50
"""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetMapper, MetadataCatalog,
                             build_detection_train_loader)
from detectron2.data import transforms as T
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# mask2former is imported lazily: importing it is what registers the
# MaskFormer architecture, and the Mask R-CNN arm must not depend on it.

import config as C

DATASET = dict((s, "kidney_%s" % s) for s in ["train", "val", "test"])


def register(tiles_dir):
    tiles_dir = Path(tiles_dir)
    for split, name in DATASET.items():
        if name in MetadataCatalog.list():
            continue
        json_file = tiles_dir / "annotations" / ("instances_%s.json" % split)
        if not json_file.exists():
            print("[warn] missing %s" % json_file)
            continue
        register_coco_instances(name, {}, str(json_file), str(tiles_dir))
        MetadataCatalog.get(name).thing_classes = list(C.STRUCTURE_CLASSES)


def lsj_augmentations(cfg, rot90):
    """Large-scale jitter, spelled out so both architectures get the same one.

    Mask2Former ships this inside its own mapper; Mask R-CNN would otherwise
    train under Detectron2's defaults, which would make the comparison a test
    of two augmentation recipes rather than two architectures.
    """
    augs = []
    if rot90:
        augs.append(T.RandomRotation(angle=[0, 90, 180, 270],
                                     sample_style="choice", expand=False))
    size = cfg.INPUT.IMAGE_SIZE
    augs += [
        T.ResizeScale(min_scale=cfg.INPUT.MIN_SCALE,
                      max_scale=cfg.INPUT.MAX_SCALE,
                      target_height=size, target_width=size),
        T.FixedSizeCrop(crop_size=(size, size)),
        T.RandomFlip(),
    ]
    return augs


def build_mapper(cfg, arch, rot90=False):
    if arch == "maskrcnn":
        m = DatasetMapper(cfg, is_train=True,
                          augmentations=lsj_augmentations(cfg, rot90),
                          instance_mask_format=cfg.INPUT.MASK_FORMAT)
        print("mapper: detectron2 DatasetMapper with LSJ%s"
              % ("  + rot90" if rot90 else ""))
        return m

    try:
        from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper \
            import COCOInstanceNewBaselineDatasetMapper as M
        mapper = M(cfg, True)
    except Exception:
        from mask2former.data.dataset_mappers.mask_former_instance_dataset_mapper \
            import MaskFormerInstanceDatasetMapper as M
        mapper = M(cfg, True)

    if rot90:
        rot = T.RandomRotation(angle=[0, 90, 180, 270],
                               sample_style="choice", expand=False)
        for attr in ("tfm_gens", "augmentations"):
            gens = getattr(mapper, attr, None)
            if gens is not None:
                try:
                    gens.insert(0, rot)
                except AttributeError:
                    setattr(mapper, attr, [rot] + list(gens))
                print("rot90 augmentation added to mapper.%s" % attr)
                break
        else:
            raise RuntimeError(
                "could not find the augmentation list on %s -- turn rot90 off "
                "or adapt build_mapper" % type(mapper).__name__)
    return mapper


class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=build_mapper(cfg, cfg.ARCH_NAME, cfg.INPUT.ROT90))

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Mask2Former's own optimizer. Mask R-CNN keeps Detectron2's.

        The block below is specific to Mask2Former: AdamW, a smaller rate for
        the backbone, and full-model gradient clipping. Mask R-CNN was designed
        around Detectron2's SGD and is left alone.

        DefaultTrainer would build plain SGD, which barely trains this model,
        and it does not understand CLIP_TYPE 'full_model'. What matters here:
        AdamW, a smaller learning rate for the backbone, no weight decay on the
        positional tables, and gradient clipping across the whole model.
        """
        if cfg.ARCH_NAME != "mask2former":
            opt = DefaultTrainer.build_optimizer(cfg, model)
            print("optimizer: detectron2 default for %s  lr %g"
                  % (cfg.ARCH_NAME, cfg.SOLVER.BASE_LR))
            return opt

        defaults = {"lr": cfg.SOLVER.BASE_LR,
                    "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                      torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm,
                      torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
                      torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
                      torch.nn.LayerNorm, torch.nn.LocalResponseNorm)

        params, seen = [], set()
        for module_name, module in model.named_modules():
            for param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in seen:
                    continue
                seen.add(value)
                hp = copy.copy(defaults)
                if "backbone" in module_name:
                    hp["lr"] = hp["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if ("relative_position_bias_table" in param_name
                        or "absolute_pos_embed" in param_name):
                    hp["weight_decay"] = 0.0
                if isinstance(module, norm_types):
                    hp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, torch.nn.Embedding):
                    hp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
                params.append(dict(params=[value], **hp))

        clip_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        full_clip = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                     and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                     and clip_val > 0.0)

        def wrap(optim_cls):
            if not full_clip:
                return optim_cls

            class ClippedOptimizer(optim_cls):
                def step(self, closure=None):
                    allp = itertools.chain(
                        *[g["params"] for g in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(allp, clip_val)
                    super().step(closure=closure)

            return ClippedOptimizer

        kind = cfg.SOLVER.OPTIMIZER
        if kind == "ADAMW":
            opt = wrap(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        elif kind == "SGD":
            opt = wrap(torch.optim.SGD)(params, cfg.SOLVER.BASE_LR,
                                        momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError("unknown SOLVER.OPTIMIZER: %s" % kind)

        if not full_clip:
            opt = maybe_add_gradient_clipping(cfg, opt)

        print("optimizer: %s  lr %g  backbone_mult %g  full_model_clip %s"
              % (kind, cfg.SOLVER.BASE_LR, cfg.SOLVER.BACKBONE_MULTIPLIER,
                 full_clip))
        return opt

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        # segm only: Mask2Former predicts no boxes, so bbox AP is always 0 and
        # computing it just slows every evaluation down.
        return COCOEvaluator(dataset_name, tasks=("segm",),
                             output_dir=output_folder)


def weights_for_query_count(weights, n_queries, out_dir):
    """Strip the query embeddings when the checkpoint disagrees on their count.

    Changing --num-queries changes the shape of query_feat and query_embed, so
    the pretrained values cannot be reused and have to be learned from scratch.
    Detectron2 normally drops parameters whose shape does not match, but
    Mask2Former's decoder installs a version-conversion hook that runs after
    that filtering and puts the keys back, so the load fails. Filtering them out
    of a copy of the checkpoint avoids the collision without touching the
    original file.
    """
    if not n_queries:
        return weights
    print("query parameters in %s:" % Path(weights).name)

    import pickle
    import numpy as np

    p = Path(weights)
    is_pkl = p.suffix == ".pkl"
    if is_pkl:
        with open(str(p), "rb") as f:
            obj = pickle.load(f, encoding="latin1")
        state = obj.get("model", obj)
    else:
        obj = torch.load(str(p), map_location="cpu")
        state = obj.get("model", obj)

    # static_query is the older name for query_feat. Mask2Former's decoder
    # renames it in a _load_from_state_dict hook that runs AFTER Detectron2 has
    # filtered mismatched shapes, so a checkpoint storing the old name slips
    # past a filter that only knows the new one.
    marks = ("query_feat", "query_embed", "static_query")
    keys = [k for k in state if any(m in k for m in marks)]
    if not keys:
        print("no query parameters found in the checkpoint -- nothing to filter")
        return weights

    drop = []
    for k in keys:
        n = int(state[k].shape[0])
        print("  %-60s %d" % (k, n))
        if n != n_queries:
            drop.append((k, n))
    if not drop:
        return weights

    for k, _ in drop:
        del state[k]
    out = Path(out_dir) / ("pretrained_q%d%s" % (n_queries, p.suffix))
    out.parent.mkdir(parents=True, exist_ok=True)
    if is_pkl:
        with open(str(out), "wb") as f:
            pickle.dump(obj, f)
    else:
        torch.save(obj, str(out))

    for k, had in drop:
        print("dropped %s (had %d, model wants %d)" % (k, had, n_queries))
    print("using filtered checkpoint %s" % out)
    return str(out)


def build_cfg(args):
    t = C.TRAIN
    cfg = get_cfg()
    add_deeplab_config(cfg)

    if args.arch == "mask2former":
        from mask2former import add_maskformer2_config
        add_maskformer2_config(cfg)

    cfg.merge_from_file(args.base_yaml)

    cfg.DATASETS.TRAIN = (DATASET["train"],)
    cfg.DATASETS.TEST = (DATASET["val"],)
    cfg.DATALOADER.NUM_WORKERS = args.workers

    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # The class count lives in a different place in each architecture.
    n = len(C.STRUCTURE_CLASSES)
    if args.arch == "mask2former":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "bitmask"
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n
        cfg.MODEL.RETINANET.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "polygon"

    cfg.INPUT.IMAGE_SIZE = t["image_size"]
    cfg.INPUT.MIN_SCALE = (args.min_scale if args.min_scale is not None
                           else t["min_scale"])
    cfg.INPUT.MAX_SCALE = (args.max_scale if args.max_scale is not None
                           else t["max_scale"])
    cfg.INPUT.MIN_SIZE_TEST = t["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = t["test_size"]

    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = tuple(s for s in t["steps"] if s < args.max_iter)
    if args.lr_schedule == "cosine":
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
        cfg.SOLVER.STEPS = ()          # unused by the cosine schedule
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.INPUT.ROT90 = bool(args.rot90)
    cfg.ARCH_NAME = args.arch
    if args.num_queries and args.arch == "mask2former":
        cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = args.num_queries
    if args.optimizer:
        cfg.SOLVER.OPTIMIZER = args.optimizer
        if args.optimizer == "SGD":
            cfg.SOLVER.MOMENTUM = 0.9

    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = args.out_dir
    cfg.freeze()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--run", required=True,
                    help="name for this run, e.g. baseline_seed42")
    ap.add_argument("--arch", default="mask2former",
                    choices=sorted(C.ARCH.keys()),
                    help="every other stage is identical between the two, so "
                         "the comparison has one variable")
    ap.add_argument("--tiles-dir", default=str(Path(C.WORK_DIR) / "s03_tiles"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--base-yaml", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seed", type=int, default=C.TRAIN["seed"])
    ap.add_argument("--batch", type=int, default=C.TRAIN["ims_per_batch"])
    ap.add_argument("--lr", type=float, default=C.TRAIN["base_lr"])
    ap.add_argument("--max-iter", type=int, default=C.TRAIN["max_iter"])
    ap.add_argument("--eval-period", type=int, default=C.TRAIN["eval_period"])
    ap.add_argument("--checkpoint-period", type=int,
                    default=C.TRAIN["checkpoint_period"])
    ap.add_argument("--lr-schedule", default=C.TRAIN["lr_schedule"],
                    choices=["step", "cosine"])
    ap.add_argument("--min-scale", type=float, default=None,
                    help="lower bound of the scale jitter. Set both bounds to "
                         "1.0 to switch it off: every slide is scanned at one "
                         "magnification, so the size variation this simulates "
                         "never occurs at inference.")
    ap.add_argument("--max-scale", type=float, default=None)
    ap.add_argument("--num-queries", type=int, default=None,
                    help="object queries per tile. Mask2Former's default of 100 "
                         "comes from COCO; the right floor for a dataset can be "
                         "read off its objects-per-tile distribution before "
                         "training. Ignored for maskrcnn.")
    ap.add_argument("--optimizer", default=None, choices=["ADAMW", "SGD"],
                    help="default comes from the yaml (ADAMW). SGD is the "
                         "legacy system's choice, but its 0.008 learning rate "
                         "was tuned for a convolutional backbone -- on a "
                         "transformer it usually diverges, so treat this as a "
                         "separate experiment with its own lr.")
    ap.add_argument("--rot90", action="store_true", default=C.TRAIN["rot90"],
                    help="rotate tiles by multiples of 90 degrees")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = str(Path(C.WORK_DIR) / "s04_train" / args.run)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    spec = C.ARCH[args.arch]

    def resolve(p):
        p = Path(p)
        return str(p if p.is_absolute() else here / p)

    if args.base_yaml is None:
        if spec["from_model_zoo"]:
            args.base_yaml = model_zoo.get_config_file(spec["yaml"])
        else:
            args.base_yaml = resolve(spec["yaml"] or C.BASE_YAML)
    if args.weights is None:
        args.weights = resolve(spec["weights"] or C.PRETRAINED)

    if args.num_queries and args.arch == "mask2former":
        args.weights = weights_for_query_count(args.weights, args.num_queries,
                                               args.out_dir)

    if args.arch != "mask2former" and args.lr == C.TRAIN["base_lr"]:
        # 2.5e-5 is an AdamW rate; SGD on Mask R-CNN needs far more.
        args.lr = 0.0025
        print("lr defaulted to %g for %s (SGD)" % (args.lr, args.arch))

    for p in (args.base_yaml, args.weights):
        if not Path(p).exists():
            raise SystemExit(
                "not found: %s\nCopy it into the pipeline folder, or pass "
                "--base-yaml / --weights." % p)

    register(args.tiles_dir)
    cfg = build_cfg(args)
    setup_logger(output=cfg.OUTPUT_DIR, name="s04_train")

    print("=" * 70)
    print("stage 04 - train | run %s  arch %s" % (args.run, args.arch))
    print("classes %d  batch %d  lr %g  iters %d  seed %d"
          % (len(C.STRUCTURE_CLASSES), args.batch, args.lr,
             args.max_iter, args.seed))
    print("scale jitter %.2f-%.2f%s"
          % (args.min_scale if args.min_scale is not None else C.TRAIN["min_scale"],
             args.max_scale if args.max_scale is not None else C.TRAIN["max_scale"],
             "  (off)" if args.min_scale == args.max_scale == 1.0 else ""))
    print("schedule %s  rot90 %s  optimizer %s  queries %s"
          % (args.lr_schedule, bool(args.rot90), args.optimizer or "from yaml",
             args.num_queries or "from yaml"))
    print("output %s" % cfg.OUTPUT_DIR)
    print("=" * 70)

    with open(os.path.join(cfg.OUTPUT_DIR, "resolved_config.yaml"), "w") as f:
        f.write(cfg.dump())

    trainer = Trainer(cfg)
    trainer.register_hooks([
        hooks.BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer,
                               C.TRAIN["best_metric"], mode="max",
                               file_prefix="model_best"),
    ])
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    best = {}
    metrics = Path(cfg.OUTPUT_DIR) / "metrics.json"
    if metrics.exists():
        vals = []
        for line in metrics.read_text().splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if C.TRAIN["best_metric"] in d:
                vals.append((d[C.TRAIN["best_metric"]], d.get("iteration")))
        if vals:
            v, it = max(vals)
            best = {"best_" + C.TRAIN["best_metric"]: v, "at_iteration": it,
                    "n_evals": len(vals)}
            print("\nbest %s = %.3f at iteration %s"
                  % (C.TRAIN["best_metric"], v, it))

    C.write_run_meta(cfg.OUTPUT_DIR, "s04_train", args, extra=best)


if __name__ == "__main__":
    main()