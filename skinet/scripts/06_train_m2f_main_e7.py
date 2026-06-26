"""
E7 Stage 6 - Train Mask2Former SWIN-B on MAIN.

Config-driven (imports e7_config). All knobs come from e7_config.TRAIN, so the
training recipe lives in one place:

  - MIN_SCALE 0.5 (0.1 default destroyed thin structures via LSJ down-scaling),
  - MAX_ITER 40000 / STEPS (28000, 36000): val AP plateaus ~30k at batch 1 on
    ~200 images; 120k was ~3x over-trained,
  - BestCheckpointer on segm/AP -> model_best.pth (last checkpoint can regress),
  - AdamW + standard M2F recipe kept verbatim (correct; not the bottleneck).

Datasets: skinet_e7_main_{train,val}. Output: e7_config.OUT_MAIN.
"""

import argparse
import copy
import itertools
import logging
import os
from pathlib import Path

import torch

from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    DatasetCatalog, MetadataCatalog,
    build_detection_train_loader, build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import (
    COCOInstanceNewBaselineDatasetMapper,
)

import e7_config as C


def register_e7_main_datasets():
    for split in ["train", "val", "test"]:
        name = C.DATASET_NAMES_MAIN[split]
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        ann_file = C.COCO_MAIN_DIR / "annotations" / f"instances_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Missing annotation file: {ann_file}")
        register_coco_instances(
            name=name,
            metadata={"thing_classes": C.MAIN_CLASSES},
            json_file=str(ann_file),
            image_root=str(C.COCO_MAIN_DIR),
        )


def build_config(resume: bool = False) -> CfgNode:
    t = C.TRAIN
    main = t["main"]

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(C.BASE_CONFIG))

    cfg.DATASETS.TRAIN = (C.DATASET_NAMES_MAIN["train"],)
    cfg.DATASETS.TEST = (C.DATASET_NAMES_MAIN["val"],)

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(C.MAIN_CLASSES)
    cfg.MODEL.WEIGHTS = "" if resume else str(C.PRETRAINED)

    # Solver (standard M2F AdamW; kept verbatim)
    cfg.SOLVER.IMS_PER_BATCH = t["ims_per_batch"]
    cfg.SOLVER.BASE_LR = t["base_lr"]
    cfg.SOLVER.MAX_ITER = main["max_iter"]
    cfg.SOLVER.STEPS = tuple(main["steps"])
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = t["checkpoint_period"]

    # Input / LSJ augmentation (small-object fix lives here)
    cfg.INPUT.IMAGE_SIZE = t["input_size"]
    cfg.INPUT.MIN_SCALE = t["min_scale"]
    cfg.INPUT.MAX_SCALE = t["max_scale"]
    cfg.INPUT.MIN_SIZE_TEST = t["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = t["test_size"]
    cfg.INPUT.MASK_FORMAT = "bitmask"

    cfg.TEST.EVAL_PERIOD = t["eval_period"]
    cfg.DATALOADER.NUM_WORKERS = 4

    if getattr(C, "USE_REPEAT_SAMPLER", False):
        cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        cfg.DATALOADER.REPEAT_THRESHOLD = C.REPEAT_THRESHOLD

    cfg.OUTPUT_DIR = str(C.OUT_MAIN / "checkpoints")
    cfg.freeze()
    return cfg


class E7MAINTrainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                      torch.nn.SyncBatchNorm, torch.nn.GroupNorm,
                      torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d,
                      torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
                      torch.nn.LocalResponseNorm)
        params, memo = [], set()
        for module_name, module in model.named_modules():
            for pname, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyp = copy.copy(defaults)
                if "backbone" in module_name:
                    hyp["lr"] = hyp["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if pname in ("relative_position_bias_table", "absolute_pos_embed"):
                    hyp["weight_decay"] = 0.0
                if isinstance(module, norm_types):
                    hyp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, torch.nn.Embedding):
                    hyp["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY_EMBED
                params.append({"params": [value], **hyp})

        clip = cfg.SOLVER.CLIP_GRADIENTS
        use_full = clip.ENABLED and clip.CLIP_TYPE == "full_model" and clip.CLIP_VALUE > 0.0

        def maybe_clip(optim_cls):
            if not use_full:
                return optim_cls
            class FullClip(optim_cls):
                def step(self, closure=None):
                    allp = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(allp, clip.CLIP_VALUE)
                    super().step(closure=closure)
            return FullClip

        if cfg.SOLVER.OPTIMIZER == "ADAMW":
            opt = maybe_clip(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        elif cfg.SOLVER.OPTIMIZER == "SGD":
            opt = maybe_clip(torch.optim.SGD)(params, cfg.SOLVER.BASE_LR,
                                              momentum=cfg.SOLVER.MOMENTUM)
        else:
            raise NotImplementedError(cfg.SOLVER.OPTIMIZER)
        if clip.ENABLED and clip.CLIP_TYPE != "full_model":
            opt = maybe_add_gradient_clipping(cfg, opt)
        return opt

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = C.OUT_MAIN / "eval" / dataset_name
        os.makedirs(output_folder, exist_ok=True)
        return DatasetEvaluators([
            COCOEvaluator(dataset_name, tasks=("segm",), distributed=True,
                          output_dir=str(output_folder))
        ])

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
        if cfg.DATALOADER.SAMPLER_TRAIN == "RepeatFactorTrainingSampler":
            dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)
            repeat_factors = (RepeatFactorTrainingSampler
                              .repeat_factors_from_category_frequency(
                                  dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD))
            sampler = RepeatFactorTrainingSampler(repeat_factors)
            return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        hl = super().build_hooks()
        if comm.is_main_process():
            best = hooks.BestCheckpointer(self.cfg.TEST.EVAL_PERIOD,
                                          self.checkpointer, C.TRAIN["best_metric"],
                                          mode="max")
            hl.insert(-1, best)
        return hl


def parse_args():
    ap = argparse.ArgumentParser(description="Train M2F SWIN-B on E7 MAIN")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-gpus", type=int, default=1)
    return ap.parse_args()


def main(args):
    register_e7_main_datasets()
    cfg = build_config(resume=args.resume)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    log_dir = C.OUT_MAIN / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output=str(log_dir / "train.log"), distributed_rank=comm.get_rank())
    log = logging.getLogger(__name__)

    log.info("=" * 80)
    log.info("E7 Stage 6 - Train Mask2Former MAIN")
    log.info(f"COCO dir   : {C.COCO_MAIN_DIR}")
    log.info(f"Output dir : {C.OUT_MAIN}")
    log.info(f"Num classes: {len(C.MAIN_CLASSES)} {C.MAIN_CLASSES}")
    log.info(f"Input size : {cfg.INPUT.IMAGE_SIZE}  LSJ [{cfg.INPUT.MIN_SCALE},{cfg.INPUT.MAX_SCALE}]")
    log.info(f"Batch      : {cfg.SOLVER.IMS_PER_BATCH}  LR {cfg.SOLVER.BASE_LR}")
    log.info(f"Max iter   : {cfg.SOLVER.MAX_ITER}  steps {cfg.SOLVER.STEPS}")
    log.info(f"Eval period: {cfg.TEST.EVAL_PERIOD}  best={C.TRAIN['best_metric']}")
    log.info(f"Sampler    : {cfg.DATALOADER.SAMPLER_TRAIN} "
             f"(repeat_thresh {cfg.DATALOADER.REPEAT_THRESHOLD})")
    log.info(f"Loss wts   : {getattr(C, 'CLASS_LOSS_WEIGHTS', {})}")
    log.info(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)} "
                 f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log.info("=" * 80)

    if not args.resume and not C.PRETRAINED.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {C.PRETRAINED}")
    if not C.BASE_CONFIG.exists():
        raise FileNotFoundError(f"Base config not found: {C.BASE_CONFIG}")

    default_setup(cfg, args)
    trainer = E7MAINTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    # Per-class classification-loss weights -> recalibrate grade scores so that
    # plain highest-score post-processing recovers all grades. Injected directly
    # into the M2F criterion's empty_weight (no repo modification).
    weights = getattr(C, "CLASS_LOSS_WEIGHTS", {})
    if weights:
        m = trainer.model
        m = m.module if hasattr(m, "module") else m
        crit = getattr(m, "criterion", None)
        if crit is not None and hasattr(crit, "empty_weight"):
            with torch.no_grad():
                for cls, w in weights.items():
                    if cls in C.MAIN_ID:
                        crit.empty_weight[C.MAIN_ID[cls] - 1] = float(w)
            log.info(f"Applied per-class loss weights: {weights}")
            log.info(f"criterion.empty_weight = {crit.empty_weight.tolist()}")
        else:
            log.warning("criterion.empty_weight not found; loss weights NOT applied")

    log.info("Starting E7 MAIN training...")
    trainer.train()
    log.info("E7 MAIN training finished.")


if __name__ == "__main__":
    args = parse_args()
    launch(main, num_gpus_per_machine=args.num_gpus, num_machines=1,
           machine_rank=0, dist_url="auto", args=(args,))