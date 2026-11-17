"""
What dtype does the deformable-attention kernel actually receive?

The pixel decoder is the one component that gains nothing from FP16 (39.2 ms
against 38.9), and the source rules out the easy explanations: MSDeformAttnFunction
calls the CUDA kernel directly with no fallback, and there is no custom_fwd, no
cast_inputs and no .float() anywhere in the module. Two possibilities remain, and
one line of output separates them.

  value arrives as float32  -> something upstream is holding the precision back,
                               and that cast is findable and removable
  value arrives as float16  -> the kernel does run in half and still gains
                               nothing, so its cost is bound by irregular memory
                               access rather than arithmetic. Precision cannot
                               lower it, and only fewer layers or fewer tokens can.

The second is the more interesting result: an operation at the centre of
Mask2Former whose cost is irreducible by precision, which holds for anyone
running it, not just this dataset.

Usage:
  python probe_msdeform.py --run baseline_seed42          # fp32 reference
  python probe_msdeform.py --run baseline_seed42 --fp16   # the one that matters
"""

import argparse
import contextlib
from collections import OrderedDict
from pathlib import Path

import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config

import config as C


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--run", required=True)
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base-yaml", default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    args = ap.parse_args()

    work = Path(args.work_dir)
    if args.weights is None:
        args.weights = str(work / "s04_train" / args.run / "model_best.pth")
    if args.base_yaml is None:
        p = Path(C.BASE_YAML)
        args.base_yaml = str(p if p.is_absolute() else here / p)

    from mask2former import add_maskformer2_config
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.base_yaml)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(C.STRUCTURE_CLASSES)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.INPUT.MIN_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Wrap the Function's forward. This is where the kernel is actually entered,
    # so whatever dtype shows up here is what the CUDA code has to handle.
    from mask2former.modeling.pixel_decoder.ops.functions import ms_deform_attn_func as F
    seen = OrderedDict()
    original = F.MSDeformAttnFunction.forward

    @staticmethod
    def probed(ctx, value, shapes, level_start, sampling_locations,
               attention_weights, im2col_step):
        if not seen:
            seen["value"] = str(value.dtype)
            seen["sampling_locations"] = str(sampling_locations.dtype)
            seen["attention_weights"] = str(attention_weights.dtype)
            seen["value_shape"] = tuple(value.shape)
        return original.__func__(ctx, value, shapes, level_start,
                                 sampling_locations, attention_weights,
                                 im2col_step)

    F.MSDeformAttnFunction.forward = probed

    dev = torch.device(cfg.MODEL.DEVICE)
    img = torch.randint(0, 255, (3, args.tile_size, args.tile_size),
                        dtype=torch.uint8).to(dev).to(torch.float32)
    batch = [{"image": img, "height": args.tile_size, "width": args.tile_size}]
    amp = (torch.autocast("cuda", dtype=torch.float16)
           if (args.fp16 and dev.type == "cuda") else contextlib.nullcontext())

    with torch.no_grad(), amp:
        model(batch)

    F.MSDeformAttnFunction.forward = original

    print("=" * 68)
    print("MSDeformAttn kernel inputs | run %s | %s"
          % (args.run, "autocast fp16" if args.fp16 else "fp32"))
    print("=" * 68)
    for k, v in seen.items():
        print("   %-22s%s" % (k, v))
    print("-" * 68)
    if not seen:
        print("the kernel was never reached -- the pixel decoder may not be")
        print("using deformable attention in this configuration")
    elif seen["value"] == "torch.float16":
        print("The kernel runs in half and still gains nothing, so its cost is")
        print("bound by irregular memory access rather than arithmetic.")
        print("Precision cannot lower it; only fewer layers, fewer sampling")
        print("points or fewer tokens can.")
    else:
        print("The kernel receives float32 under autocast, so the precision is")
        print("being held back upstream. Find that cast and the pixel decoder")
        print("should pick up the speedup the other components already get.")
    print("=" * 68)


if __name__ == "__main__":
    main()