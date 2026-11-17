"""
Where does the cost actually live?

Before cutting anything, measure. Mask2Former has three large parts -- the Swin
backbone, the deformable-attention pixel decoder, and the transformer decoder --
and which one dominates decides the whole compression plan. Trimming queries is
wasted effort if 80% of the time sits in the backbone.

Reports, for one tile-sized forward pass at inference settings:
  parameters per component, and their share
  wall time per component, and their share (pixel_decoder is timed through its
  forward_features method, which a plain forward hook never sees)
  peak activation memory per component, and weight memory per component
  total peak GPU memory
  throughput in tiles per second, and the implied seconds per ROI

Time and memory do not tell the same story, so both are reported: the pixel
decoder is 5% of the weights but a third of the time. Measuring only one of them
points at the wrong component.

Whether they should agree is decided by arithmetic intensity. A component that
is compute-bound spends its time on arithmetic, so time tracks operation count
and precision helps; one that is bandwidth-bound spends it moving data, so time
tracks memory traffic and precision helps much less. --flops reports achieved
TFLOPS per component, which separates the two: against a V100's ~14 TFLOPS FP32
peak, 30% of peak is compute-bound and 3% is bandwidth-bound.

--probe-dtype is the companion diagnostic. MSDeformAttn is a custom autograd
Function, and custom Functions are not autocast-aware unless declared so. If the
pixel decoder turns out to be running float32 under autocast, its refusal to
speed up is a fixable software fact rather than a property of the operation.

Run it once per configuration and the numbers line up into the cost axis of the
trade-off curve.

Usage:
  python profile_cost.py --run baseline_seed42
  python profile_cost.py --run baseline_seed42 --fp16
  python profile_cost.py --run mrcnn_seed42 --arch maskrcnn
"""

import argparse
import contextlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2 import model_zoo

import config as C


def build_cfg(weights, base_yaml, arch, num_queries=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    if arch == "mask2former":
        from mask2former import add_maskformer2_config
        add_maskformer2_config(cfg)
    cfg.merge_from_file(base_yaml)
    n = len(C.STRUCTURE_CLASSES)
    if arch == "mask2former":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "bitmask"
        if num_queries:
            cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = num_queries
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = n
        cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.MODEL.WEIGHTS = weights
    cfg.INPUT.MIN_SIZE_TEST = C.TRAIN["test_size"]
    cfg.INPUT.MAX_SIZE_TEST = C.TRAIN["test_size"]
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg


def component_map(model, arch):
    """Top-level parts worth reporting separately."""
    if arch == "mask2former":
        return OrderedDict([
            ("backbone", model.backbone),
            ("pixel_decoder", model.sem_seg_head.pixel_decoder),
            ("transformer_decoder", model.sem_seg_head.predictor),
        ])
    return OrderedDict([
        ("backbone", model.backbone),
        ("proposal_generator", model.proposal_generator),
        ("roi_heads", model.roi_heads),
    ])


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--run", required=True)
    ap.add_argument("--arch", default="mask2former", choices=sorted(C.ARCH.keys()))
    ap.add_argument("--work-dir", default=C.WORK_DIR)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--base-yaml", default=None)
    ap.add_argument("--num-queries", type=int, default=None)
    ap.add_argument("--fp16", action="store_true",
                    help="half precision at inference; costs nothing to try and "
                         "is the first thing anyone does for an edge device")
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--tiles-per-roi", type=int, default=18)
    ap.add_argument("--flops", action="store_true",
                    help="count operations per component and divide by the "
                         "measured time, giving achieved TFLOPS")
    ap.add_argument("--probe-dtype", action="store_true",
                    help="report the dtype actually flowing through each "
                         "component during the forward pass")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    work = Path(args.work_dir)
    spec = C.ARCH[args.arch]
    if args.weights is None:
        args.weights = str(work / "s04_train" / args.run / "model_best.pth")
    if args.base_yaml is None:
        if spec["from_model_zoo"]:
            args.base_yaml = model_zoo.get_config_file(spec["yaml"])
        else:
            p = Path(spec["yaml"] or C.BASE_YAML)
            args.base_yaml = str(p if p.is_absolute() else here / p)

    cfg = build_cfg(args.weights, args.base_yaml, args.arch, args.num_queries)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    # NOT model.half(): this Swin implementation builds the shifted-window
    # attention mask inside forward with torch.zeros, which stays float32 and
    # promotes the attention back to float32, then meets half weights. autocast
    # handles the mixed dtypes, and it is what real deployment uses anyway.

    dev = torch.device(cfg.MODEL.DEVICE)
    parts = component_map(model, args.arch)
    total_params = count_params(model)

    # time each component with hooks; CUDA is async so synchronise around them
    timings = dict((k, []) for k in parts)
    act_peak = dict((k, []) for k in parts)   # peak allocated during the part
    retained = dict((k, []) for k in parts)   # still held when it returns
    handles = []
    measure_mem = {"on": False}

    def enter(name, mod):
        if dev.type == "cuda":
            torch.cuda.synchronize()
            if measure_mem["on"]:
                mod._m0 = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
        mod._t0 = time.perf_counter()

    def leave(name, mod):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        timings[name].append(time.perf_counter() - mod._t0)
        if dev.type == "cuda" and measure_mem["on"]:
            act_peak[name].append(torch.cuda.max_memory_allocated() / 1024 ** 2)
            retained[name].append(
                (torch.cuda.memory_allocated() - mod._m0) / 1024 ** 2)

    def make_pre(name):
        def hook(mod, inp):
            enter(name, mod)
        return hook

    def make_post(name):
        def hook(mod, inp, out):
            leave(name, mod)
        return hook

    for name, mod in parts.items():
        handles.append(mod.register_forward_pre_hook(make_pre(name)))
        handles.append(mod.register_forward_hook(make_post(name)))

    # Mask2Former calls pixel_decoder.forward_features(), not forward(), so a
    # forward hook never fires and its cost lands silently in "other". Wrap the
    # method instead.
    restore = None
    if args.arch == "mask2former":
        pd = parts["pixel_decoder"]
        original = pd.forward_features

        def timed_forward_features(*a, **kw):
            enter("pixel_decoder", pd)
            out = original(*a, **kw)
            leave("pixel_decoder", pd)
            return out

        pd.forward_features = timed_forward_features
        restore = (pd, original)

    img = torch.randint(0, 255, (3, args.tile_size, args.tile_size),
                        dtype=torch.uint8).to(dev).to(torch.float32)
    batch = [{"image": img, "height": args.tile_size, "width": args.tile_size}]

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=" * 74)
    print("cost profile | run %s  arch %s  tile %d  %s"
          % (args.run, args.arch, args.tile_size, "fp16" if args.fp16 else "fp32"))
    print("=" * 74)

    amp = (torch.autocast("cuda", dtype=torch.float16)
           if (args.fp16 and dev.type == "cuda") else contextlib.nullcontext())

    with torch.no_grad(), amp:
        for _ in range(args.warmup):
            model(batch)
        for k in timings:
            timings[k] = []
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model(batch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - t0) / args.iters

        # Per-component memory needs reset_peak_memory_stats inside the forward
        # pass, which destroys the global peak. So it gets its own pass, run
        # after the timed one.
        peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2
                   if dev.type == "cuda" else float("nan"))
        measure_mem["on"] = True
        for _ in range(3):
            model(batch)
        measure_mem["on"] = False

    for h in handles:
        h.remove()
    if restore is not None:
        restore[0].forward_features = restore[1]

    bytes_per_param = 2 if args.fp16 else 4
    print("%-21s%12s%7s%10s%7s%10s%10s"
          % ("component", "params", "share", "ms/tile", "share",
             "weights", "act peak"))
    print("-" * 78)
    accounted_t = 0.0
    rows = OrderedDict()
    for name, mod in parts.items():
        p = count_params(mod)
        t = float(np.median(timings[name])) if timings[name] else 0.0
        a = float(np.median(act_peak[name])) if act_peak[name] else 0.0
        r = float(np.median(retained[name])) if retained[name] else 0.0
        accounted_t += t
        w = p * bytes_per_param / 1024 ** 2
        rows[name] = {"params": p, "ms": round(t * 1000, 2),
                      "weights_mb": round(w, 1),
                      "act_peak_mb": round(a, 1), "retained_mb": round(r, 1)}
        print("%-21s%12s%6.1f%%%10.1f%6.1f%%%9.0fM%9.0fM"
              % (name, "{:,}".format(p), 100.0 * p / total_params,
                 t * 1000, 100.0 * t / total_time, w, a))
    other_p = total_params - sum(r["params"] for r in rows.values())
    other_t = total_time - accounted_t
    print("%-21s%12s%6.1f%%%10.1f%6.1f%%"
          % ("other / overhead", "{:,}".format(other_p),
             100.0 * other_p / total_params, other_t * 1000,
             100.0 * other_t / total_time))
    print("-" * 78)
    print("%-21s%12s%7s%10.1f" % ("TOTAL", "{:,}".format(total_params), "",
                                  total_time * 1000))
    if args.flops:
        try:
            from detectron2.utils.analysis import FlopCountAnalysis
            fa = FlopCountAnalysis(model, batch)
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            total_flops = fa.total()
            by_mod = fa.by_module()
            if total_flops == 0:
                # A silent zero is worse than an error. Say what happened.
                print("\nflop counting returned nothing. Most likely the model "
                      "could not be traced")
                print("in inference mode, where Mask2Former's post-processing "
                      "has data-dependent")
                print("control flow. Unsupported ops seen:")
                unsup = fa.unsupported_ops()
                for op, n in sorted(unsup.items(), key=lambda kv: -kv[1])[:10]:
                    print("   %-45s%8d" % (op, n))
                raise RuntimeError("nothing counted")
            present = [k for k in by_mod if k and "." not in k]
            print("\ntop-level modules seen by the counter: %s"
                  % ", ".join(sorted(present)))
            peak = 112.0 if args.fp16 else 14.0   # V100 TFLOPS, rough
            print()
            print("%-21s%14s%12s%12s" % ("component", "GFLOPs", "TFLOPS", "of peak"))
            print("-" * 78)
            name_of = {"backbone": "backbone",
                       "pixel_decoder": "sem_seg_head.pixel_decoder",
                       "transformer_decoder": "sem_seg_head.predictor",
                       "proposal_generator": "proposal_generator",
                       "roi_heads": "roi_heads"}
            for name in parts:
                key = name_of.get(name, name)
                g = 2.0 * by_mod.get(key, 0) / 1e9      # fvcore counts MACs
                t = float(np.median(timings[name])) if timings[name] else 0.0
                ach = (g / 1e3) / t if t > 0 else 0.0
                rows[name]["gflops"] = round(g, 1)
                rows[name]["tflops_achieved"] = round(ach, 2)
                print("%-21s%14.1f%12.2f%11.0f%%"
                      % (name, g, ach, 100.0 * ach / peak))
            print("-" * 78)
            print("Counted as 2 x MACs. Custom CUDA ops such as MSDeformAttn are")
            print("invisible to the counter, so the pixel decoder is undercounted")
            print("and its true achieved rate is higher than shown.")
            print("Rule of thumb against a V100 (%.0f TFLOPS peak): near 30%% means"
                  % peak)
            print("compute-bound and precision will help; a few percent means")
            print("bandwidth-bound and it will not.")
        except Exception as e:
            print("\nflop counting unavailable: %s" % e)

    if args.probe_dtype and dev.type == "cuda":
        seen = {}

        def probe(name):
            def hook(mod, inp, out):
                for t in (inp if isinstance(inp, tuple) else [inp]):
                    if torch.is_tensor(t) and t.is_floating_point():
                        seen.setdefault(name, str(t.dtype))
                        return
            return hook

        hs = [m.register_forward_hook(probe(n)) for n, m in parts.items()]

        # Same trap as the timing: pixel_decoder is entered through
        # forward_features, which no forward hook ever sees.
        pd_restore = None
        if args.arch == "mask2former":
            pdm = parts["pixel_decoder"]
            orig_ff = pdm.forward_features

            def probed_ff(*a, **kw):
                for t in a:
                    if torch.is_tensor(t) and t.is_floating_point():
                        seen.setdefault("pixel_decoder", str(t.dtype))
                        break
                    if isinstance(t, dict):
                        for v in t.values():
                            if torch.is_tensor(v) and v.is_floating_point():
                                seen.setdefault("pixel_decoder", str(v.dtype))
                                break
                        break
                out = orig_ff(*a, **kw)
                if "pixel_decoder_out" not in seen:
                    for v in (out if isinstance(out, (tuple, list)) else [out]):
                        if torch.is_tensor(v) and v.is_floating_point():
                            seen["pixel_decoder_out"] = str(v.dtype)
                            break
                return out

            pdm.forward_features = probed_ff
            pd_restore = (pdm, orig_ff)

        with torch.no_grad(), (torch.autocast("cuda", dtype=torch.float16)
                               if args.fp16 else contextlib.nullcontext()):
            model(batch)
        for h in hs:
            h.remove()
        if pd_restore is not None:
            pd_restore[0].forward_features = pd_restore[1]
        print()
        print("dtype entering each component%s:"
              % (" under autocast" if args.fp16 else ""))
        for n in parts:
            print("   %-21s%s" % (n, seen.get(n, "unknown")))
        if "pixel_decoder_out" in seen:
            print("   %-21s%s" % ("pixel_decoder OUT", seen["pixel_decoder_out"]))
        if args.fp16:
            print("Anything still float32 here is not benefiting from mixed")
            print("precision, and is worth checking for a custom op that opts out.")

    print()
    print("act peak is the highest allocation reached while that component ran,")
    print("so the parts overlap and do not sum to the total.")
    print()
    print("total peak GPU memory%8.0f MB" % peak_mb)
    print("throughput           %8.1f tiles/s" % (1.0 / total_time))
    print("per ROI (%d tiles)    %8.1f s" % (args.tiles_per_roi,
                                             args.tiles_per_roi * total_time))
    print()
    print("Jetson memory budgets for reference: Orin Nano 8 GB, Xavier NX 8 GB,")
    print("AGX Orin 32-64 GB. Peak above is inference only, one tile, no batching.")
    print("=" * 74)

    report = OrderedDict(run=args.run, arch=args.arch, fp16=bool(args.fp16),
                         tile_size=args.tile_size,
                         num_queries=args.num_queries,
                         total_params=int(total_params),
                         components=rows,
                         ms_per_tile=round(total_time * 1000, 2),
                         peak_memory_mb=round(float(peak_mb), 1),
                         seconds_per_roi=round(args.tiles_per_roi * total_time, 2))
    out = args.out or str(work / "profile" / ("%s%s.json" % (
        args.run, "_fp16" if args.fp16 else "")))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()