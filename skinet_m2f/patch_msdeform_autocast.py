"""
Narrow the precision guard in the pixel decoder.

Mask2Former disables autocast for the whole of forward_features to protect one
operation: MSDeformAttn, whose CUDA kernel is built without half support. The
guard is correct but far wider than the problem. Everything else inside that
function -- the encoder's feed-forward networks, the FPN convolutions, the mask
features head -- is ordinary matrix arithmetic that would take half precision
without complaint, and the feed-forward networks alone are roughly 70% of the
component's arithmetic.

The measured consequence: under autocast the backbone halves (97.9 to 49.0 ms)
and the transformer decoder gains a little (25.1 to 20.4), while the pixel
decoder does not move at all (39.2 to 38.9). Its share of the runtime therefore
rises from 23% to 33% -- the faster everything else gets, the more it dominates.

This rewrites a COPY of the file so the original tree keeps working:
  1. removes @autocast(enabled=False) from forward_features
  2. drops the .float() casts that decorator made necessary
  3. re-applies the guard around the self_attn call alone

Usage:
  python patch_msdeform_autocast.py --root /work/.../Mask2Former_fp16
  python patch_msdeform_autocast.py --root ... --revert
"""

import argparse
import re
import shutil
from pathlib import Path

REL = "mask2former/modeling/pixel_decoder/msdeformattn.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="the FORK of Mask2Former, not the shared tree")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    path = Path(args.root) / REL
    backup = path.with_suffix(".py.orig")
    if not path.exists():
        raise SystemExit("not found: %s" % path)

    if args.revert:
        if not backup.exists():
            raise SystemExit("no backup at %s" % backup)
        shutil.copy2(str(backup), str(path))
        print("restored %s" % path)
        return

    if "Mask2Former_fp16" not in str(path) and "merdeka" not in str(path):
        print("WARNING: %s does not look like a fork." % args.root)
        print("Patching the shared tree would change every other run. Stopping.")
        print("Copy it first:")
        print("  cp -r <shared>/Mask2Former <somewhere>/Mask2Former_fp16")
        raise SystemExit(2)

    src = path.read_text()
    if not backup.exists():
        shutil.copy2(str(path), str(backup))
        print("backup written to %s" % backup)

    changed = []

    # 1. the decorator on forward_features
    dec = re.search(r"\n([ \t]*)@autocast\(enabled=False\)\s*\n([ \t]*def forward_features)",
                    src)
    if dec:
        src = src[:dec.start()] + "\n" + dec.group(2) + src[dec.end():]
        changed.append("removed @autocast(enabled=False) from forward_features")
    else:
        print("note: no decorator found on forward_features -- already patched?")

    # 2. the casts it forced
    n_float = len(re.findall(r"features\[f\]\.float\(\)", src))
    if n_float:
        src = src.replace("features[f].float()", "features[f]")
        changed.append("dropped %d features[f].float() cast(s)" % n_float)

    # 3. put the guard back around the one operation that needs it
    m = re.search(r"\n([ \t]*)src2 = self\.self_attn\((.*?)\)\n", src, re.S)
    if m:
        ind, call_args = m.group(1), m.group(2)
        flat = " ".join(call_args.split())
        new = (
            "\n%s# The CUDA kernel behind MSDeformAttn is compiled without half\n"
            "%s# support, so this one call still has to run in float32. Guarding\n"
            "%s# it here rather than the whole of forward_features lets the feed\n"
            "%s# forward networks and the convolutions take mixed precision.\n"
            "%swith autocast(enabled=False):\n"
            "%s    src2 = self.self_attn(%s)\n"
            "%ssrc2 = src2.to(src.dtype)\n"
            % (ind, ind, ind, ind, ind, ind,
               flat.replace("self.with_pos_embed(src, pos)",
                            "self.with_pos_embed(src, pos).float()")
                   .replace(", src,", ", src.float(),"),
               ind))
        src = src[:m.start()] + new + src[m.end():]
        changed.append("wrapped the self_attn call in its own guard")
    else:
        print("note: self_attn call not matched -- check it by hand")

    if "from torch.cuda.amp import autocast" not in src and \
       "from torch.cuda.amp.autocast_mode import autocast" not in src:
        src = src.replace("import torch\n",
                          "import torch\nfrom torch.cuda.amp import autocast\n", 1)
        changed.append("added the autocast import")

    path.write_text(src)
    print("\npatched %s" % path)
    for c in changed:
        print("  - %s" % c)
    print("""
Now profile against the fork and compare the pixel decoder line, which sat at
38.9 ms unchanged by fp16:

  qsub -q gpu -v RUN=patient_100k,FP16=1,M2F_ROOT=%s,\\
    WORK_DIR=...,PIPELINE=... job_profile.sh

Then verify accuracy is untouched before claiming anything -- run stages 05 to
08 with the fork and check the macro F1 against 0.723. A speedup that changes
the predictions is not free.

  python patch_msdeform_autocast.py --root %s --revert   (to undo)
""" % (args.root, args.root))


if __name__ == "__main__":
    main()