#!/bin/bash
# =============================================================================
#  vendor_scripts.sh -- copy YOUR remaining stage scripts into the package, so
#  `pip install .` ships a self-contained pipeline.
#
#  Run this ONCE in the package folder, pointing at your repo's scripts dir:
#
#      ./vendor_scripts.sh /work/lead/mk657917/SKINET/scripts/10_e7_clinical_pipeline
#
#  It copies every *.py from there into skinet/scripts/, EXCEPT the two files
#  that ship pre-optimized in this package (e7_run.py and *stitch_main_e7.py),
#  which are preserved so you always get the fast stitch + --sam-mode replace.
# =============================================================================
set -euo pipefail

SRC="${1:-}"
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  echo "usage: ./vendor_scripts.sh <path-to-your-scripts-dir>"
  echo "  e.g. ./vendor_scripts.sh /work/lead/mk657917/SKINET/scripts/10_e7_clinical_pipeline"
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/skinet/scripts"
mkdir -p "$DST"

[ -f "$SRC/e7_run.py" ] || { echo "ERROR: $SRC has no e7_run.py -- wrong dir?"; exit 1; }

echo "==> vendoring from: $SRC"
copied=0; skipped=0
shopt -s nullglob
for py in "$SRC"/*.py; do
  base="$(basename "$py")"
  # skip junk filenames containing spaces (e.g. an accidental "python 10_*.py")
  case "$base" in
    *" "*) echo "    skip (bad name): $base"; continue;;
  esac
  # preserve the pre-optimized versions shipped with this package
  case "$base" in
    e7_run.py|*stitch_main_e7.py)
      echo "    keep (pre-optimized): $base"; skipped=$((skipped+1)); continue;;
  esac
  cp "$py" "$DST/$base"; copied=$((copied+1))
done
shopt -u nullglob

# completeness marker so the CLI knows the packaged scripts are usable
touch "$DST/VENDORED"

n=$(find "$DST" -name '*.py' | wc -l | tr -d ' ')
echo "==> copied $copied, preserved $skipped pre-optimized"
echo "==> skinet/scripts now has $n python files"
echo ""
echo "Sanity check -- these must all be present for the pipeline to run:"
for need in e7_run.py e7_config.py tissue_filter.py; do
  [ -f "$DST/$need" ] && echo "    OK  $need" || echo "    !!  MISSING $need (copy it manually into $DST)"
done
for pat in "*infer_tiles_main_e7.py" "*stitch_main_e7.py" "*fuse_main_e7.py" "*filter_main_e7.py" "*refine_sam_e7.py"; do
  f=$(find "$DST" -name "$pat" | head -1)
  [ -n "$f" ] && echo "    OK  $(basename "$f")" || echo "    !!  MISSING a file matching $pat"
done
echo ""
echo "If refine_sam imports e7_sam_dataset / kidney_adapter, make sure those .py"
echo "are in $SRC too (they get vendored automatically if present)."
echo ""
echo "Next:  pip install .   (then: skinet doctor)"
