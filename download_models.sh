#!/bin/bash
# Download the released SKINET-M2F models into models/.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://github.com/haekalkh/skinet/releases/download/v1.0"
mkdir -p "$HERE/models"
for f in skinet_m2f_structures.pth skinet_m2f_cortex.pth; do
    if [ -f "$HERE/models/$f" ]; then echo "already present: models/$f"; continue; fi
    echo "downloading $f"
    curl -L --fail -o "$HERE/models/$f.part" "$BASE/$f" && mv "$HERE/models/$f.part" "$HERE/models/$f"
done
( cd "$HERE/models" && sha256sum -c SHA256SUMS 2>/dev/null || shasum -a 256 -c SHA256SUMS )
