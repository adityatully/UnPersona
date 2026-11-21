#!/usr/bin/env bash
# Build a patched, runnable copy of the paper's repo. third_party/ stays untouched.
#   bash patches/make_run_copy.sh [DST]      default DST = /data/coe9/artifacts/unpersona/pv_run
# Re-running refreshes the code from third_party/ and re-applies all patches; pipeline outputs
# (eval_persona_extract/, eval_persona_eval/, persona_vectors/) in DST are kept.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/third_party/persona_vectors"
DST="${1:-/data/coe9/artifacts/unpersona/pv_run}"
mkdir -p "$DST"
rsync -a --exclude .git --exclude dataset.zip --exclude eval_persona_extract \
      --exclude eval_persona_eval --exclude persona_vectors "$SRC/" "$DST/"
for p in "$REPO"/patches/*.patch; do
  echo "applying $(basename "$p")"
  patch -d "$DST" -p1 --no-backup-if-mismatch < "$p"
done
echo "run copy ready: $DST"
