#!/usr/bin/env bash
# Judge the extraction responses with the local judge server (start it first with serve_judge.sh).
# Usage: bash patches/run_judge_extract.sh [JUDGE_MODEL] [SCORING]
# Reads  pv_run/eval_persona_extract/Qwen2.5-7B-Instruct/{trait}_{pos,neg}_instruct.csv (unjudged)
# Writes /data/coe9/artifacts/unpersona/judged_extract/Qwen2.5-7B-Instruct/ same names, plus columns
#        <trait> and coherence (the names generate_vec.py filters on). Existing outputs are skipped.
set -euo pipefail
JUDGE_MODEL=${1:-Qwen/Qwen2.5-14B-Instruct-AWQ}
SCORING=${2:-text}
TRAITS=${TRAITS:-"evil sycophantic hallucinating"}
SRC=/data/coe9/artifacts/unpersona/pv_run/eval_persona_extract/Qwen2.5-7B-Instruct
DST=/data/coe9/artifacts/unpersona/judged_extract/Qwen2.5-7B-Instruct
PY=/data/coe9/envs/persona/bin/python
mkdir -p "$DST"
cd /data/coe9/unpersona
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=local-judge

echo "waiting for judge server..."
until curl -sf http://localhost:8000/v1/models >/dev/null; do sleep 10; done
echo "server up: $(curl -s http://localhost:8000/v1/models | head -c 200)"

for t in $TRAITS; do
  for pol in pos neg; do
    f=${t}_${pol}_instruct.csv
    if [ -f "$DST/$f" ]; then echo "skip $f (exists)"; continue; fi
    echo "=== $(date '+%F %T') $f judge=$JUDGE_MODEL scoring=$SCORING"
    $PY -m unpersona.judge "$SRC/$f" --trait "$t" --judge_model "$JUDGE_MODEL" \
        --scoring "$SCORING" --version extract --out "$DST/$f.tmp"
    mv "$DST/$f.tmp" "$DST/$f"
  done
done
echo "=== $(date '+%F %T') all done"
