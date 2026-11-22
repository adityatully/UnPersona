#!/usr/bin/env bash
# Task 4 known-answer check (paper Fig. 3): evil, layer 20, coef 2.0, steering_type response, eval questions,
# plus the unsteered baseline. Generation only (--judge False); judged afterwards.
set -euo pipefail
cd /data/coe9/artifacts/unpersona/pv_run
export HF_HOME=/data/coe9/hf_cache OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
unset OPENAI_BASE_URL; export OPENAI_API_KEY=unused
PY=/data/coe9/envs/persona/bin/python
M=Qwen/Qwen2.5-7B-Instruct
OUT=eval_persona_eval/Qwen2.5-7B-Instruct
V=/data/coe9/artifacts/unpersona/vectors/Qwen2.5-7B-Instruct/evil_response_avg_diff.pt
echo "=== $(date '+%F %T') steered: evil L20 coef 2.0 response"
$PY -m eval.eval_persona --model $M --trait evil --output_path $OUT/evil_steer_response_L20_c2.0.csv \
    --version eval --steering_type response --coef 2.0 --vector_path $V --layer 20 --judge False
echo "=== $(date '+%F %T') baseline (coef 0)"
$PY -m eval.eval_persona --model $M --trait evil --output_path $OUT/evil_baseline.csv --version eval --judge False
echo "=== $(date '+%F %T') done"
