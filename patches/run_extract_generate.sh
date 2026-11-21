#!/usr/bin/env bash
# Task 3, step 1: generate extraction responses (no judging) for evil, sycophantic, hallucinating.
# 5 instruction pairs x 20 questions x 10 samples = 1000 responses per (trait, pos|neg), vLLM, bf16.
# Run inside tmux (session unpersona-gen). Output: pv_run/eval_persona_extract/Qwen2.5-7B-Instruct/
set -euo pipefail
cd /data/coe9/artifacts/unpersona/pv_run
export HF_HOME=/data/coe9/hf_cache TMPDIR=/data/coe9/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
       TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset OPENAI_BASE_URL              # no judge server during generation
export OPENAI_API_KEY=unused       # config.py requires the variable to exist
PY=/data/coe9/envs/persona/bin/python
MODEL=Qwen/Qwen2.5-7B-Instruct
OUT=eval_persona_extract/Qwen2.5-7B-Instruct
for trait in ${TRAITS:-evil sycophantic hallucinating}; do
  for pol in pos neg; do
    # README rule: pos -> the trait adjective; neg -> antonym if clear, else "helpful".
    # Only evil is given explicitly (generate_vec.sh); "helpful" for all negs is our choice.
    if [ "$pol" = pos ]; then name="$trait"; else name=helpful; fi
    echo "=== $(date '+%F %T') $trait $pol (assistant_name=$name)"
    $PY -m eval.eval_persona --model "$MODEL" --trait "$trait" \
        --output_path "$OUT/${trait}_${pol}_instruct.csv" \
        --persona_instruction_type "$pol" --assistant_name "$name" \
        --version extract --judge False
  done
done
echo "=== $(date '+%F %T') all done"
