#!/usr/bin/env bash
# Build the persona vectors (bf16 forward, fp32 averaging) from the merged judge files.
# Writes /data/coe9/artifacts/unpersona/vectors/Qwen2.5-7B-Instruct/{trait}_{response_avg,prompt_avg,prompt_last}_diff.pt
set -euo pipefail
cd /data/coe9/artifacts/unpersona/pv_run
export HF_HOME=/data/coe9/hf_cache OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
J=/data/coe9/artifacts/unpersona/judged_final/Qwen2.5-7B-Instruct
OUT=/data/coe9/artifacts/unpersona/vectors/Qwen2.5-7B-Instruct
for t in ${TRAITS:-evil sycophantic hallucinating}; do
  echo "=== $(date '+%F %T') $t"
  /data/coe9/envs/persona/bin/python generate_vec.py --model_name Qwen/Qwen2.5-7B-Instruct \
      --pos_path $J/${t}_pos_instruct.csv --neg_path $J/${t}_neg_instruct.csv --trait $t \
      --save_dir $OUT --dtype bf16 2>&1 | grep -v -E "it/s\]|s/it\]"
done
echo "=== $(date '+%F %T') done"
