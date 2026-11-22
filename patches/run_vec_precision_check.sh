#!/usr/bin/env bash
# Precision check: the evil vector from the same 50 random kept pairs, bf16 (fp32 averaging) vs fp32.
set -euo pipefail
cd /data/coe9/artifacts/unpersona/pv_run
export HF_HOME=/data/coe9/hf_cache OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
J=/data/coe9/artifacts/unpersona/judged_final/Qwen2.5-7B-Instruct
OUT=/data/coe9/artifacts/unpersona/vectors_check
for dt in bf16 fp32; do
  echo "=== $(date '+%F %T') $dt"
  /usr/bin/time -v /data/coe9/envs/persona/bin/python generate_vec.py --model_name Qwen/Qwen2.5-7B-Instruct \
      --pos_path $J/evil_pos_instruct.csv --neg_path $J/evil_neg_instruct.csv --trait evil \
      --save_dir $OUT/$dt --dtype $dt --max_pairs 50 2>&1 | grep -v -E "it/s\]|s/it\]" | grep -E "unpersona|saved|Elapsed|Maximum resident|Error|error|Traceback" 
done
echo "=== $(date '+%F %T') done"
