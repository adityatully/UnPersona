#!/usr/bin/env bash
# Start the local judge (vLLM OpenAI-compatible server) in env gen, on 127.0.0.1 only.
#   bash patches/serve_judge.sh [MODEL] [PORT] [GPU_MEM_FRACTION]
# Run inside tmux (session name: unpersona-judge). Stop it with: tmux kill-session -t unpersona-judge
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
PORT="${2:-8000}"
MEM="${3:-0.6}"   # fraction of the 24 GB card; GROMACS jobs need ~0.6 GB, leave room
export HF_HOME=/data/coe9/hf_cache TMPDIR=/data/coe9/tmp OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
exec /data/coe9/envs/gen/bin/vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$MEM" --max-model-len 4096 --max-num-seqs 32 \
  --max-logprobs 20 --seed 0 --disable-log-requests
