#!/usr/bin/env bash
# E10 arm U: serve the untrained advisor (Qwen2.5-7B-Instruct, bf16) on the Spark
# next to the student (serve_student.sh, port 8100, 0.40 of GPU memory), on port
# 8101 as model "advisor", for advisor_models/swe_smith/eval_e10.py --arm untrained.
#
# Same image and weights volume as the student. --max-model-len 32768 is E7's
# eval context (eval_advisor.py). ADVISOR_EVAL_MEM is the advisor's share of GPU
# memory; it has no default because U has not been measured on the Spark yet:
# the U memory check (student + this server + one U session, peak `free -h`
# every 10 s) fixes it, by amendment, before any U pass.
#
#   ADVISOR_EVAL_MEM=0.25 tmux new -d -s e10-advisor \
#     'scripts/e10_serve_advisor.sh > ~/e10-logs/advisor.log 2>&1'
#   curl -s localhost:8101/v1/models
#   docker stop e10-advisor        # when done
set -euo pipefail

if ! [[ "${ADVISOR_EVAL_MEM:-}" =~ ^0\.[0-9]+$ ]]; then
    echo "E10 STOP: set ADVISOR_EVAL_MEM (the advisor vLLM's GPU memory share, e.g. 0.25)." >&2
    exit 1
fi

exec docker run --rm --name e10-advisor --gpus all --ipc=host -p 127.0.0.1:8101:8000 \
  -v e10-hf:/root/.cache/huggingface \
  vllm/vllm-openai:v0.20.0 \
  Qwen/Qwen2.5-7B-Instruct \
  --served-model-name advisor --max-model-len 32768 \
  --gpu-memory-utilization "$ADVISOR_EVAL_MEM" --seed 0
