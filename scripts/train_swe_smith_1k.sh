#!/bin/bash

# Training script for SWE-Smith domain on a 2x A100 80GB box (the "$1k" plan).
#
# Differs from scripts/train_swe_smith_advisor.sh (8x H100) as follows:
#   - repo-relative paths instead of the hardcoded /advisor-models
#   - 2 GPUs, train batch 4 (paper: 8), 18 epochs (paper: 30, stock script: 40)
#   - eval off during training (the Tenacity test pass is run separately)
#   - checkpoint every 2 epochs and resume from the latest one (spot preemption)
#   - student is selectable via AGENT_MODEL (default Gemini 2.5 Flash-Lite)
# Unchanged: Qwen2.5-7B-Instruct advisor, GRPO group 8, LR 1e-6, KL on,
# max prompt 28672, max generation 4096, temperature 1.0. If 28672 OOMs on
# 2x A100, lower TRAIN_BATCH_SIZE, not the context.
#
# Usage:
#   AGENT_SERVER_URL=http://<worker-ip>:8081 EVAL_SERVER_URL=http://<worker-ip>:8082 \
#   AGENT_MODEL=gemini/gemini-2.5-flash-lite bash scripts/train_swe_smith_1k.sh [extra hydra overrides]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Required: agent and eval servers on the swe-worker VM
: "${AGENT_SERVER_URL:?Set AGENT_SERVER_URL, e.g. http://<worker-ip>:8081}"
: "${EVAL_SERVER_URL:?Set EVAL_SERVER_URL, e.g. http://<worker-ip>:8082}"
export AGENT_SERVER_URL EVAL_SERVER_URL

# Student (frozen, via LiteLLM). Plan options:
#   A gemini/gemini-2.5-flash-lite   B gemini/gemini-2.5-flash
#   C deepseek/deepseek-chat         D openai/gpt-4.1-mini
export AGENT_MODEL="${AGENT_MODEL:-gemini/gemini-2.5-flash-lite}"
export ADVISOR_MODELS_MODE="advisor"

export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/SkyRL/skyrl-train:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/swe_smith}"
REPO="${REPO:-jd__tenacity}"
NUM_GPUS="${NUM_GPUS:-2}"
EPOCHS="${EPOCHS:-18}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
# generator.max_turns: 1 (upstream default) caps all advice + observations after the
# prompt at max_generate_length (4,096) and, with zero_reward_on_non_stop, zeroes
# longer rollouts even when resolved. Larger values let only the 28,672 prompt cap bind.
MAX_TURNS="${MAX_TURNS:-1}"
LOGGER="${LOGGER:-wandb}"  # "console" to print to stdout

# ckpt_interval / hf_save_interval are in optimizer steps, not epochs.
# 56 train issues / batch 4 = 14 steps per epoch -> every 2 epochs = 28 steps.
NUM_TRAIN=$(grep -c . "$DATA_DIR/train_${REPO}.json")  # JSONL, one issue per line
STEPS_PER_EPOCH=$(( NUM_TRAIN / TRAIN_BATCH_SIZE ))
CKPT_INTERVAL="${CKPT_INTERVAL:-$(( 2 * STEPS_PER_EPOCH ))}"

STUDENT_TAG="$(echo "$AGENT_MODEL" | tr '/.' '__')"
RUN_NAME="${RUN_NAME:-swe_smith_${REPO}_${STUDENT_TAG}_2xa100}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/ckpts}"
EXPORT_ROOT="${EXPORT_ROOT:-$HOME/exports}"

echo "run=$RUN_NAME student=$AGENT_MODEL gpus=$NUM_GPUS epochs=$EPOCHS batch=$TRAIN_BATCH_SIZE max_turns=$MAX_TURNS"
echo "train=$NUM_TRAIN steps/epoch=$STEPS_PER_EPOCH ckpt_interval=$CKPT_INTERVAL"

cd "$REPO_ROOT"
"$REPO_ROOT/SkyRL/skyrl-train/.venv/bin/python" -m advisor_models.swe_smith.main_swe_smith \
    data.train_data="['$DATA_DIR/train_${REPO}.json']" \
    data.val_data="['$DATA_DIR/validation_${REPO}.json']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="Qwen/Qwen2.5-7B-Instruct" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    generator.num_inference_engines=$NUM_GPUS \
    generator.inference_engine_tensor_parallel_size=1 \
    trainer.epochs=$EPOCHS \
    trainer.eval_batch_size=49 \
    trainer.eval_before_train=false \
    trainer.eval_interval=-1 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=$TRAIN_BATCH_SIZE \
    trainer.policy_mini_batch_size=4 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.ckpt_interval=$CKPT_INTERVAL \
    trainer.max_ckpts_to_keep=2 \
    trainer.hf_save_interval=$CKPT_INTERVAL \
    trainer.max_prompt_length=28672 \
    generator.sampling_params.max_generate_length=4096 \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=0.999 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.async_engine=true \
    generator.batched=false \
    environment.env_class=swe_smith \
    generator.n_samples_per_prompt=8 \
    generator.gpu_memory_utilization=0.8 \
    trainer.logger="$LOGGER" \
    trainer.project_name="advisor_models" \
    trainer.run_name="$RUN_NAME" \
    trainer.resume_mode=latest \
    trainer.ckpt_path="$CKPT_ROOT/$RUN_NAME" \
    generator.zero_reward_on_non_stop=true \
    generator.max_turns=$MAX_TURNS \
    trainer.export_path="$EXPORT_ROOT/$RUN_NAME" \
    "$@"
