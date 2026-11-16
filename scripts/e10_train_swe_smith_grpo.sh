#!/bin/bash
# E10 GRPO arm (G): E7's SWE-smith environment on upstream SkyRL 3e8700f, LoRA,
# one GB10 on the DGX Spark, local FP8 student. GRPO only: nothing here is for
# the PPO arms (P, P6).
#
# NOT AUTHORIZED TO RUN. As of amendment A2 (13 Nov 2026) no E10 training step,
# G1 or G2 is authorized. This script exists so the port's config can be reviewed.
# Settings marked G1/G2 are fixed by those gates, by amendment, before any run.
#
# Print the resolved overrides without starting anything (used by the tests):
#   E10_PRINT_OVERRIDES=1 bash scripts/e10_train_swe_smith_grpo.sh
#
# Settings from E10 spec §2-§3, mapped to upstream names (E7 name -> upstream):
#   trainer.strategy=fsdp2                        -> trainer.strategy=fsdp
#   generator.num_inference_engines               -> generator.inference_engine.num_engines
#   generator.inference_engine_tensor_parallel_size -> generator.inference_engine.tensor_parallel_size
#   generator.{backend,run_engines_locally,weight_sync_backend,gpu_memory_utilization}
#                                                 -> generator.inference_engine.*
#   generator.async_engine=true                   -> removed (upstream is always async)
#   LoRA (new in E10, spec §2)                    -> trainer.policy.model.lora.*
# Required on the Spark (spark/README.md): trainer.flash_attn=false,
# trainer.remove_microbatch_padding=false.
#
# zero_reward_on_non_stop: upstream's rule, kept as is (Scott, 13 Nov; recorded
# as a deviation from E7 in the spec). Exact condition at SkyRL 3e8700f,
# skyrl/train/generators/skyrl_gym_generator.py:
#   - agent_loop: before each advisor turn, if len(input_ids) > max_input_length,
#     stop_reason = "length" and the rollout ends. input_ids is the whole
#     conversation so far (prompt + every advisor turn + every observation), and
#     max_input_length defaults to trainer.max_prompt_length = 28672 (strictly
#     greater than: exactly 28,672 tokens still generates).
#   - otherwise stop_reason is what vLLM returned for the rollout's LAST advisor
#     generation: "stop", or "length" if that turn hit
#     sampling_params.max_generate_length = 4096 (or another reason, e.g. "abort").
#   - generate(): with zero_reward_on_non_stop=true, _zero_reward_if_not_stop sets
#     the reward to 0 (every per-token reward) for every rollout whose
#     stop_reason != "stop".
# E7's fork differed in two ways (advisor-models SkyRL/skyrl-train,
# skyrl_gym_generator.py): (1) it checked len(input_ids) > max_input_length
# after every env step, including the last one, so a rollout the env had
# already ended (done=True) could still be marked "length"; upstream checks only
# before another turn, so a finished rollout keeps its last generation's stop
# reason. (2) After the loop it also set "length" when the total response (all
# turns and observations) exceeded max_generate_length + max_input_length -
# prompt length; upstream has no such check. The per-step counts of "length", other
# non-"stop" and zeroed rollouts are logged by generator_upstream.py
# (generate/e10_rollouts_* and an "E10 rollouts: ..." line).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/swe_smith}"
REPO="${REPO:-jd__tenacity}"
RUN_NAME="${RUN_NAME:-exp-10_arm-grpolora_split-tenacitytrain_seed-0}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/e10-work/ckpts}"
EXPORT_ROOT="${EXPORT_ROOT:-$HOME/e10-work/exports}"
LOGGER="${LOGGER:-console}"
# G2 fixes the step count (295 or 150) by amendment; 56 issues / batch 4 = 14 steps per epoch.
EPOCHS="${EPOCHS:-TBD_BY_G2}"
# G1 fixes the advisor vLLM's memory share next to the student (0.40) and the trainer.
ADVISOR_VLLM_MEM="${ADVISOR_VLLM_MEM:-TBD_BY_G1}"
CKPT_INTERVAL="${CKPT_INTERVAL:-28}"

OVERRIDES=(
    data.train_data="['$DATA_DIR/train_${REPO}.json']"
    data.val_data="['$DATA_DIR/validation_${REPO}.json']"
    environment.env_class=swe_smith
    # Algorithm (spec §3, arm G)
    trainer.algorithm.advantage_estimator=grpo
    trainer.algorithm.use_kl_loss=true
    trainer.algorithm.kl_loss_coef=0.001
    trainer.policy.optimizer_config.lr=1.0e-5
    trainer.update_epochs_per_batch=1
    trainer.train_batch_size=4
    trainer.policy_mini_batch_size=4
    generator.n_samples_per_prompt=8
    trainer.seed=0
    # Advisor and LoRA (spec §2)
    trainer.policy.model.path=Qwen/Qwen2.5-7B-Instruct
    trainer.policy.model.lora.rank=32
    trainer.policy.model.lora.alpha=32
    trainer.policy.model.lora.dropout=0.0
    trainer.policy.model.lora.target_modules=all-linear
    # Lengths and sampling (E7's limits, spec §3)
    trainer.max_prompt_length=28672
    generator.sampling_params.max_generate_length=4096
    generator.sampling_params.temperature=1.0
    generator.sampling_params.top_p=0.999
    generator.max_turns=9
    generator.batched=false
    generator.zero_reward_on_non_stop=true
    # One GB10, colocated
    trainer.strategy=fsdp
    trainer.placement.colocate_all=true
    trainer.placement.policy_num_gpus_per_node=1
    trainer.placement.ref_num_gpus_per_node=1
    trainer.micro_forward_batch_size_per_gpu=1
    trainer.micro_train_batch_size_per_gpu=1
    generator.inference_engine.backend=vllm
    generator.inference_engine.run_engines_locally=true
    generator.inference_engine.num_engines=1
    generator.inference_engine.tensor_parallel_size=1
    generator.inference_engine.weight_sync_backend=nccl
    generator.inference_engine.gpu_memory_utilization=$ADVISOR_VLLM_MEM
    # Required on the Spark: no flash-attn kernels for the GB10
    trainer.flash_attn=false
    trainer.remove_microbatch_padding=false
    # Schedule, checkpoints, logging
    trainer.epochs=$EPOCHS
    trainer.eval_before_train=false
    trainer.eval_interval=-1
    trainer.eval_batch_size=49
    trainer.ckpt_interval=$CKPT_INTERVAL
    trainer.hf_save_interval=$CKPT_INTERVAL
    trainer.max_ckpts_to_keep=2
    trainer.resume_mode=latest
    trainer.ckpt_path="$CKPT_ROOT/$RUN_NAME"
    trainer.export_path="$EXPORT_ROOT/$RUN_NAME"
    trainer.logger="$LOGGER"
    trainer.project_name=e10
    trainer.run_name="$RUN_NAME"
)

if [ "${E10_PRINT_OVERRIDES:-0}" = "1" ]; then
    printf '%s\n' "${OVERRIDES[@]}"
    exit 0
fi

echo "E10 STOP: no amendment authorizes an E10 training run yet (A2, 13 Nov 2026)." >&2
echo "Remove this guard only in the commit that records the authorizing amendment." >&2
exit 1

# Environment the run will need (agent and eval servers on the Spark, local student):
#   AGENT_SERVER_URL=http://localhost:8081 EVAL_SERVER_URL=http://localhost:8082
#   AGENT_MODEL=hosted_vllm/student HOSTED_VLLM_API_BASE=http://localhost:8100/v1
#   LITELLM_MODEL_REGISTRY_PATH=$REPO_ROOT/configs/litellm_local_student_registry.json
#   SWESMITH_IMAGE_OVERRIDE=hecate/swesmith.arm64.jd_1776_tenacity.0d40e76f:e10
cd "$REPO_ROOT"
python -m advisor_models.swe_smith.main_swe_smith_upstream "${OVERRIDES[@]}" "$@"
