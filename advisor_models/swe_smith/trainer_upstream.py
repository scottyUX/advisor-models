"""E10 trainer: upstream SkyRL's RayPPOTrainer plus saved trajectories (spec §6).

Training is upstream's except for the reference model, plus two additions
for bookkeeping:

  reference model       spec §2: the reference is the policy's base with its
                        LoRA adapter disabled. Upstream builds a separate
                        frozen copy instead (about 17.6 GB for Qwen2.5-7B),
                        and its "offload to CPU" frees nothing on the GB10's
                        unified memory, so G1 ran out of memory (16 Nov). E10
                        builds no reference actors; the policy worker computes
                        the reference log-probs under peft's
                        ``disable_adapter()``. Same weights, same numbers: at
                        step 1, before any update, the LoRA output is zero and
                        the two log-probs must be equal.
  E10RayPPOTrainer      keeps CPU copies of the per-token log-probs, values,
                        advantages and returns right after
                        ``compute_advantages_and_returns``. Upstream later
                        rescales ``advantages`` in place for its loss
                        reduction, so the copy is what the estimator gave.
  E10TrajectoryWriter   on ``on_step_end``, joins that copy with the
                        generator's and the env's per-rollout data and writes
                        ``<run_dir>/trajectories/step-<NNNN>.jsonl.gz`` plus
                        the students' session logs and test reports.

A step whose records cannot be written fails the run: a run without its
trajectories is incomplete (spec §6, G3).

Environment variables:
  E10_RUN_DIR         run directory (default: trainer.export_path)
  E10_RUN_ID          run id in every record (default: trainer.run_name)
  E10_AGENT_LOG_DIR   the agent server's --log-dir, to copy session logs from
                      (unset: the records say where the logs would be, and the
                      step's log line counts them as missing)
"""

import contextlib
import os
from pathlib import Path

from loguru import logger
from skyrl.train import trainer as upstream_trainer
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.callbacks import TrainingCallback

from . import trajectories


class _NoReferenceWorker:
    """Stands in for upstream's RefWorker so build_models creates no reference actors."""


@contextlib.contextmanager
def _without_reference_actors():
    """While upstream's build_models runs, the reference actor group comes back as None.

    Upstream then treats the run as having no reference model: it initializes
    and offloads none, and gives the dispatcher none.
    """
    group_class = upstream_trainer.PPORayActorGroup

    def make_group(*args, **kwargs):
        worker = args[3] if len(args) > 3 else kwargs.get("ray_actor_type")
        return None if worker is _NoReferenceWorker else group_class(*args, **kwargs)

    upstream_trainer.PPORayActorGroup = make_group
    try:
        yield
    finally:
        upstream_trainer.PPORayActorGroup = group_class


def adapter_off_policy_worker():
    """Upstream's FSDP policy worker, with ``forward(..., reference=True)`` run with the adapter off."""
    import ray
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    class E10FSDPPolicyWorker(FSDPPolicyWorkerBase):
        def init_model(self, *args, **kwargs):
            # The policy model lives in this worker process, so the lean forward
            # (no full logits, no padding; spark_oom.py) is installed here.
            from .spark_oom import install_lean_forward

            if install_lean_forward():
                from loguru import logger as worker_logger

                worker_logger.info("E10 Spark: lean policy forward (chunked LM head, unpadded rows)")
            return super().init_model(*args, **kwargs)

        def forward(self, data, loss_fn=None, loss_fn_config=None, return_per_token_outputs=True,
                    reference=False, **extra):
            # **extra passes through what Ray adds to actor methods (e.g. _ray_trace_ctx).
            kwargs = dict(loss_fn=loss_fn, loss_fn_config=loss_fn_config,
                          return_per_token_outputs=return_per_token_outputs, **extra)
            if not reference:
                return super().forward(data, **kwargs)
            peft_model = self.model.model  # HFModelWrapper -> PeftModel when LoRA is on
            if not hasattr(peft_model, "disable_adapter"):
                raise RuntimeError("E10 reference log-probs need a LoRA (peft) policy")
            with peft_model.disable_adapter():
                return super().forward(data, **kwargs)

    return ray.remote(num_gpus=1)(E10FSDPPolicyWorker)


class E10RayPPOTrainer(RayPPOTrainer):
    e10_snapshot = None

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        algorithm = self.cfg.trainer.algorithm
        if self.cfg.trainer.strategy != "fsdp" or self.cfg.trainer.policy.model.lora.rank <= 0:
            raise ValueError("E10's adapter-off reference needs trainer.strategy=fsdp and a LoRA policy")
        if algorithm.use_kl_in_reward or self.cfg.trainer.update_ref_every_epoch:
            raise ValueError("E10's adapter-off reference supports the KL loss only, with a fixed reference")
        with _without_reference_actors():
            super().build_models(adapter_off_policy_worker(), CriticWorker, _NoReferenceWorker)
        if self.ref_model is not None:
            raise RuntimeError("E10: a reference actor group was built; expected none")

    def fwd_logprobs_values_reward(self, training_input):
        training_input = super().fwd_logprobs_values_reward(training_input)
        if not self.cfg.trainer.algorithm.use_kl_loss:
            return training_input
        # Same inputs as upstream's reference forward: one full-batch forward pass.
        fwd_keys = ["sequences", "attention_mask"]
        for key in ("rollout_expert_indices", "router_padding_mask", "pixel_values", "image_grid_thw"):
            if training_input.get(key) is not None:
                fwd_keys.append(key)
        if training_input.get(upstream_trainer.SAMPLE_SUPPORT_FIELD) is not None:
            fwd_keys.extend([upstream_trainer.SAMPLE_SUPPORT_FIELD, "loss_mask"])
        data_fwd_pass = training_input.select(keys=fwd_keys, metadata_keys=["response_length"])
        output = self.dispatch.forward("policy", data_fwd_pass, reference=True)
        base_log_probs = upstream_trainer.loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")
        training_input["base_action_log_probs"] = base_log_probs[: len(training_input["sequences"])]
        self.dispatch.empty_cache()
        return training_input

    def compute_advantages_and_returns(self, data):
        data = super().compute_advantages_and_returns(data)
        self.e10_snapshot = trajectories.TensorSnapshot(data)
        return data


class E10TrajectoryWriter(TrainingCallback):
    def __init__(self, cfg):
        self.run_dir = Path(os.environ.get("E10_RUN_DIR") or cfg.trainer.export_path)
        self.run_id = os.environ.get("E10_RUN_ID") or cfg.trainer.run_name
        log_dir = os.environ.get("E10_AGENT_LOG_DIR")
        self.agent_log_dir = Path(log_dir) if log_dir else None
        self.algorithm = cfg.trainer.algorithm.advantage_estimator
        self.trajectories_dir = self.run_dir / "trajectories"

    def on_step_end(self, trainer, callback_input, control):
        step = callback_input.global_step
        order = trajectories.REGISTRY.pop_order()
        if callback_input.trajectory_ids is not None:
            order = [tid.to_string() for tid in callback_input.trajectory_ids]
        snapshot = getattr(trainer, "e10_snapshot", None)
        if not order or snapshot is None:
            raise RuntimeError(f"E10 trajectories: step {step} has no batch order or tensor snapshot")

        parts = trajectories.REGISTRY.pop(order)
        rows, missing_env = [], 0
        for i, key in enumerate(order):
            gen = parts[key].get("generation")
            if gen is None:
                raise RuntimeError(f"E10 trajectories: step {step}, rollout {key} has no generation data")
            env = parts[key].get("env")
            missing_env += env is None
            instance_id, _, repetition = key.rpartition("_")
            rows.append({
                "key": key,
                "instance_id": instance_id,
                "repetition_id": int(repetition),
                "generation": gen,
                "env": env,
                "tokens": snapshot.row(i, len(gen["response_ids"])),
            })
        trainer.e10_snapshot = None

        records = trajectories.build_records(self.run_id, step, rows, trainer.tokenizer.decode, self.algorithm)
        path = trajectories.write_step(self.trajectories_dir, step, records)
        student = trajectories.save_student_files(self.trajectories_dir, step, rows, self.agent_log_dir)
        leftover = trajectories.REGISTRY.clear()
        logger.info(
            f"E10 trajectories: step={step} rollouts={len(rows)} records={len(records)} "
            f"missing_env={missing_env} agent_logs={student['agent_logs']} "
            f"agent_logs_missing={student['agent_logs_missing']} test_reports={student['test_reports']} "
            f"leftover={leftover} bytes={path.stat().st_size} file={path}"
        )
