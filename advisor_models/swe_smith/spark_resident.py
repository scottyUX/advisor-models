"""E10 on the DGX Spark: keep every model where it is (no CPU offload, no vLLM sleep).

Upstream SkyRL's colocated mode (one GPU shared by the trainer and the
advisor's vLLM) frees GPU memory by moving models to CPU memory: the trainer's
models between phases, and the vLLM's base weights whenever it sleeps (level 1,
forced whenever FSDP syncs only the LoRA adapter). On the GB10 the GPU and the
CPU share one 119 GB pool, so each move makes a second copy instead of freeing
anything. G1 ran out of memory three times on these copies (16 Nov,
spark/README.md in the hecate repo).

``keep_models_resident()`` makes the moves no-ops: the actor groups'
``offload_to_cpu``/``backload_to_gpu`` do nothing, and the inference client's
sleep and wake calls do nothing, with ``is_sleeping()`` False. Everything else
runs as upstream wrote it. The advisor's vLLM stays awake, and its LoRA adapter
is synced into the running engine, the path upstream itself uses when the
trainer and vLLM are not colocated; the FSDP worker resets vLLM's prefix cache
on every adapter sync, so no cached prompt from an older adapter is reused.
Training and sampling are unchanged; only where the weights sit is.

Call it in the process that runs the trainer (the Ray entrypoint task), before
the experiment is built.
"""

from loguru import logger

_applied = False


async def _no_sleep_or_wake(self, *args, **kwargs):
    return {}


async def _never_sleeping(self):
    return False


def _stay_resident(self, nonblocking=False, *args, **kwargs):
    return [] if nonblocking else None


def keep_models_resident() -> None:
    global _applied
    if _applied:
        return
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
    from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup

    for name in ("sleep", "wake_up", "sleep_for_weight_sync", "wake_for_weight_sync"):
        if not hasattr(RemoteInferenceClient, name):
            raise RuntimeError(f"upstream RemoteInferenceClient has no {name}(); check the SkyRL commit")
        setattr(RemoteInferenceClient, name, _no_sleep_or_wake)
    RemoteInferenceClient.is_sleeping = _never_sleeping
    for name in ("offload_to_cpu", "backload_to_gpu"):
        if not hasattr(PPORayActorGroup, name):
            raise RuntimeError(f"upstream PPORayActorGroup has no {name}(); check the SkyRL commit")
        setattr(PPORayActorGroup, name, _stay_resident)
    _applied = True
    logger.info("E10 Spark: models stay resident (no CPU offload, no vLLM sleep); unified memory")
