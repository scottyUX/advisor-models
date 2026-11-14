"""E10 generator: upstream SkyRL's SkyRLGymGenerator plus per-step rollout counts.

Behaviour is upstream's, unchanged: same agent loop, same stop reasons, same
``zero_reward_on_non_stop``. The only addition is bookkeeping. Each
``generate()`` call (one training step's batch, or one eval batch) adds to
``rollout_metrics``, which the trainer logs with the step's other metrics:

    generate/e10_rollouts_total      rollouts in the batch
    generate/e10_rollouts_length     stop_reason == "length"
    generate/e10_rollouts_not_stop   stop_reason != "stop" (what gets zeroed)
    generate/e10_rollouts_rewarded   raw reward > 0, before zeroing
    generate/e10_rollouts_zeroed     raw reward > 0 and zeroed

and logs one line, ``E10 rollouts: n=.. length=.. not_stop=.. rewarded=.. zeroed=..``.

The counts are taken inside ``_zero_reward_if_not_stop``, where upstream holds
the raw rewards and the final stop reasons together, before it zeroes them.
"""

import contextvars
from typing import Dict, List, Optional

from loguru import logger
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

from .rollout_counts import as_metrics, count_rollouts, log_line

# One box per generate() call, so concurrent calls cannot mix their counts.
_counts_box: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "e10_rollout_counts", default=None
)


class E10SkyRLGymGenerator(SkyRLGymGenerator):
    async def generate(
        self, input_batch: GeneratorInput, disable_tqdm: bool = False
    ) -> GeneratorOutput:
        box: dict = {}
        token = _counts_box.set(box)
        try:
            output = await super().generate(input_batch, disable_tqdm=disable_tqdm)
        finally:
            _counts_box.reset(token)

        counts: Optional[Dict[str, int]] = box.get("counts")
        if counts is None:
            # zero_reward_on_non_stop is off (or the batched path ran): nothing
            # was zeroed, and the output's rewards are still the raw rewards.
            counts = count_rollouts(
                output["rewards"], output["stop_reasons"], zero_reward_on_non_stop=False
            )
        metrics = output.get("rollout_metrics") or {}
        metrics.update(as_metrics(counts))
        output["rollout_metrics"] = metrics
        logger.info(log_line(counts))
        return output

    def _zero_reward_if_not_stop(self, rewards: List, stop_reasons: List[str]) -> List:
        box = _counts_box.get()
        if box is not None:
            box["counts"] = count_rollouts(rewards, stop_reasons, zero_reward_on_non_stop=True)
        return super()._zero_reward_if_not_stop(rewards, stop_reasons)
