"""Per-step rollout counts for E10 (Scott's review, question 3).

E7's SkyRL fork logged, every training step, how often the length cap ended a
rollout and how many rewarded rollouts ``zero_reward_on_non_stop`` zeroed.
Upstream SkyRL 3e8700f does not, so the E10 generator computes the counts
here, from the rewards and stop reasons just before upstream zeroes them.

Pure Python, no SkyRL import, so it is unit-tested offline.
"""

from typing import Dict, List, Sequence, Union

Reward = Union[float, List[float]]

METRIC_PREFIX = "generate/e10_"


def _total(reward: Reward) -> float:
    return float(sum(reward)) if isinstance(reward, list) else float(reward)


def count_rollouts(
    rewards: Sequence[Reward], stop_reasons: Sequence[str], zero_reward_on_non_stop: bool
) -> Dict[str, int]:
    """Counts for one generate() call (one training step's batch).

    Args:
        rewards: raw rewards, before zeroing (a float, or per-token rewards).
        stop_reasons: one per rollout, as upstream's generator sets them.
        zero_reward_on_non_stop: whether upstream will zero non-"stop" rollouts.

    Returns:
        rollouts_total     every rollout in the batch
        rollouts_length    stop_reason == "length"
        rollouts_not_stop  stop_reason != "stop" (what upstream zeroes:
                           "length" plus anything else, e.g. "abort")
        rollouts_rewarded  raw reward > 0
        rollouts_zeroed    raw reward > 0 and zeroed by zero_reward_on_non_stop
    """
    if len(rewards) != len(stop_reasons):
        raise ValueError(f"{len(rewards)} rewards but {len(stop_reasons)} stop reasons")
    not_stop = [r != "stop" for r in stop_reasons]
    rewarded = [_total(r) > 0.0 for r in rewards]
    return {
        "rollouts_total": len(stop_reasons),
        "rollouts_length": sum(r == "length" for r in stop_reasons),
        "rollouts_not_stop": sum(not_stop),
        "rollouts_rewarded": sum(rewarded),
        "rollouts_zeroed": (
            sum(n and w for n, w in zip(not_stop, rewarded)) if zero_reward_on_non_stop else 0
        ),
    }


def as_metrics(counts: Dict[str, int]) -> Dict[str, int]:
    """Keys as logged with the trainer's per-step metrics."""
    return {METRIC_PREFIX + k: v for k, v in counts.items()}


def log_line(counts: Dict[str, int]) -> str:
    """One greppable line per step, like E7's "E7 rollouts: ..." line."""
    return (
        f"E10 rollouts: n={counts['rollouts_total']} length={counts['rollouts_length']} "
        f"not_stop={counts['rollouts_not_stop']} rewarded={counts['rollouts_rewarded']} "
        f"zeroed={counts['rollouts_zeroed']}"
    )
