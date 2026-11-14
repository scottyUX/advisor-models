"""E7's SWE-smith reward, as one pure function.

Kept apart from env.py so the reward can be unit-tested without a remote agent,
a Docker container, a GPU or a model (E10 spec, amendment A2 and gate G4).
"""

LENGTH_COEFFICIENT = 0.5
MAX_STEPS = 40


def e7_reward(eval_score: float, total_steps: int) -> float:
    """Reward for one finished rollout.

    Args:
        eval_score: 1.0 if the patch passed the hidden tests, else 0.0
            (what ``config.compute_score`` returns).
        total_steps: student steps taken, as reported by the agent server.

    Returns:
        ``0.5 + 0.5 * (40 - steps) / 40`` if resolved, else ``eval_score``
        unchanged (0.0), exactly as E7's ``SWESmithEnv._return_final_result``.
    """
    if eval_score == 1.0:
        return (1 - LENGTH_COEFFICIENT) + LENGTH_COEFFICIENT * (
            MAX_STEPS - total_steps
        ) / MAX_STEPS
    return eval_score
