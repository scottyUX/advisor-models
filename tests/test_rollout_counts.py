"""Offline tests of the E10 per-step rollout counts. No GPU, network or model.

The counting tests always run. The generator test needs upstream SkyRL
3e8700f importable (it runs in hecate/skyrl-upstream-e10:3e8700f) and is
skipped otherwise; it fakes upstream's generate() and calls upstream's real
_zero_reward_if_not_stop.

Run: python -m unittest discover -s tests -p "test_rollout_counts.py"
"""

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor_models.swe_smith.rollout_counts import as_metrics, count_rollouts, log_line


def _have_upstream_generator() -> bool:
    if importlib.util.find_spec("skyrl") is None:
        return False
    try:
        import skyrl.train.generators.skyrl_gym_generator  # noqa: F401
    except Exception:
        return False
    return True


class CountRollouts(unittest.TestCase):
    STOPS = ["stop", "length", "stop", "abort", "length", "stop"]
    REWARDS = [0.9, 0.725, 0.0, 0.5, 0.0, 0.0]

    def test_counts(self):
        c = count_rollouts(self.REWARDS, self.STOPS, zero_reward_on_non_stop=True)
        self.assertEqual(c, {"rollouts_total": 6, "rollouts_length": 2, "rollouts_not_stop": 3,
                             "rollouts_rewarded": 3, "rollouts_zeroed": 2})

    def test_per_token_rewards(self):
        # Upstream gives per-token rewards: the reward sits on the last token.
        rewards = [[0.0, 0.0, 0.9], [0.0, 0.725], [0.0, 0.0], [0.5], [0.0], [0.0, 0.0]]
        c = count_rollouts(rewards, self.STOPS, zero_reward_on_non_stop=True)
        self.assertEqual(c["rollouts_rewarded"], 3)
        self.assertEqual(c["rollouts_zeroed"], 2)

    def test_nothing_zeroed_when_off(self):
        c = count_rollouts(self.REWARDS, self.STOPS, zero_reward_on_non_stop=False)
        self.assertEqual(c["rollouts_zeroed"], 0)
        self.assertEqual(c["rollouts_not_stop"], 3)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            count_rollouts([0.0], ["stop", "stop"], zero_reward_on_non_stop=True)

    def test_metric_names_and_log_line(self):
        c = count_rollouts(self.REWARDS, self.STOPS, zero_reward_on_non_stop=True)
        self.assertEqual(set(as_metrics(c)), {
            "generate/e10_rollouts_total", "generate/e10_rollouts_length",
            "generate/e10_rollouts_not_stop", "generate/e10_rollouts_rewarded",
            "generate/e10_rollouts_zeroed"})
        self.assertEqual(log_line(c), "E10 rollouts: n=6 length=2 not_stop=3 rewarded=3 zeroed=2")


@unittest.skipUnless(_have_upstream_generator(), "upstream SkyRL generator not importable")
class E10Generator(unittest.TestCase):
    def _run(self, zero_on: bool):
        from advisor_models.swe_smith.generator_upstream import E10SkyRLGymGenerator
        from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

        stops = ["stop", "length", "length", "stop"]
        raw = [[0.0, 0.9], [0.0, 0.8], [0.0, 0.0], [0.0, 0.0]]

        async def fake_generate(self, input_batch, disable_tqdm=False):
            # What upstream does at the end of generate(): metrics, then zeroing.
            rewards = [list(r) for r in raw]
            if zero_on:
                rewards = self._zero_reward_if_not_stop(rewards, stops)
            return {"rewards": rewards, "stop_reasons": stops, "rollout_metrics": {"x": 1}}

        gen = object.__new__(E10SkyRLGymGenerator)
        with mock.patch.object(SkyRLGymGenerator, "generate", fake_generate):
            return asyncio.run(gen.generate({}))

    def test_counts_taken_before_zeroing(self):
        out = self._run(zero_on=True)
        m = out["rollout_metrics"]
        self.assertEqual(m["x"], 1)  # upstream's own metrics kept
        self.assertEqual(m["generate/e10_rollouts_total"], 4)
        self.assertEqual(m["generate/e10_rollouts_length"], 2)
        self.assertEqual(m["generate/e10_rollouts_rewarded"], 2)
        self.assertEqual(m["generate/e10_rollouts_zeroed"], 1)
        self.assertEqual(out["rewards"][1], [0.0, 0.0])  # upstream still zeroed it

    def test_zeroing_off(self):
        m = self._run(zero_on=False)["rollout_metrics"]
        self.assertEqual(m["generate/e10_rollouts_zeroed"], 0)
        self.assertEqual(m["generate/e10_rollouts_length"], 2)


if __name__ == "__main__":
    unittest.main()
