"""Offline tests (E10 gate G4, amendment A2): the ported SWE-smith reward
matches E7's on fixture rollouts. No GPU, network, Docker or model.

The fixture rewards were recorded from E7's own SWESmithEnv (see
tests/fixtures/make_swe_smith_reward_fixtures.py). Each rollout is replayed
through the current env with a fake agent server and a fake grader.

Run: python -m unittest discover -s tests -p "test_swe_smith_reward.py"
(Not `python -m unittest tests/...`: the swesmith wheel installs a top-level
`tests` package that shadows this folder once swesmith is installed.)
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from swe_smith_fake_rollout import run_rollout  # noqa: E402

from advisor_models.swe_smith import env as env_module  # noqa: E402
from advisor_models.swe_smith.images import IMAGE_OVERRIDE_ENV, resolve_image_name  # noqa: E402
from advisor_models.swe_smith.reward import e7_reward  # noqa: E402

FIXTURES = json.loads((TESTS / "fixtures" / "swe_smith_reward_fixtures.json").read_text())


class PortedRewardMatchesE7(unittest.TestCase):
    def test_every_fixture_rollout(self):
        # Upstream SkyRL passes env_config as a plain dict (E7's fork: DictConfig).
        for r in FIXTURES["rollouts"]:
            with self.subTest(rollout=r["name"]):
                outputs, fake, _ = run_rollout(env_module, r, env_config={})
                self.assertEqual([o[0] for o in outputs], r["e7_rewards"])
                self.assertEqual([o[1] for o in outputs], r["e7_dones"])
                self.assertEqual(fake.advice, r["e7_advice_sent"])

    def test_env_config_none_also_works(self):
        r = FIXTURES["rollouts"][1]
        outputs, _, _ = run_rollout(env_module, r, env_config=None)
        self.assertEqual([o[0] for o in outputs], r["e7_rewards"])

    def test_session_cleaned_up_after_final_result(self):
        r = next(x for x in FIXTURES["rollouts"] if x["name"] == "resolved_in_6_steps")
        _, fake, score_calls = run_rollout(env_module, r, env_config={})
        self.assertEqual(fake.cleaned, ["session-" + r["instance_id"]])
        self.assertEqual(score_calls, [r["patch"]])


class E7RewardFormula(unittest.TestCase):
    def test_spec_formula(self):
        # E10 spec §2: 0.5 + 0.5 * (40 - steps) / 40 if resolved, else 0.
        for steps in range(0, 41):
            self.assertAlmostEqual(e7_reward(1.0, steps), 0.5 + 0.5 * (40 - steps) / 40)
            self.assertEqual(e7_reward(0.0, steps), 0.0)

    def test_known_values(self):
        self.assertEqual(e7_reward(1.0, 6), 0.925)
        self.assertEqual(e7_reward(1.0, 40), 0.5)


class ImageOverride(unittest.TestCase):
    INSTANCE = {"image_name": "jyangballin/swesmith.x86_64.jd_1776_tenacity.0d40e76f"}

    def test_unset_keeps_instance_image(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(IMAGE_OVERRIDE_ENV, None)
            self.assertEqual(resolve_image_name(self.INSTANCE), self.INSTANCE["image_name"])

    def test_set_uses_arm_image(self):
        arm = "hecate/swesmith.arm64.jd_1776_tenacity.0d40e76f:e10"
        with mock.patch.dict(os.environ, {IMAGE_OVERRIDE_ENV: arm}):
            self.assertEqual(resolve_image_name(self.INSTANCE), arm)


if __name__ == "__main__":
    unittest.main()
