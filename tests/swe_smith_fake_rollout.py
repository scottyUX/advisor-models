"""Drive SWESmithEnv through a scripted rollout with no agent server, Docker or model.

A fixture rollout is a dict:
    instance_id   str
    init_error    bool, optional: create_session raises
    steps         list of agent-server responses to execute_step, in order;
                  the string "raise" makes that call raise
    patch         str, or "raise" to make get_patch raise
    eval_score    float compute_score returns (1.0 resolved, 0.0 not)
    actions       advisor outputs, one per env.step (cycled if short)
Used by tests/test_swe_smith_reward.py and by the fixture generator.
"""

import json
from unittest import mock


class FakeRemoteAgentClient:
    def __init__(self, rollout):
        self.rollout = rollout
        self.calls = 0
        self.advice = []
        self.cleaned = []

    def create_session(self, instance, model_name=None, extra_instruction=None):
        if self.rollout.get("init_error"):
            raise RuntimeError("fixture: create_session failed")
        return "session-" + self.rollout["instance_id"]

    def execute_step(self, session_id, advisor_feedback, max_steps=5):
        self.advice.append(advisor_feedback)
        result = self.rollout["steps"][self.calls]
        self.calls += 1
        if result == "raise":
            raise RuntimeError("fixture: execute_step failed")
        return dict(result)

    def get_patch(self, session_id):
        if self.rollout["patch"] == "raise":
            raise RuntimeError("fixture: get_patch failed")
        return self.rollout["patch"]

    def cleanup_session(self, session_id):
        self.cleaned.append(session_id)


def extras_for(rollout):
    instance = {"instance_id": rollout["instance_id"], "image_name": "fixture"}
    return {
        "reward_spec": {"ground_truth_json": json.dumps(instance)},
        "original_question": "fixture problem",
    }


def run_rollout(env_module, rollout, env_config=None, max_env_steps=20):
    """Run one scripted rollout through ``env_module.SWESmithEnv``.

    Returns (outputs, fake_client, compute_score_calls): ``outputs`` holds
    (reward, done) per env.step.
    """
    fake = FakeRemoteAgentClient(rollout)
    score_calls = []

    def fake_compute_score(patch, instance):
        score_calls.append(patch)
        return rollout["eval_score"], "eval-fixture", "fixture"

    with mock.patch.object(env_module, "RemoteAgentClient", lambda *a, **k: fake), \
         mock.patch.object(env_module, "compute_score", fake_compute_score):
        env = env_module.SWESmithEnv(env_config, extras_for(rollout))
        env.init([{"role": "user", "content": "advisor prompt"}])
        outputs = []
        actions = rollout.get("actions") or ["advice"]
        for i in range(max_env_steps):
            out = env.step(actions[i % len(actions)])
            outputs.append((out["reward"], out["done"]))
            if out["done"]:
                break
    return outputs, fake, score_calls
