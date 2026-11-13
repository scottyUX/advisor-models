"""Record E7's rewards on the fixture rollouts (run once; output is committed).

Loads E7's SWESmithEnv exactly as it was at E7's last commit on
scott/train-2xa100 (git show <E7_SHA>:advisor_models/swe_smith/env.py), runs each
scripted rollout through it, and writes the rewards it gave to
tests/fixtures/swe_smith_reward_fixtures.json. The test then checks the ported
env gives the same rewards. Needs git history, skyrl_gym and omegaconf; no GPU,
network or model.

    python tests/fixtures/make_swe_smith_reward_fixtures.py
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from swe_smith_fake_rollout import run_rollout  # noqa: E402
import advisor_models.swe_smith  # noqa: E402,F401  (parent package for relative imports)

E7_SHA = "2403225"  # last E7 commit on scott/train-2xa100


def obs(steps):
    return {"terminated": False, "observation": f"obs after {steps}", "total_steps": steps,
            "rolling_summary": "", "last_summarized_step": 0}


def end(steps, **extra):
    return {"terminated": True, "total_steps": steps, **extra}


ROLLOUTS = [
    {"name": "resolved_in_1_step", "instance_id": "jd__tenacity.fx1",
     "steps": [end(1)], "patch": "diff a", "eval_score": 1.0},
    {"name": "resolved_in_6_steps", "instance_id": "jd__tenacity.fx2",
     "steps": [obs(5), end(6)], "patch": "diff b", "eval_score": 1.0},
    {"name": "resolved_in_22_steps", "instance_id": "jd__tenacity.fx3",
     "steps": [obs(5), obs(10), obs(15), obs(20), end(22)], "patch": "diff c", "eval_score": 1.0},
    {"name": "resolved_at_step_cap_40", "instance_id": "jd__tenacity.fx4",
     "steps": [obs(5 * k) for k in range(1, 8)] + [end(40)], "patch": "diff d", "eval_score": 1.0},
    {"name": "unresolved_at_step_cap_40", "instance_id": "jd__tenacity.fx5",
     "steps": [obs(5 * k) for k in range(1, 8)] + [end(40)], "patch": "diff e", "eval_score": 0.0},
    {"name": "unresolved_early_submit", "instance_id": "jd__tenacity.fx6",
     "steps": [obs(5), end(9)], "patch": "diff f", "eval_score": 0.0},
    {"name": "think_block_stripped", "instance_id": "jd__tenacity.fx7",
     "steps": [obs(5), end(8)], "patch": "diff g", "eval_score": 1.0,
     "actions": ["<think>private reasoning</think>\n  Check retry.py first."]},
    {"name": "server_error_terminates_then_scored", "instance_id": "jd__tenacity.fx8",
     "steps": [end(3, error="No successful model calls")], "patch": "", "eval_score": 1.0},
    {"name": "create_session_fails", "instance_id": "jd__tenacity.fx9", "init_error": True,
     "steps": [], "patch": "diff i", "eval_score": 1.0},
    {"name": "execute_step_raises", "instance_id": "jd__tenacity.fx10",
     "steps": [obs(5), "raise"], "patch": "diff j", "eval_score": 1.0},
    {"name": "get_patch_raises", "instance_id": "jd__tenacity.fx11",
     "steps": [obs(5), end(12)], "patch": "raise", "eval_score": 1.0},
]


def load_e7_env():
    src = subprocess.run(["git", "show", f"{E7_SHA}:advisor_models/swe_smith/env.py"],
                         cwd=ROOT, check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "env_e7.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("advisor_models.swe_smith._env_e7", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    e7 = load_e7_env()
    records = []
    for r in ROLLOUTS:
        outputs, fake, _ = run_rollout(e7, r, env_config={})
        records.append({**r, "e7_rewards": [o[0] for o in outputs],
                        "e7_dones": [o[1] for o in outputs], "e7_advice_sent": fake.advice})
    out = {"source": f"E7 SWESmithEnv at advisor-models {E7_SHA}", "rollouts": records}
    path = ROOT / "tests" / "fixtures" / "swe_smith_reward_fixtures.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {len(records)} rollouts to {path}")


if __name__ == "__main__":
    main()
