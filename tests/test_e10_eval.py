"""Offline tests of the E10 A/U eval runner (eval_e10.py; spec §3, §6, G4).

The agent server, grader and advisor are fakes: no GPU, network, Docker or model.

Run: python -m unittest discover -s tests -p "test_e10_eval.py"
"""

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from advisor_models.swe_smith import eval_e10  # noqa: E402

_spec = importlib.util.spec_from_file_location("compare_swe_arms", ROOT / "scripts/compare_swe_arms.py")
compare_swe_arms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_swe_arms)


def issue(iid="jd__tenacity.0d40e76f.func_basic__aaa"):
    return {"instance": {"instance_id": iid, "repo": "jd/tenacity", "problem_statement": "bug",
                         "patch": "gold"},
            "prompt": f"ADVISOR PROMPT for {iid}"}


class FakeAgent:
    """Scripted agent server: each execute_step pops the next result."""

    def __init__(self, results, patch="diff --git a b"):
        self.results = list(results)
        self.patch = patch
        self.calls = []
        self.cleaned = []

    def create_session(self, instance, model_name):
        self.calls.append(("create", instance["instance_id"], model_name))
        return "abcdef0123456789"

    def execute_step(self, session_id, advisor_feedback, max_steps=5):
        self.calls.append(("step", advisor_feedback, max_steps))
        return self.results.pop(0)

    def get_patch(self, session_id):
        return self.patch

    def cleanup_session(self, session_id):
        self.cleaned.append(session_id)


class FakeAdvisor:
    def __init__(self, texts):
        self.texts = list(texts)
        self.seen = []

    def complete(self, messages):
        self.seen.append(messages)
        text = self.texts.pop(0)
        return {"text": text, "finish_reason": "stop", "token_ids": [1, 2],
                "logprobs": [-0.1, -0.2], "prompt_tokens": 100}


def feats(k):
    return {"inputs": {"advice_turn": k / 9}, "raw": {"steps_used": 5 * k}}


def resolved_score(patch, instance):
    return 1.0, "eval_1234", "Status: completed, Resolved: True"


class RunIdTest(unittest.TestCase):
    def test_run_ids_follow_spec(self):
        self.assertEqual(eval_e10.run_id("alone", 1), "exp-10_arm-alone_split-tenacityval_seed-0")
        self.assertEqual(eval_e10.run_id("untrained", 2),
                         "exp-10_arm-untrained_split-tenacityval_seed-0_run-2")
        self.assertEqual(eval_e10.run_id("untrained", 3),
                         "exp-10_arm-untrained_split-tenacityval_seed-0_run-3")
        with self.assertRaises(eval_e10.E10EvalError):
            eval_e10.run_id("grpolora", 1)
        with self.assertRaises(eval_e10.E10EvalError):
            eval_e10.run_id("alone", 4)

    def test_nothing_authorized_yet(self):
        self.assertEqual(eval_e10.AUTHORIZED, {})


class AloneTest(unittest.TestCase):
    def test_one_forty_step_call_with_empty_advice(self):
        agent = FakeAgent([{"terminated": True, "total_steps": 12, "cost": 0.0}])
        row, turns = eval_e10.run_issue(issue(), "alone", agent, None, resolved_score, "hosted_vllm/student")
        self.assertEqual(agent.calls[1], ("step", "", 40))
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(turns, [])
        self.assertTrue(row["resolved"])
        self.assertIsNone(row["error"])
        self.assertEqual(row["total_steps"], 12)
        self.assertAlmostEqual(row["reward"], 0.5 + 0.5 * 28 / 40)


class UntrainedTest(unittest.TestCase):
    def test_e7_eval_protocol(self):
        agent = FakeAgent([
            {"terminated": False, "total_steps": 5, "observation": "obs1", "features": feats(1)},
            {"terminated": False, "total_steps": 10, "observation": "obs2", "features": feats(2)},
            {"terminated": True, "total_steps": 13},
        ])
        advisor = FakeAdvisor(["a0", "<think>x</think> a1", "a2"])
        it = issue()
        row, turns = eval_e10.run_issue(it, "untrained", agent, advisor, resolved_score, "m")

        # Advice before the first step, then every 5 steps; think block stripped for the student.
        self.assertEqual([c for c in agent.calls if c[0] == "step"],
                         [("step", "a0", 5), ("step", "a1", 5), ("step", "a2", 5)])
        prompt = {"role": "user", "content": it["prompt"]}
        self.assertEqual(advisor.seen[0], [prompt])
        # Task prompt, previous advice (unstripped), latest observation only.
        self.assertEqual(advisor.seen[1], [prompt, {"role": "assistant", "content": "a0"},
                                           {"role": "user", "content": "obs1"}])
        self.assertEqual(advisor.seen[2], [prompt,
                                           {"role": "assistant", "content": "<think>x</think> a1"},
                                           {"role": "user", "content": "obs2"}])
        self.assertEqual(row["advisor_calls"], 3)
        self.assertEqual(row["total_steps"], 13)
        # Features: turn 0 before any step, then the agent server's.
        self.assertEqual(turns[0]["features"]["raw"]["steps_used"], 0)
        self.assertEqual(turns[1]["features"], feats(1))
        self.assertEqual([t["student_step"] for t in turns], [0, 5, 10])

        records = eval_e10.build_turn_records("run", row, turns)
        self.assertEqual(len(records), 3)
        self.assertEqual([r["is_last_turn"] for r in records], [False, False, True])
        self.assertIsNone(records[0]["training"])
        self.assertEqual(records[2]["action"]["logprobs_policy"], [-0.1, -0.2])
        self.assertTrue(records[0]["reward"]["resolved"])

    def test_long_observation_is_tail_capped(self):
        long = "x" * (eval_e10.ADVISOR_OBSERVATION_CHARS + 10)
        agent = FakeAgent([{"terminated": False, "total_steps": 5, "observation": long, "features": feats(1)},
                           {"terminated": True, "total_steps": 6}])
        advisor = FakeAdvisor(["a0", "a1"])
        eval_e10.run_issue(issue(), "untrained", agent, advisor, resolved_score, "m")
        shown = advisor.seen[1][2]["content"]
        self.assertTrue(shown.startswith("[earlier context omitted, 10 characters]"))

    def test_never_ending_session_is_an_error(self):
        agent = FakeAgent([{"terminated": False, "total_steps": 5, "observation": "o", "features": feats(1)}] * 20)
        advisor = FakeAdvisor(["a"] * 20)
        row, _ = eval_e10.run_issue(issue(), "untrained", agent, advisor, resolved_score, "m")
        self.assertIn("advice turns", row["error"])
        self.assertEqual(len(advisor.seen), eval_e10.MAX_ADVICE_TURNS)


class ErrorsAreNotUnresolvedTest(unittest.TestCase):
    def test_grading_error(self):
        agent = FakeAgent([{"terminated": True, "total_steps": 3}])
        row, _ = eval_e10.run_issue(issue(), "alone", agent, None,
                                    lambda p, i: (0.0, "ERROR", "Remote evaluation failed: boom"), "m")
        self.assertIn("grading failed", row["error"])
        self.assertEqual(row["status"], "error")

    def test_agent_error_cleans_up(self):
        agent = FakeAgent([{"terminated": True, "total_steps": 0, "error": "Rate limit hit"}])
        row, _ = eval_e10.run_issue(issue(), "alone", agent, None, resolved_score, "m")
        self.assertIn("Rate limit hit", row["error"])
        self.assertEqual(agent.cleaned, ["abcdef0123456789"])

    def test_advisor_error(self):
        class Broken:
            def complete(self, messages):
                raise ConnectionError("advisor down")
        agent = FakeAgent([])
        row, _ = eval_e10.run_issue(issue(), "untrained", agent, Broken(), resolved_score, "m")
        self.assertIn("advisor down", row["error"])


class LogprobTest(unittest.TestCase):
    def test_token_ids(self):
        choice = {"logprobs": {"content": [{"token": "token_id:42", "logprob": -0.5},
                                           {"token": "token_id:7", "logprob": -1.0}]}}
        self.assertEqual(eval_e10.parse_logprobs(choice), ([42, 7], [-0.5, -1.0]))

    def test_missing(self):
        self.assertEqual(eval_e10.parse_logprobs({}), (None, None))

    def test_advisor_request_uses_e7_eval_flags(self):
        resp = mock.Mock()
        resp.json.return_value = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop",
                                               "logprobs": None}], "usage": {"prompt_tokens": 9}}
        with mock.patch.object(eval_e10.requests, "post", return_value=resp) as post:
            out = eval_e10.AdvisorClient("http://x/v1/", "advisor").complete([{"role": "user", "content": "q"}])
        body = post.call_args.kwargs["json"]
        self.assertEqual(post.call_args.args[0], "http://x/v1/chat/completions")
        for k, v in {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "max_tokens": 1024}.items():
            self.assertEqual(body[k], v)
        self.assertEqual(out["text"], "hi")
        self.assertEqual(out["prompt_tokens"], 9)


class RunPassTest(unittest.TestCase):
    def test_run_dir_and_compare_format(self):
        issues = [issue("i1"), issue("i2")]

        class PerIssueAgent(FakeAgent):
            def __init__(self):
                super().__init__([])

            def create_session(self, instance, model_name):
                return f"{instance['instance_id']}-session0000"

            def execute_step(self, session_id, advisor_feedback, max_steps=5):
                return {"terminated": True, "total_steps": 4}

        def score(patch, instance):
            if instance["instance_id"] == "i2":
                return 0.0, "ERROR", "Remote evaluation failed"
            return resolved_score(patch, instance)

        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "agent-logs"
            logs.mkdir()
            (logs / "20261122_000000_i1_i1-sessi.jsonl").write_text("{}\n")
            run_dir = Path(tmp) / "out" / eval_e10.run_id("untrained", 1)
            summary = eval_e10.run_pass("untrained", 1, issues, run_dir, PerIssueAgent(),
                                        FakeAdvisor(["a", "b"]), score, "m", {"code_sha": "abc"},
                                        max_workers=2, agent_log_dir=logs)
            self.assertEqual(summary["errored"], 1)
            self.assertEqual(summary["resolved"], 1)
            self.assertEqual(summary["advice_records"], 2)
            self.assertEqual(summary["agent_logs"], 1)

            manifest = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["run_id"], "exp-10_arm-untrained_split-tenacityval_seed-0")
            self.assertEqual(manifest["code_sha"], "abc")
            rows = compare_swe_arms.load_mapping(str(run_dir / "session_mapping.json"))
            self.assertEqual(set(rows), {"i1", "i2"})
            self.assertTrue(rows["i2"]["error"])
            with gzip.open(run_dir / "trajectories/eval.jsonl.gz", "rt") as f:
                recs = [json.loads(line) for line in f]
            self.assertEqual({r["issue_id"] for r in recs}, {"i1", "i2"})
            self.assertTrue((run_dir / "trajectories/students/i1.agent.jsonl").exists())
            self.assertTrue((run_dir / "trajectories/students/i1.test_report.txt").exists())

            with self.assertRaises(eval_e10.E10EvalError):
                eval_e10.run_pass("untrained", 1, issues, run_dir, PerIssueAgent(),
                                  FakeAdvisor([]), score, "m", {})


class GuardTest(unittest.TestCase):
    def test_refuses_to_run_without_amendment(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(eval_e10, "server_info", return_value={"models": []}), \
                mock.patch.object(eval_e10, "code_state", return_value={"code_sha": "x", "code_dirty": False}), \
                mock.patch.object(eval_e10, "run_pass") as run_pass, \
                mock.patch.dict("os.environ", {k: "set" for k in eval_e10.REQUIRED_ENV}):
            code = eval_e10.main(["--arm", "alone", "--pass", "1", "--out_root", tmp])
        self.assertEqual(code, 1)
        run_pass.assert_not_called()

    def test_test_split_has_49_issues(self):
        self.assertEqual(len(eval_e10.load_issues(str(ROOT / "data/swe_smith/validation_jd__tenacity.json"))),
                         eval_e10.N_TEST_ISSUES)


if __name__ == "__main__":
    unittest.main()
