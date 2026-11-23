"""E10 eval passes for the untrained arms A and U (spec §3, §4, §6).

  A  ``alone``      the student alone: one agent session per issue, no advice
  U  ``untrained``  the student with the untrained Qwen2.5-7B-Instruct advisor

One invocation is one pass of one arm over the 49 test issues, with its own run
id (spec §3): ``exp-10_arm-<arm>_split-tenacityval_seed-0``, then ``_run-2`` and
``_run-3`` for passes 2 and 3. Each pass writes a run dir:

    <out_root>/<run_id>/
        manifest.json            run id, CODE_SHA, config, server versions
        session_mapping.json     per issue, the format scripts/compare_swe_arms.py reads
        trajectories/eval.jsonl.gz
                                 U only: one record per advice turn (spec §6;
                                 no advantage or value fields)
        trajectories/students/<instance_id>.agent.jsonl
        trajectories/students/<instance_id>.test_report.txt

Protocol, the same as E7's eval scripts so A and U match how E7 measured them:

  A  ``eval_swe_smith.py``: one ``execute_step`` with empty advice and a
     40-step budget.
  U  ``eval_advisor.py``: advice before the first student step, then after
     every 5 student steps until the session ends. Each advisor call sees the
     task prompt, its previous advice and the latest observation (tail-capped
     at 60,000 characters), not the whole transcript. Sampling: E7's eval
     flags (temperature 0.7, top_p 0.8, top_k 20, min_p 0, 1,024-token
     replies). The task prompt is the data file's own prompt, the text the
     trained arms see in training.

Two differences from E7's scripts, both so a failure never counts as
unresolved: a grading error (``compute_score`` returns eval id ``ERROR``) is an
errored row, and so is an advisor call that fails. ``compare_swe_arms.py``
drops errored issues from every arm.

The advisor is any OpenAI-compatible vLLM server (``scripts/e10_serve_advisor.sh``
for U). The student runs inside the agent server; this process never calls it.

Guard: no amendment authorizes an A or U pass yet (A4 excludes them), so the
runner refuses to run issues. ``--check`` only checks the servers and prints the
plan. Add an arm to ``AUTHORIZED`` only in the commit that records the
amendment authorizing it.

Usage (on the Spark, agent and eval servers up as for G1):
    AGENT_SERVER_URL=http://localhost:8081 EVAL_SERVER_URL=http://localhost:8082 \\
    SWESMITH_IMAGE_OVERRIDE=hecate/swesmith.arm64.jd_1776_tenacity.0d40e76f:e10 \\
    E10_AGENT_LOG_DIR=~/e10-work/agent-logs \\
    python -m advisor_models.swe_smith.eval_e10 --arm untrained --pass 1 [--check]
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .features import FeatureTracker
from .reward import e7_reward

SCHEMA_VERSION = 1

# Arm -> the amendment that authorizes its eval passes. Empty: A4 does not
# authorize A or U. Add an entry only in the commit that records the amendment.
AUTHORIZED: Dict[str, str] = {}

ARMS = ("alone", "untrained")
PASSES = (1, 2, 3)
N_TEST_ISSUES = 49
UNTRAINED_ADVISOR = "Qwen/Qwen2.5-7B-Instruct"

# E7's eval protocol (eval_advisor.py, eval_swe_smith.py, env.py).
EVAL_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "max_tokens": 1024}
ADVISOR_OBSERVATION_CHARS = 60_000
STEPS_PER_ADVICE = 5
STUDENT_MAX_STEPS = 40
# 40 steps / 5 per turn = 8 turns, and the agent server ends the session at its
# step cap; more than this means it never reported the end.
MAX_ADVICE_TURNS = STUDENT_MAX_STEPS // STEPS_PER_ADVICE + 2


class E10EvalError(RuntimeError):
    pass


def run_id(arm: str, pass_index: int) -> str:
    if arm not in ARMS:
        raise E10EvalError(f"arm must be one of {ARMS}, got {arm!r}")
    if pass_index not in PASSES:
        raise E10EvalError(f"pass must be one of {PASSES}, got {pass_index!r}")
    base = f"exp-10_arm-{arm}_split-tenacityval_seed-0"
    return base if pass_index == 1 else f"{base}_run-{pass_index}"


def tail_text(text: str, limit: int = ADVISOR_OBSERVATION_CHARS) -> str:
    """E7's eval_advisor._tail_text."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"[earlier context omitted, {omitted} characters]\n{text[-limit:]}"


def strip_think(advice: str) -> str:
    """What the student is shown, as env.py and eval_advisor.py do."""
    if "</think>" in advice:
        return advice.split("</think>")[1].strip()
    return advice


def load_issues(data_file: str) -> List[Dict[str, Any]]:
    """Rows of a SkyRL SWE-smith data file: the instance and its advisor prompt."""
    issues = []
    with open(data_file) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            spec = row["reward_spec"]
            instance = (json.loads(spec["ground_truth_json"]) if "ground_truth_json" in spec
                        else spec["ground_truth"])
            issues.append({"instance": instance, "prompt": row["prompt"][-1]["content"]})
    return issues


# --------------------------------------------------------------------------- advisor


def parse_logprobs(choice: Dict[str, Any]) -> Tuple[Optional[List[int]], Optional[List[float]]]:
    """Token ids and log-probs from a vLLM chat choice requested with
    ``logprobs`` and ``return_tokens_as_token_ids`` (tokens read "token_id:<n>")."""
    content = (choice.get("logprobs") or {}).get("content")
    if not content:
        return None, None
    ids, lps = [], []
    for entry in content:
        token = entry.get("token", "")
        if not token.startswith("token_id:"):
            return None, [float(e["logprob"]) for e in content]
        ids.append(int(token.split(":", 1)[1]))
        lps.append(float(entry["logprob"]))
    return ids, lps


class AdvisorClient:
    """One advisor call per advice turn, on an OpenAI-compatible vLLM server."""

    def __init__(self, base_url: str, model: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": messages, **EVAL_SAMPLING,
                  "logprobs": True, "return_tokens_as_token_ids": True},
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        token_ids, logprobs = parse_logprobs(choice)
        return {
            "text": choice["message"]["content"] or "",
            "finish_reason": choice.get("finish_reason"),
            "token_ids": token_ids,
            "logprobs": logprobs,
            "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
        }


# --------------------------------------------------------------------------- one issue


def _base_row(instance: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instance_id": instance["instance_id"],
        "session_id": None,
        "eval_id": None,
        "diff": "",
        "resolved": False,
        "status": "error",
        "total_steps": 0,
        "agent_cost": 0.0,
        "advisor_cost": 0.0,
        "advisor_calls": 0,
        "advisor_guidance_history": [],
        "reward": None,
        "test_report": None,
        "error": None,
    }


def _cleanup(agent: Any, session_id: Optional[str]) -> None:
    if session_id:
        try:
            agent.cleanup_session(session_id)
        except Exception:
            pass


def _check_step(result: Dict[str, Any]) -> None:
    # A failed session (dead model server, timeout) is an error, not an unresolved attempt.
    if result.get("error"):
        raise E10EvalError(f"agent session failed: {result['error']}")


def run_issue(
    issue: Dict[str, Any],
    arm: str,
    agent: Any,
    advisor: Optional[AdvisorClient],
    score_fn: Callable[[str, Dict[str, Any]], Tuple[float, str, str]],
    agent_model: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One issue, one arm: the session_mapping row and the advice turns (U only).

    Never raises: any failure is returned as an errored row.
    """
    instance = issue["instance"]
    row = _base_row(instance)
    turns: List[Dict[str, Any]] = []
    session_id = None
    try:
        session_id = agent.create_session(instance, model_name=agent_model)
        row["session_id"] = session_id

        if arm == "alone":
            result = agent.execute_step(session_id, "", max_steps=STUDENT_MAX_STEPS)
            _check_step(result)
        else:
            prompt = {"role": "user", "content": issue["prompt"]}
            messages = [prompt]
            features = FeatureTracker().snapshot(advice_turn=0)
            student_steps = 0
            while True:
                if len(turns) >= MAX_ADVICE_TURNS:
                    raise E10EvalError(f"session still running after {len(turns)} advice turns")
                reply = advisor.complete(messages)
                turns.append({
                    "messages": messages,
                    "reply": reply,
                    "features": features,
                    "student_step": student_steps,
                })
                result = agent.execute_step(session_id, strip_think(reply["text"]),
                                            max_steps=STEPS_PER_ADVICE)
                _check_step(result)
                student_steps = result.get("total_steps", 0)
                if result.get("terminated"):
                    break
                features = result.get("features")
                messages = [
                    prompt,
                    {"role": "assistant", "content": reply["text"]},
                    {"role": "user", "content": tail_text(result.get("observation", ""))},
                ]
            row["advisor_calls"] = len(turns)
            row["advisor_guidance_history"] = [t["reply"]["text"] for t in turns]

        row["total_steps"] = result.get("total_steps", 0)
        row["agent_cost"] = result.get("cost", 0.0)
        patch = agent.get_patch(session_id)
        row["diff"] = patch
        _cleanup(agent, session_id)
        session_id = None

        score, eval_id, info = score_fn(patch, instance)
        row["eval_id"] = eval_id
        row["test_report"] = info
        if eval_id == "ERROR":
            raise E10EvalError(f"grading failed: {info}")
        row["resolved"] = score > 0.5
        row["reward"] = e7_reward(score, row["total_steps"])
        status = info.split("Status: ")[1].split(",")[0] if "Status: " in info else "unknown"
        row["status"] = f"{status} (steps: {row['total_steps']}, advisor_calls: {row['advisor_calls']})"
    except Exception as e:
        row["resolved"] = False
        row["status"] = "error"
        row["error"] = f"[{type(e).__name__}] {str(e)[:500]}"
        _cleanup(agent, session_id)
    return row, turns


def build_turn_records(run: str, row: Dict[str, Any], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Spec §6 records for one U issue, one per advice turn (no training fields)."""
    iid = row["instance_id"]
    reward = {
        "resolved": None if row["error"] else row["resolved"],
        "student_steps": row["total_steps"],
        "reward": row["reward"],
        "error": row["error"],
    }
    records = []
    for t, turn in enumerate(turns):
        reply = turn["reply"]
        features = turn["features"]
        records.append({
            "schema": SCHEMA_VERSION,
            "run_id": run,
            "step": None,
            "trajectory_key": f"{iid}_0",
            "issue_id": iid,
            "group_index": None,
            "rollout_index": 0,
            "turn_index": t,
            "n_turns": len(turns),
            "student_step": turn["student_step"],
            "state": {"messages": turn["messages"], "prompt_tokens": reply["prompt_tokens"]},
            "action": {
                "text": reply["text"],
                "finish_reason": reply["finish_reason"],
                "token_ids": reply["token_ids"],
                "logprobs_policy": reply["logprobs"],
                "logprobs_policy_source": "vllm_eval",
                "logprobs_reference": None,
            },
            "reward": reward,
            "final_reward": row["reward"],
            "is_last_turn": t == len(turns) - 1,
            "features": features,
            "training": None,
            "student": {
                "session_id": row["session_id"],
                "agent_log": f"students/{iid}.agent.jsonl",
                "test_report": f"students/{iid}.test_report.txt",
            },
        })
    return records


# --------------------------------------------------------------------------- run dir


def save_student_files(students_dir: Path, rows: List[Dict[str, Any]],
                       agent_log_dir: Optional[Path]) -> Dict[str, int]:
    """Each issue's agent session log (copied from the agent server's --log-dir) and test report."""
    students_dir.mkdir(parents=True, exist_ok=True)
    counts = {"agent_logs": 0, "agent_logs_missing": 0, "test_reports": 0}
    for row in rows:
        iid = row["instance_id"]
        if row["test_report"] is not None:
            (students_dir / f"{iid}.test_report.txt").write_text(row["test_report"], encoding="utf-8")
            counts["test_reports"] += 1
        sid = row["session_id"]
        matches = (sorted(agent_log_dir.glob(f"*_{sid[:8]}.jsonl"))
                   if agent_log_dir is not None and sid else [])
        if matches:
            shutil.copyfile(matches[-1], students_dir / f"{iid}.agent.jsonl")
            counts["agent_logs"] += 1
        else:
            counts["agent_logs_missing"] += 1
    return counts


def code_state(repo_root: Path) -> Dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    return {"code_sha": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--porcelain"))}


def server_info(base_url: str) -> Dict[str, Any]:
    """Served model ids and the vLLM version of an OpenAI-compatible server."""
    base = base_url.rstrip("/")
    info: Dict[str, Any] = {"url": base}
    models = requests.get(f"{base}/models", timeout=10)
    models.raise_for_status()
    info["models"] = [{"id": m.get("id"), "root": m.get("root"), "max_model_len": m.get("max_model_len")}
                      for m in models.json().get("data", [])]
    root = base[:-3] if base.endswith("/v1") else base
    try:
        version = requests.get(f"{root}/version", timeout=10)
        info["vllm_version"] = version.json().get("version") if version.ok else None
    except Exception:
        info["vllm_version"] = None
    return info


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_pass(
    arm: str,
    pass_index: int,
    issues: List[Dict[str, Any]],
    run_dir: Path,
    agent: Any,
    advisor: Optional[AdvisorClient],
    score_fn: Callable[[str, Dict[str, Any]], Tuple[float, str, str]],
    agent_model: str,
    manifest: Dict[str, Any],
    max_workers: int = 8,
    agent_log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run every issue once and write the run dir. Returns the summary."""
    run = run_id(arm, pass_index)
    if run_dir.exists():
        raise E10EvalError(f"{run_dir} exists; a pass is never overwritten")
    run_dir.mkdir(parents=True)
    manifest = {**manifest, "schema": SCHEMA_VERSION, "run_id": run, "arm": arm,
                "pass": pass_index, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lock = threading.Lock()
    done = [0]

    def one(issue: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        out = run_issue(issue, arm, agent, advisor, score_fn, agent_model)
        with lock:
            done[0] += 1
            row = out[0]
            mark = "ERROR" if row["error"] else ("resolved" if row["resolved"] else "unresolved")
            print(f"[{done[0]}/{len(issues)}] {row['instance_id']}: {mark}, "
                  f"{row['total_steps']} steps, {row['advisor_calls']} advice"
                  + (f" ({row['error']})" if row["error"] else ""), flush=True)
        return out

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        outputs = list(pool.map(one, issues))

    rows = [r for r, _ in outputs]
    mapping = {r["instance_id"]: r for r in rows}
    if len(mapping) != len(rows):
        raise E10EvalError("duplicate instance_id in the data file")
    (run_dir / "session_mapping.json").write_text(json.dumps(mapping, indent=2))

    traj_dir = run_dir / "trajectories"
    records = [rec for row, turns in outputs for rec in build_turn_records(run, row, turns)]
    if arm != "alone":
        traj_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(traj_dir / "eval.jsonl.gz", "wt", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    files = save_student_files(traj_dir / "students", rows, agent_log_dir)

    ok = [r for r in rows if not r["error"]]
    summary = {
        "run_id": run,
        "issues": len(rows),
        "errored": len(rows) - len(ok),
        "resolved": sum(r["resolved"] for r in ok),
        "resolve_rate_of_ok": (sum(r["resolved"] for r in ok) / len(ok)) if ok else None,
        "mean_steps_of_ok": (sum(r["total_steps"] for r in ok) / len(ok)) if ok else None,
        "advice_records": len(records),
        **files,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    manifest["summary"] = summary
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


# --------------------------------------------------------------------------- CLI


REQUIRED_ENV = ("AGENT_SERVER_URL", "EVAL_SERVER_URL", "SWESMITH_IMAGE_OVERRIDE")


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="E10 eval pass for arm A (alone) or U (untrained)")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--pass", dest="pass_index", type=int, required=True, choices=PASSES)
    p.add_argument("--data_file", default=str(repo_root / "data/swe_smith/validation_jd__tenacity.json"))
    p.add_argument("--out_root", default=os.path.expanduser("~/e10-work/evals"))
    p.add_argument("--agent_model", default=os.environ.get("AGENT_MODEL", "hosted_vllm/student"))
    p.add_argument("--student_url", default=os.environ.get("HOSTED_VLLM_API_BASE", "http://localhost:8100/v1"))
    p.add_argument("--advisor_url", default="http://localhost:8101/v1")
    p.add_argument("--advisor_model", default="advisor", help="served model name of the advisor vLLM")
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--agent_log_dir", default=os.environ.get("E10_AGENT_LOG_DIR"))
    p.add_argument("--check", action="store_true",
                   help="check servers and data and print the plan; run no issue")
    args = p.parse_args(argv)

    run = run_id(args.arm, args.pass_index)
    run_dir = Path(args.out_root) / run
    problems = []

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        problems.append(f"environment not set: {', '.join(missing)}")
    issues = load_issues(args.data_file)
    if len(issues) != N_TEST_ISSUES:
        problems.append(f"{args.data_file} has {len(issues)} issues, the test split has {N_TEST_ISSUES}")
    if run_dir.exists():
        problems.append(f"{run_dir} exists; a pass is never overwritten")
    code = code_state(repo_root)
    if code["code_dirty"]:
        problems.append("the advisor-models tree has uncommitted changes; CODE_SHA would not describe the code")

    servers: Dict[str, Any] = {}
    try:
        servers["student"] = server_info(args.student_url)
    except Exception as e:
        problems.append(f"student server at {args.student_url}: {e}")
    advisor = None
    if args.arm == "untrained":
        try:
            servers["advisor"] = server_info(args.advisor_url)
            served = {m["id"]: m for m in servers["advisor"]["models"]}
            if args.advisor_model not in served:
                problems.append(f"advisor server serves {sorted(served)}, not {args.advisor_model!r}")
            elif served[args.advisor_model].get("root") not in (UNTRAINED_ADVISOR, None):
                problems.append(f"advisor {args.advisor_model!r} is {served[args.advisor_model]['root']}, "
                                f"not {UNTRAINED_ADVISOR}")
        except Exception as e:
            problems.append(f"advisor server at {args.advisor_url}: {e}")
        advisor = AdvisorClient(args.advisor_url, args.advisor_model)

    manifest = {
        **code,
        "authorized_by": AUTHORIZED.get(args.arm),
        "split": "tenacityval",
        "data_file": os.path.relpath(args.data_file, repo_root),
        "data_sha256": sha256(args.data_file),
        "n_issues": len(issues),
        "agent_model": args.agent_model,
        "advisor": None if args.arm == "alone" else {"model": UNTRAINED_ADVISOR,
                                                     "served_name": args.advisor_model},
        "protocol": {"eval_sampling": EVAL_SAMPLING, "steps_per_advice": STEPS_PER_ADVICE,
                     "student_max_steps": STUDENT_MAX_STEPS,
                     "advisor_observation_chars": ADVISOR_OBSERVATION_CHARS},
        "servers": servers,
        "env": {k: os.environ.get(k) for k in REQUIRED_ENV},
        "max_workers": args.max_workers,
    }

    print(f"E10 eval {run}: arm {args.arm}, pass {args.pass_index}, {len(issues)} issues -> {run_dir}")
    print(json.dumps({"code": code, "servers": servers}, indent=2))
    for prob in problems:
        print(f"PROBLEM: {prob}", file=sys.stderr)
    if args.check:
        return 1 if problems else 0
    if args.arm not in AUTHORIZED:
        print(f"E10 STOP: no amendment authorizes the {args.arm} eval passes yet (A4 excludes A and U). "
              "Add the arm to AUTHORIZED only in the commit that records the amendment.", file=sys.stderr)
        return 1
    if problems:
        return 1

    from .config import compute_score
    from .remote_agent_client import RemoteAgentClient

    summary = run_pass(
        args.arm, args.pass_index, issues, run_dir, RemoteAgentClient(), advisor, compute_score,
        args.agent_model, manifest, max_workers=args.max_workers,
        agent_log_dir=Path(args.agent_log_dir).expanduser() if args.agent_log_dir else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
