"""Evaluate a student in the Codex harness on SWE-Smith, alone or with the advisor.

E7 Codex harness pilot (experiments/e07-route-then-advise/spec.md in hecate).
Each instance runs `codex exec` inside its SWE-smith container. Codex talks to
OpenRouter through a local CodexProxy (codex_proxy.py), which counts steps,
enforces the step cap and, with --advisor_model, adds advice. The patch is the
container repo's `git diff`, scored with the same compute_score as
eval_swe_smith. The mapping file has the same per-instance rows, so
scripts/compare_swe_arms.py joins it with the other arms.

Example (alone):
    python -m advisor_models.swe_smith.eval_codex \
        --model openai/gpt-5.4-nano \
        --codex_package ~/codex/pkg \
        --data_file data/swe_smith/validation_jd__tenacity.json \
        --run_dir runs/exp-07_arm-codexnano_split-tenacityval_seed-0 \
        --max_workers 4

Add --advisor_model openai/local-advisor-model --advisor_api_base http://...:8000/v1
for the advisor arm. Needs OPENROUTER_API_KEY and docker on this machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

from .codex_proxy import (
    RETRY_BACKOFF_MAX_S,
    RETRY_BACKOFF_S,
    RETRY_STATUSES,
    UPSTREAM_ATTEMPTS,
    CodexProxy,
    RunState,
    history_to_messages,
)
from .config import (
    ADVISOR_TEMPLATE,
    build_advisor_observation,
    compute_score,
)
from .eval_advisor import ADVISOR_MAX_TOKENS, ADVISOR_OBSERVATION_CHARS, _tail_text

REPO_DIR = "/testbed"
CODEX_HOME = "/opt/codexhome"
# The codex-package release (codex, codex-code-mode-host, rg) goes here.
CODEX_DIR = "/opt/codex"
# Off: anything that reaches outside the container or adds agents. Web search
# would let the agent look the fix up.
DISABLED_FEATURES = [
    "multi_agent", "goals", "image_generation", "computer_use", "in_app_browser",
    "browser_use", "browser_use_external", "apps", "memories",
]
STEP_LIMIT = 40
CODEX_TIMEOUT_S = 45 * 60
TASK_INSTRUCTION = (
    "Fix the issue above in the repository at /testbed. Edit non-test source "
    "files only; do not modify the tests."
)
# OpenRouter list, 27 Oct 2026, USD per 1M tokens. Used only when a response
# carries no cost of its own.
PRICES = {
    "openai/gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "openai/gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "openai/gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "openai/gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
}
# Per-student Codex settings (spec, "Students"). None leaves Codex's own
# default for the model; a model Codex has no metadata for needs its context
# window, or Codex compacts against a fallback window. Flash-Lite ran without thinking in E7,
# so its reasoning settings are stripped.
STUDENTS = {
    "openai/gpt-5.4-nano": {"reasoning_effort": "medium", "drop_reasoning": False, "context_window": 400000},
    "openai/gpt-6-luna": {"reasoning_effort": None, "drop_reasoning": False, "context_window": None},
    "google/gemini-2.5-flash-lite": {
        "reasoning_effort": None, "drop_reasoning": True, "context_window": 1048576,
        "catalog_base": "gpt-5.5",
    },
    "openai/gpt-5-nano": {
        "reasoning_effort": "medium", "drop_reasoning": False, "context_window": 400000,
        "catalog_base": "gpt-5.5",
    },
}
# A model with no Codex metadata gets Codex's fallback toolset: no apply_patch,
# an exec_command with a sandbox-escalation option that approval "never"
# refuses, and request_user_input. "catalog_base" instead gives the student a
# catalog entry copied from that built-in model (gpt-5.5: plain Responses, no
# Code Mode), with the overrides below. apply_patch stays freeform, the only
# type Codex 0.164 accepts. A student without reasoning (drop_reasoning) also
# gets NO_REASONING_OVERRIDES; one with reasoning keeps gpt-5.5's reasoning
# fields, limited to the efforts every GPT-5 model takes.
NO_REASONING_OVERRIDES = {
    "supported_reasoning_levels": [],
    "supports_reasoning_summaries": False,
    "supports_reasoning_summary_parameter": False,
    "supports_reasoning_effort_updates": False,
}
REASONING_EFFORTS = ("low", "medium", "high")
CATALOG_OVERRIDES = {
    "supports_search_tool": False,
    "input_modalities": ["text"],
    "supports_image_detail_original": False,
    "prefer_websockets": False,
    "service_tiers": [],
    "additional_speed_tiers": [],
    "upgrade": None,
    "visibility": "list",
}


def codex_model_name(model: str) -> str:
    """The name Codex runs under: OpenRouter's id without the provider prefix,
    so Codex uses its own metadata for the model."""
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def builtin_catalog(codex_bin: str) -> Dict[str, Any]:
    """Codex's built-in model catalog, read from the binary it ships in."""
    data = Path(codex_bin).read_bytes()
    i = data.find(b'"models": [')
    if i < 0:
        raise RuntimeError(f"no model catalog found in {codex_bin}")
    start = data.rfind(b"{", 0, i)
    catalog, _ = json.JSONDecoder().raw_decode(data[start:start + 4_000_000].decode("utf-8", "replace"))
    return catalog


def student_catalog(model: str, codex_bin: str) -> Optional[Dict[str, Any]]:
    """A one-model catalog for a student with a catalog_base, else None."""
    settings = STUDENTS.get(model, {})
    base = settings.get("catalog_base")
    if not base:
        return None
    entries = [m for m in builtin_catalog(codex_bin)["models"] if m["slug"] == base]
    if not entries:
        raise RuntimeError(f"Codex catalog has no {base}")
    entry = dict(entries[0], **CATALOG_OVERRIDES)
    if settings.get("drop_reasoning"):
        entry.update(NO_REASONING_OVERRIDES)
    else:
        entry["supported_reasoning_levels"] = [
            level for level in entry["supported_reasoning_levels"] if level["effort"] in REASONING_EFFORTS
        ]
        entry["default_reasoning_level"] = settings["reasoning_effort"]
    entry.update(
        slug=codex_model_name(model),
        display_name=model,
        description=f"{model} via OpenRouter (entry copied from {base})",
        context_window=settings["context_window"],
        max_context_window=settings["context_window"],
    )
    return {"models": [entry]}


def codex_config(model: str, port: int, catalog_path: Optional[str] = None) -> str:
    settings = STUDENTS.get(model, {})
    effort = settings.get("reasoning_effort")
    effort_line = f'model_reasoning_effort = "{effort}"\n' if effort else ""
    if catalog_path:
        effort_line += f'model_catalog_json = "{catalog_path}"\n'
    if settings.get("context_window"):
        effort_line += f"model_context_window = {settings['context_window']}\n"
    features = "".join(f"{name} = false\n" for name in DISABLED_FEATURES)
    return f"""model = "{codex_model_name(model)}"
model_provider = "hecate"
web_search = "disabled"
{effort_line}
[features]
{features}
[model_providers.hecate]
name = "hecate-proxy"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
env_key = "HECATE_PROXY_KEY"
"""


def _docker(*args: str, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {proc.stderr.strip()[:500]}")
    return proc


class Advisor:
    """The trained advisor, called the way eval_advisor calls it."""

    def __init__(self, model: str, api_base: str, temperature: float, top_p: float,
                 top_k: int, min_p: float, no_think: bool):
        self.model = model
        self.api_base = api_base
        self.sampling = dict(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p)
        self.no_think = no_think

    def complete(self, messages: List[Dict[str, str]]) -> str:
        extra_body = {}
        if self.no_think:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        response = litellm.completion(
            model=self.model,
            api_base=self.api_base,
            messages=messages,
            max_tokens=ADVISOR_MAX_TOKENS,
            extra_body=extra_body,
            **self.sampling,
        )
        return response.choices[0].message.content or ""

    def for_instance(self, instance: Dict[str, Any]):
        """A per-run callback for RunState: initial advice, then follow-ups."""
        task_prompt = {
            "role": "user",
            "content": ADVISOR_TEMPLATE.format(
                problem_statement=instance["problem_statement"], repo=instance["repo"]
            ),
        }
        history: List[str] = []

        def advise(n_request: int, items: List[Dict[str, Any]]) -> Optional[str]:
            if not history:
                guidance = self.complete([task_prompt])
            else:
                observation, _, _, _ = build_advisor_observation(
                    agent_history=history_to_messages(items),
                    rolling_summary="",
                    last_summarized_step=0,
                    current_step=n_request - 1,
                )
                guidance = self.complete(
                    [
                        task_prompt,
                        {"role": "assistant", "content": history[-1]},
                        {"role": "user", "content": _tail_text(observation, ADVISOR_OBSERVATION_CHARS)},
                    ]
                )
            history.append(guidance)
            # The agent sees the advice without the think block, as in eval_advisor.
            if "</think>" in guidance:
                guidance = guidance.split("</think>")[1].strip()
            return guidance

        advise.history = history
        return advise


def run_instance(instance: Dict[str, Any], args, advisor: Optional[Advisor], api_key: str) -> Dict[str, Any]:
    instance_id = instance["instance_id"]
    inst_dir = Path(args.run_dir) / "instances" / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    container = f"codex-{uuid.uuid4().hex[:10]}"
    row: Dict[str, Any] = {
        "instance_id": instance_id,
        "session_id": container,
        "eval_id": None,
        "diff": "",
        "resolved": False,
        "status": "error",
        "total_steps": 0,
        "agent_cost": 0.0,
        "error": None,
    }
    advise = advisor.for_instance(instance) if advisor else None
    state = RunState(
        step_limit=STEP_LIMIT,
        advisor=advise,
        log_path=str(inst_dir / "proxy.jsonl"),
        first_request_path=str(inst_dir / "first_request.json"),
        drop_reasoning=STUDENTS.get(args.model, {}).get("drop_reasoning", False),
    )
    try:
        _docker("run", "-d", "--rm", "--network", "host", "--name", container,
                "-w", REPO_DIR, instance["image_name"], "sleep", "4h")
        _docker("exec", container, "git", "fetch", "--all", timeout=120, check=False)
        _docker("exec", container, "git", "checkout", instance_id)
        commit = _docker("exec", container, "git", "rev-parse", "HEAD").stdout.strip()
        _docker("cp", args.codex_package, f"{container}:{CODEX_DIR}")

        with CodexProxy(
            state, api_key, prices=PRICES.get(args.model, {}), upstream_model=args.model
        ) as proxy:
            _docker("exec", container, "mkdir", "-p", CODEX_HOME)
            catalog_path = None
            if args.catalog:
                catalog_path = f"{CODEX_HOME}/catalog.json"
                _docker("cp", args.catalog, f"{container}:{catalog_path}")
            cfg = inst_dir / "config.toml"
            cfg.write_text(codex_config(args.model, proxy.port, catalog_path))
            _docker("cp", str(cfg), f"{container}:{CODEX_HOME}/config.toml")
            task = f"{instance['problem_statement']}\n\n{TASK_INSTRUCTION}"
            events = open(inst_dir / "codex_events.jsonl", "w")
            stderr = open(inst_dir / "codex_stderr.log", "w")
            proc = subprocess.Popen(
                ["docker", "exec", "-w", REPO_DIR,
                 "-e", f"CODEX_HOME={CODEX_HOME}", "-e", "HECATE_PROXY_KEY=unused",
                 "-e", f"PATH={CODEX_DIR}/codex-path:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                 container, f"{CODEX_DIR}/bin/codex", "exec", "--json", "--strict-config",
                 "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
                 task],
                stdin=subprocess.DEVNULL, stdout=events, stderr=stderr,
            )
            start = time.time()
            timed_out = False
            while proc.poll() is None:
                if state.cap_hit:
                    break
                if time.time() - start > CODEX_TIMEOUT_S:
                    timed_out = True
                    break
                time.sleep(2)
            if proc.poll() is None:
                _docker("exec", container, "pkill", "-f", "codex", check=False)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            events.close()
            stderr.close()

        summary = state.summary()
        patch = _docker("exec", container, "git", "diff", commit).stdout
        row.update(
            diff=patch,
            total_steps=summary["steps"],
            agent_cost=summary["cost"],
            codex_exit=proc.returncode,
            proxy=summary,
        )
        if advise is not None:
            row["advisor_calls"] = summary["advisor_calls"]
            row["advisor_guidance_history"] = list(advise.history)

        # A run that never got a model response is an error, not an unresolved
        # attempt, as in eval_swe_smith.
        if summary["usage"]["output_tokens"] == 0:
            raise RuntimeError(
                f"no successful model response (upstream errors: {summary['upstream_errors']})"
            )
        # A request that still failed after the proxy's retries, or a dropped
        # stream, ends the Codex run: an infrastructure error, not a failed fix.
        if summary["failed_requests"]:
            raise RuntimeError(
                f"upstream request failed after retries ({summary['failed_requests']} requests)"
            )
        failed = turn_failures(inst_dir / "codex_events.jsonl")
        if failed and not summary["cap_hit"]:
            raise RuntimeError(f"Codex turn failed: {failed[-1][:300]}")
        if summary["stateful_requests"]:
            raise RuntimeError("Codex sent previous_response_id; history is not resent")
        if timed_out:
            raise RuntimeError(f"Codex exceeded {CODEX_TIMEOUT_S}s")

        reward, eval_id, info = compute_score(patch, instance)
        agent_status = "cap" if summary["cap_hit"] else f"exit {proc.returncode}"
        status = info.split("Status: ")[1].split(",")[0] if "Status:" in info else "unknown"
        row.update(
            eval_id=eval_id,
            resolved=reward > 0.5,
            status=f"{status} (agent: {agent_status}, steps: {summary['steps']})",
        )
    except Exception as e:
        row["error"] = str(e)
        row["status"] = "error"
    finally:
        _docker("rm", "-f", container, check=False)
        (inst_dir / "row.json").write_text(json.dumps(row, indent=2))
    return row


def turn_failures(events_path: Path) -> List[str]:
    """Messages of Codex turn.failed events."""
    out = []
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.failed":
            out.append(str((event.get("error") or {}).get("message", "")))
    return out


def load_instances(path: str, num_samples: Optional[int]) -> List[Dict[str, Any]]:
    instances = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                instances.append(json.loads(row["reward_spec"]["ground_truth_json"]))
    return instances[:num_samples] if num_samples else instances


def main():
    parser = argparse.ArgumentParser(description="Evaluate a student in Codex on SWE-Smith")
    parser.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-5.4-nano")
    parser.add_argument("--codex_package", required=True,
                        help="Unpacked codex-package-x86_64-unknown-linux-musl release dir")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--advisor_model", default=None, help="e.g. openai/local-advisor-model")
    parser.add_argument("--advisor_api_base", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--no_think", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set")
    litellm.drop_params = True

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pkg = Path(args.codex_package)
    version = subprocess.run(
        [str(pkg / "bin" / "codex"), "--version"], capture_output=True, text=True
    ).stdout.strip()
    sha256 = {
        name: hashlib.sha256((pkg / "bin" / name).read_bytes()).hexdigest()
        for name in ("codex", "codex-code-mode-host")
    }
    catalog = student_catalog(args.model, str(pkg / "bin" / "codex"))
    args.catalog = None
    if catalog is not None:
        args.catalog = str(run_dir / "catalog.json")
        Path(args.catalog).write_text(json.dumps(catalog, indent=1))
    meta = {
        "model": args.model,
        "codex_version": version,
        "codex_sha256": sha256,
        "student_settings": STUDENTS.get(args.model),
        "catalog_overrides": None if catalog is None else dict(
            CATALOG_OVERRIDES, **(NO_REASONING_OVERRIDES if STUDENTS[args.model].get("drop_reasoning") else {})
        ),
        "catalog_sha256": hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest() if args.catalog else None,
        "upstream_retry": {
            "attempts": UPSTREAM_ATTEMPTS, "statuses": sorted(RETRY_STATUSES),
            "backoff_s": RETRY_BACKOFF_S, "backoff_max_s": RETRY_BACKOFF_MAX_S,
        },
        "task_instruction": TASK_INSTRUCTION,
        "codex_config": codex_config(args.model, 0, f"{CODEX_HOME}/catalog.json" if catalog else None),
        "step_limit": STEP_LIMIT,
        "advisor_model": args.advisor_model,
        "advisor_api_base": args.advisor_api_base,
        "advisor_sampling": None if not args.advisor_model else {
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "min_p": args.min_p, "no_think": args.no_think,
        },
        "data_file": args.data_file,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    advisor = None
    if args.advisor_model:
        advisor = Advisor(args.advisor_model, args.advisor_api_base, args.temperature,
                          args.top_p, args.top_k, args.min_p, args.no_think)

    instances = load_instances(args.data_file, args.num_samples)
    rows: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_instance, inst, args, advisor, api_key): inst for inst in instances}
        for fut in as_completed(futures):
            row = fut.result()
            rows[row["instance_id"]] = row
            mark = "+" if row["resolved"] else ("!" if row["error"] else "-")
            print(f"{mark} {row['instance_id']} steps={row['total_steps']} "
                  f"cost=${row['agent_cost']:.4f} {row['status']} {row['error'] or ''}", flush=True)
            (run_dir / "session_mapping.json").write_text(json.dumps(rows, indent=2))

    ordered = {inst["instance_id"]: rows[inst["instance_id"]] for inst in instances}
    (run_dir / "session_mapping.json").write_text(json.dumps(ordered, indent=2))
    n = len(ordered)
    resolved = sum(r["resolved"] for r in ordered.values())
    errors = sum(1 for r in ordered.values() if r["error"])
    cost = sum(r["agent_cost"] for r in ordered.values())
    print(f"\nResolved {resolved}/{n}, errors {errors}, total cost ${cost:.2f}")


if __name__ == "__main__":
    main()
