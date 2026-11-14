"""E10: check swesmith's grader, through swesmith_arm, on all 105 Tenacity issues.

For each issue, swesmith's own run_evaluation grades two predictions in the
ARM image:
  fix  the gold patch (is_gold=True, so swesmith reverses the bug patch)
  bug  an empty patch (the issue's bug branch as is)
Pass, per issue, the same conditions G0b checked (g0b_check.py):
  fix: resolved, and every FAIL_TO_PASS and PASS_TO_PASS test passes
  bug: not resolved, no FAIL_TO_PASS test passes, every PASS_TO_PASS test passes
and g0b_arm64.jsonl must record the issue as ok.

Instances come from advisor-models' data files, exactly as the eval server
receives them (ground_truth_json). No model, no training, no network needed
by the check itself; Docker and the ARM image are.

Run on the Spark (tmux, log to file), from any directory:
  SWESMITH_IMAGE_OVERRIDE=hecate/swesmith.arm64.jd_1776_tenacity.0d40e76f:e10 \
  ~/venvs/e10-eval/bin/python scripts/e10_check_swesmith_arm.py \
      --g0b ~/work/hecate/experiments/e10-ppo-vs-grpo-advisor/spark/g0b_arm64.jsonl \
      --out ~/e10-logs/swesmith_arm_check.jsonl

swesmith writes its per-issue logs under <workdir>/logs/run_evaluation/.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SPLITS = [("train", "train_jd__tenacity.json"), ("test", "validation_jd__tenacity.json")]


def load_instances(data_dir: Path) -> list[dict]:
    rows = []
    for split, name in SPLITS:
        for line in open(data_dir / name):
            if line.strip():
                gt = json.loads(json.loads(line)["reward_spec"]["ground_truth_json"])
                rows.append({"split": split, "instance": gt})
    return rows


def read_report(run_id: str, instance_id: str) -> dict:
    from swebench.harness.constants import LOG_REPORT, RUN_EVALUATION_LOG_DIR

    path = Path(RUN_EVALUATION_LOG_DIR) / run_id / instance_id / LOG_REPORT
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    return report.get(instance_id, report)


def tests(report: dict, kind: str, outcome: str) -> set:
    return set(report.get("tests_status", {}).get(kind, {}).get(outcome, []))


def check_one(row: dict, run_tag: str) -> dict:
    from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
    from swesmith.harness.eval import run_evaluation

    inst = row["instance"]
    iid = inst["instance_id"]
    f2p, p2p = set(inst["FAIL_TO_PASS"]), set(inst["PASS_TO_PASS"])
    out = {"split": row["split"], "instance_id": iid, "n_f2p": len(f2p), "n_p2p": len(p2p)}

    for arm, patch, is_gold in (("fix", inst["patch"], True), ("bug", "", False)):
        run_id = f"{run_tag}-{arm}"
        pred = {KEY_INSTANCE_ID: iid, KEY_MODEL: "e10_arm_check", KEY_PREDICTION: patch}
        result = run_evaluation(pred=pred, instance=dict(inst), run_id=run_id,
                                f2p_only=False, is_gold=is_gold)
        report = read_report(run_id, iid)
        out[f"{arm}_status"] = result.get("status")
        out[f"{arm}_resolved"] = bool(result.get("resolved"))
        if arm == "fix":
            out["fix_not_passing"] = sorted((f2p - tests(report, "FAIL_TO_PASS", "success"))
                                            | (p2p - tests(report, "PASS_TO_PASS", "success")))
        else:
            out["bug_f2p_passing"] = sorted(f2p & tests(report, "FAIL_TO_PASS", "success"))
            out["bug_p2p_not_passing"] = sorted(p2p - tests(report, "PASS_TO_PASS", "success"))
    return out


def judge(row: dict, g0b: dict) -> dict:
    why = []
    if row["fix_status"] != "completed" or row["bug_status"] != "completed":
        why.append(f"status fix={row['fix_status']} bug={row['bug_status']}")
    if not row["fix_resolved"]:
        why.append("fix not resolved")
    if row["bug_resolved"]:
        why.append("bug graded resolved")
    if row["fix_not_passing"]:
        why.append(f"{len(row['fix_not_passing'])} tests not passing with the fix")
    if row["bug_f2p_passing"]:
        why.append(f"{len(row['bug_f2p_passing'])} FAIL_TO_PASS tests pass on the bug")
    if row["bug_p2p_not_passing"]:
        why.append(f"{len(row['bug_p2p_not_passing'])} PASS_TO_PASS tests not passing on the bug")
    g = g0b.get(row["instance_id"])
    if g is None:
        why.append("missing from g0b_arm64.jsonl")
    elif not g.get("ok"):
        why.append("g0b_arm64.jsonl has ok=false")
    elif (g["n_f2p"], g["n_p2p"]) != (row["n_f2p"], row["n_p2p"]):
        why.append("test counts differ from g0b_arm64.jsonl")
    return {**row, "ok": not why, "why": why}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--g0b", required=True, help="g0b_arm64.jsonl from the hecate repo")
    ap.add_argument("--out", required=True, help="output JSON Lines, one row per issue")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "swe_smith"))
    ap.add_argument("--workdir", default=os.path.expanduser("~/e10-work/swesmith-arm-check"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N issues only (0 = all 105)")
    args = ap.parse_args()

    from advisor_models.swe_smith.swesmith_arm import describe, install

    if not install():
        print("SWESMITH_IMAGE_OVERRIDE is not set; refusing to run.", file=sys.stderr)
        return 2
    print(f"swesmith_arm: {describe()}", flush=True)

    g0b = {r["instance_id"]: r for r in map(json.loads, open(args.g0b)) if r}
    rows = load_instances(Path(args.data_dir))
    if args.limit:
        rows = rows[: args.limit]
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(args.workdir)
    run_tag = time.strftime("e10armcheck-%Y%m%d-%H%M%S")

    n_ok = 0
    with cf.ThreadPoolExecutor(args.workers) as ex, open(args.out, "w") as f:
        futures = [ex.submit(check_one, r, run_tag) for r in rows]
        for fut in cf.as_completed(futures):
            row = judge(fut.result(), g0b)
            n_ok += row["ok"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(("OK  " if row["ok"] else "BAD ") + f"{row['split']} {row['instance_id']}"
                  + ("" if row["ok"] else "  " + "; ".join(row["why"])), flush=True)
    print(f"SUMMARY {n_ok}/{len(rows)} OK", flush=True)
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
