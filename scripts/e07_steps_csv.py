"""E7: turn a SkyRL console train.log into one CSV row per GRPO step.

Reads the per-step metrics dict ("Step N:"), the timing dict that follows it, and
the "E7 rollouts" line of that step. Safe to re-run on a growing log; a run that
was resumed after a restart repeats some steps, and the last occurrence wins.

Usage: python scripts/e07_steps_csv.py runs/<run>/train.log runs/<run>/steps.csv [steps_per_epoch]
"""

import ast
import csv
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
PREFIX = re.compile(r"^\(\w+ pid=\d+\)\s?")
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
ROLL = re.compile(
    r"E7 rollouts: n=(\d+) not_stop=(\d+) rewarded=(\d+) rewarded_zeroed=(\d+) raw_reward_mean=([\d.]+)"
)

COLUMNS = [
    "step", "epoch", "finished_utc", "n", "not_stop", "rewarded", "rewarded_zeroed",
    "raw_reward_mean", "reward/avg_raw_reward", "reward/avg_pass_at_8",
    "generate/avg_num_tokens", "generate/max_num_tokens", "generate/frac_stop_not_stop",
    "generate/avg_tokens_non_zero_rewards", "generate/avg_tokens_zero_rewards",
    "policy/policy_loss", "policy/policy_kl", "policy/policy_entropy", "policy/raw_grad_norm",
    "policy/ppo_clip_ratio", "loss/avg_raw_advantages_abs",
    "timing/generate", "timing/fwd_logprobs_values_reward", "timing/policy_train", "timing/step",
]


def parse_dict(lines):
    text = " ".join(lines)
    text = re.sub(r"<[^>]*>", "None", text)
    return ast.literal_eval(text)


def main(log_path, out_path, steps_per_epoch=14):
    lines = [PREFIX.sub("", ANSI.sub("", l.rstrip("\n"))) for l in open(log_path, errors="replace")]
    rows, last_roll, last_ts = {}, None, ""
    i = 0
    while i < len(lines):
        line = lines[i]
        m = TS.match(line)
        if m:
            last_ts = m.group(1)
        r = ROLL.search(line)
        if r:
            last_roll = r.groups()
        s = re.search(r"tracking:log:\d+ - Step (\d+):\s*$", line)
        if s and i + 1 < len(lines) and lines[i + 1].startswith("{"):
            step = int(s.group(1))
            block, j = [], i + 1
            while j < len(lines):
                block.append(lines[j])
                if lines[j].rstrip().endswith("}"):
                    break
                j += 1
            try:
                metrics = parse_dict(block)
            except (ValueError, SyntaxError):
                i = j + 1
                continue
            row = rows.setdefault(step, {"step": step})
            row.update({k: v for k, v in metrics.items() if v is not None})
            if "timing/step" in metrics:
                row["finished_utc"] = last_ts
            elif last_roll:
                row.update(dict(zip(
                    ["n", "not_stop", "rewarded", "rewarded_zeroed", "raw_reward_mean"], last_roll)))
            row["epoch"] = round(step / steps_per_epoch, 2)
            i = j + 1
            continue
        i += 1
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for step in sorted(rows):
            w.writerow(rows[step])
    print(f"{len(rows)} steps -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 14)
