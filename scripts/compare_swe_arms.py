"""Paired comparison of SWE-smith eval arms on the same instances.

Each arm is one or more per-instance mapping files written by
eval_swe_smith.py / eval_advisor.py (--session_mapping_file), one file per
repeat (--num_runs 1). Instances are paired on instance_id across arms and
repeats:

- An instance that errored in any file of any arm, or is missing from any
  file, is dropped from every arm, so failures never count as unresolved.
- Per arm and instance, resolve and steps are averaged over that arm's repeats.
- Differences between arms are paired per instance; 95% CIs come from a
  bootstrap over instances.

With arms named weakalone, untrained and trained, the E7 pre-registered
checks are applied (see experiments/e07-route-then-advise/spec.md in hecate).

Usage:
    python scripts/compare_swe_arms.py \
        --arm weakalone=a1.json,a2.json,a3.json \
        --arm untrained=u1.json,u2.json,u3.json \
        [--arm trained=t1.json,t2.json,t3.json] \
        [--out comparison.json]
"""

import argparse
import json
import random
import sys
from itertools import combinations
from typing import Dict, List, Tuple

N_BOOTSTRAP = 10000
CHECK1_MIN_STEP_GAP = 2.0
CHECK3_MARGIN = 0.10


def load_mapping(path: str) -> Dict[str, dict]:
    """Load one mapping file as {instance_id: row}.

    Accepts the instance_id-keyed format and the older session_id-keyed one
    (rows carry instance_id either way).
    """
    with open(path) as f:
        raw = json.load(f)
    rows = {}
    for row in raw.values():
        iid = row["instance_id"]
        if iid in rows:
            raise ValueError(f"{path}: duplicate instance_id {iid}")
        rows[iid] = row
    return rows


def parse_arms(specs: List[str]) -> Dict[str, List[str]]:
    arms = {}
    for spec in specs:
        name, _, files = spec.partition("=")
        if not name or not files:
            raise ValueError(f"--arm must be NAME=file[,file...], got {spec!r}")
        if name in arms:
            raise ValueError(f"arm {name!r} given twice")
        arms[name] = [p for p in files.split(",") if p]
    return arms


def kept_and_dropped(
    arms: Dict[str, List[Dict[str, dict]]], files: Dict[str, List[str]]
) -> Tuple[List[str], Dict[str, str]]:
    """Instances usable in every file of every arm, and why the rest were dropped."""
    all_ids = set()
    for repeats in arms.values():
        for rows in repeats:
            all_ids |= rows.keys()

    dropped = {}
    for iid in sorted(all_ids):
        for name, repeats in arms.items():
            for k, rows in enumerate(repeats):
                where = f"{name} {files[name][k]}"
                if iid not in rows:
                    dropped[iid] = f"missing in {where}"
                elif rows[iid].get("error"):
                    dropped[iid] = f"errored in {where}: {rows[iid]['error']}"
                if iid in dropped:
                    break
            if iid in dropped:
                break
    kept = sorted(all_ids - dropped.keys())
    return kept, dropped


def per_instance_means(
    repeats: List[Dict[str, dict]], kept: List[str]
) -> Dict[str, Tuple[float, float]]:
    """{instance_id: (mean resolved, mean steps)} over this arm's repeats."""
    out = {}
    for iid in kept:
        resolved = [1.0 if rows[iid]["resolved"] else 0.0 for rows in repeats]
        steps = [float(rows[iid]["total_steps"]) for rows in repeats]
        out[iid] = (sum(resolved) / len(resolved), sum(steps) / len(steps))
    return out


def bootstrap_ci(diffs: List[float], seed: int = 0) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOTSTRAP)
    )
    return means[int(0.025 * N_BOOTSTRAP)], means[int(0.975 * N_BOOTSTRAP) - 1]


def paired(a: Dict, b: Dict, kept: List[str], idx: int) -> dict:
    """Mean of (a - b) per instance for metric idx (0 resolve, 1 steps), with CI."""
    diffs = [a[iid][idx] - b[iid][idx] for iid in kept]
    lo, hi = bootstrap_ci(diffs)
    return {"mean": sum(diffs) / len(diffs), "ci95": [lo, hi]}


def arm_summary(repeats: List[Dict[str, dict]], means: Dict, kept: List[str]) -> dict:
    resolved_steps = [
        float(rows[iid]["total_steps"])
        for rows in repeats
        for iid in kept
        if rows[iid]["resolved"]
    ]
    return {
        "repeats": len(repeats),
        "resolve_rate": sum(m[0] for m in means.values()) / len(kept),
        "mean_steps": sum(m[1] for m in means.values()) / len(kept),
        "mean_steps_resolved": (
            sum(resolved_steps) / len(resolved_steps) if resolved_steps else None
        ),
    }


def e7_checks(diffs: Dict[str, dict]) -> dict:
    """Apply the E7 pre-registered checks to paired point estimates."""
    checks = {}
    au = diffs.get("weakalone-untrained")
    if au:
        step_gap = au["steps"]["mean"]  # alone - untrained; positive = untrained faster
        resolve_drop = au["resolve"]["mean"]  # alone - untrained; positive = loss
        if step_gap < CHECK1_MIN_STEP_GAP:
            verdict = "stop"
        elif resolve_drop > 0:
            verdict = "pass"
        else:
            verdict = "partial (no trade-off to recover)"
        checks["check1_tradeoff"] = {
            "verdict": verdict,
            "untrained_steps_below_alone": step_gap,
            "untrained_resolve_below_alone": resolve_drop,
        }
    ta = diffs.get("trained-weakalone")
    tu = diffs.get("trained-untrained")
    if ta and tu:
        checks["check2_speedup"] = {
            "pass": ta["steps"]["mean"] < 0,
            "trained_steps_minus_alone": ta["steps"]["mean"],
            "trained_steps_minus_untrained": tu["steps"]["mean"],
        }
        checks["check3_resolve"] = {
            "pass": tu["resolve"]["mean"] > 0
            and ta["resolve"]["mean"] >= -CHECK3_MARGIN,
            "trained_minus_untrained": tu["resolve"]["mean"],
            "trained_minus_alone": ta["resolve"]["mean"],
            "trained_minus_alone_ci95_lower (secondary)": ta["resolve"]["ci95"][0],
        }
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", action="append", required=True)
    parser.add_argument("--out", help="Write the full comparison as JSON here")
    args = parser.parse_args(argv)

    files = parse_arms(args.arm)
    arms = {name: [load_mapping(p) for p in paths] for name, paths in files.items()}
    if len({len(r) for r in arms.values()}) > 1:
        print("WARNING: arms have different numbers of repeats", file=sys.stderr)

    kept, dropped = kept_and_dropped(arms, files)
    if not kept:
        print("No instance is usable in every file of every arm.", file=sys.stderr)
        return 1

    means = {name: per_instance_means(r, kept) for name, r in arms.items()}
    summary = {name: arm_summary(arms[name], means[name], kept) for name in arms}

    order = [n for n in ("weakalone", "untrained", "trained") if n in arms]
    order += sorted(n for n in arms if n not in order)
    # E7 pairs in the direction the checks read them; any other arms pairwise.
    e7_pairs = [
        ("weakalone", "untrained"),
        ("trained", "untrained"),
        ("trained", "weakalone"),
    ]
    pairs = [(a, b) for a, b in e7_pairs if a in arms and b in arms]
    pairs += [
        (a, b)
        for a, b in combinations(order, 2)
        if (a, b) not in pairs and (b, a) not in pairs
    ]
    diffs = {
        f"{a}-{b}": {
            "resolve": paired(means[a], means[b], kept, 0),
            "steps": paired(means[a], means[b], kept, 1),
        }
        for a, b in pairs
    }

    result = {
        "n_kept": len(kept),
        "dropped": dropped,
        "arms": summary,
        "paired_diffs": diffs,
        "e7_checks": e7_checks(diffs),
    }

    print(f"Instances kept: {len(kept)}; dropped: {len(dropped)}")
    for iid, why in dropped.items():
        print(f"  dropped {iid}: {why}")
    print("\nArm         repeats  resolve  steps  steps(resolved)")
    for name in order:
        s = summary[name]
        sr = s["mean_steps_resolved"]
        sr_text = f"{sr:.1f}" if sr is not None else "-"
        print(
            f"{name:<11} {s['repeats']:>7}  {s['resolve_rate']:>6.1%}  "
            f"{s['mean_steps']:>5.1f}  {sr_text:>15}"
        )
    print("\nPaired difference (A-B)    resolve [95% CI]              steps [95% CI]")
    for key, d in diffs.items():
        r, st = d["resolve"], d["steps"]
        print(
            f"{key:<26} {r['mean']:+6.1%} [{r['ci95'][0]:+.1%}, {r['ci95'][1]:+.1%}]"
            f"   {st['mean']:+5.1f} [{st['ci95'][0]:+.1f}, {st['ci95'][1]:+.1f}]"
        )
    if result["e7_checks"]:
        print("\nE7 pre-registered checks (point estimates):")
        print(json.dumps(result["e7_checks"], indent=2))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
