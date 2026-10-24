#!/bin/bash

# E7: one-step GRPO smoke on 4 issues, then (if it passes) the full 18-epoch run.
#
# Run on gpu-advisor (a2-ultragpu-2g) inside tmux, after the agent and eval servers
# on hecate-worker are up on the same commit:
#
#   tmux new -s grpo
#   set -a; source ~/advisor-models/.env; set +a
#   bash scripts/e07_smoke_then_train.sh
#
# Required env: AGENT_SERVER_URL, EVAL_SERVER_URL (worker internal IP).
# Optional env:
#   AGENT_MODEL     student (default openrouter/google/gemini-2.5-flash-lite)
#   AUTO_LAUNCH     1 = start the full run if the smoke passes (default 1)
#   SMOKE_ONLY      1 = stop after the smoke (default 0)
#   MAX_NOT_STOP    smoke fails if more than this fraction of rollouts ended
#                   at the length cap (default 0.9)
#   MAX_ERR_FRAC    smoke fails if more than this fraction of rollouts logged a
#                   student/agent error (default 0.25)
#   MIN_CREDIT      USD; checked only if OPENROUTER_API_KEY is set (default 450)
#   TRAIN_RUN_NAME  default exp-07_arm-trained_split-tenacitytrain_seed-0
#   SMOKE_RUN_NAME  default exp-07_arm-trainsmoke_split-tenacitytrain_seed-0
#
# Smoke pass rules (all must hold):
#   - the smoke command exits 0 and the log has no CUDA out-of-memory
#   - at least one rollout earned a raw environment reward > 0 (before the
#     zero_reward_on_non_stop rule), i.e. the student and eval path score work
#   - error lines / rollouts <= MAX_ERR_FRAC
#   - rollouts ended by the length cap / rollouts <= MAX_NOT_STOP
# The smoke never writes a checkpoint (CKPT_INTERVAL=-1), so the full run cannot
# resume from it. Results go to runs/<SMOKE_RUN_NAME>/smoke_summary.json.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${AGENT_SERVER_URL:?Set AGENT_SERVER_URL, e.g. http://<worker-int-ip>:8081}"
: "${EVAL_SERVER_URL:?Set EVAL_SERVER_URL, e.g. http://<worker-int-ip>:8082}"
export AGENT_SERVER_URL EVAL_SERVER_URL
export AGENT_MODEL="${AGENT_MODEL:-openrouter/google/gemini-2.5-flash-lite}"
export LOGGER="${LOGGER:-console}"

AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
MAX_NOT_STOP="${MAX_NOT_STOP:-0.9}"
MAX_ERR_FRAC="${MAX_ERR_FRAC:-0.25}"
MIN_CREDIT="${MIN_CREDIT:-450}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-exp-07_arm-trained_split-tenacitytrain_seed-0}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-exp-07_arm-trainsmoke_split-tenacitytrain_seed-0}"

fail() { echo "E7 STOP: $*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "A100")
[ "$GPUS" -eq 2 ] || fail "expected 2 A100s, found $GPUS (do not train on one GPU)"
for url in "$AGENT_SERVER_URL" "$EVAL_SERVER_URL"; do
    curl -sf -m 10 "$url/health" >/dev/null || fail "server not healthy: $url"
done
if pgrep -f "vllm serve" >/dev/null; then fail "vllm serve is running; stop it first"; fi
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    LEFT=$(curl -s -m 20 -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        https://openrouter.ai/api/v1/credits |
        python3 -c "import json,sys; d=json.load(sys.stdin)['data']; print(d['total_credits']-d['total_usage'])")
    python3 -c "import sys; sys.exit(0 if float('$LEFT') >= float('$MIN_CREDIT') else 1)" ||
        fail "OpenRouter credit \$$LEFT is below MIN_CREDIT \$$MIN_CREDIT"
    echo "OpenRouter credit remaining: \$$LEFT"
fi
echo "commit $(git rev-parse --short HEAD) $(git status --short | grep -vc '^??') modified tracked files"

# --- smoke -----------------------------------------------------------------
SMOKE_DIR="runs/$SMOKE_RUN_NAME"
mkdir -p "$SMOKE_DIR"
git rev-parse --short HEAD > "$SMOKE_DIR/CODE_SHA"
echo "E7 smoke: 4 issues x 8 samples, 1 step, no checkpoint -> $SMOKE_DIR/smoke.log"
DATA_DIR="$REPO_ROOT/data/swe_smith_smoke" EPOCHS=1 CKPT_INTERVAL=-1 RUN_NAME="$SMOKE_RUN_NAME" \
    bash scripts/train_swe_smith_1k.sh 2>&1 | tee "$SMOKE_DIR/smoke.log"
SMOKE_EXIT=${PIPESTATUS[0]}

python3 - "$SMOKE_DIR/smoke.log" "$SMOKE_EXIT" "$MAX_NOT_STOP" "$MAX_ERR_FRAC" \
    > "$SMOKE_DIR/smoke_summary.json" <<'EOF'
import json, re, sys
log, exit_code, max_not_stop, max_err = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
text = open(log, errors="replace").read()
m = re.findall(r"E7 rollouts: n=(\d+) not_stop=(\d+) rewarded=(\d+) rewarded_zeroed=(\d+) raw_reward_mean=([\d.]+)", text)
n, not_stop, rewarded, zeroed, raw = (int(m[-1][0]), int(m[-1][1]), int(m[-1][2]), int(m[-1][3]), float(m[-1][4])) if m else (0, 0, 0, 0, 0.0)
err_pat = r"Error creating remote session|Error in remote agent step|Agent step failed|Agent step timeout|Error executing agent step|HTTP 402|\b402\b.*credit"
errors = len(re.findall(err_pat, text))
oom = bool(re.search(r"CUDA out of memory|OutOfMemoryError", text))
reasons = []
if exit_code != 0: reasons.append(f"smoke exited {exit_code}")
if oom: reasons.append("CUDA out of memory")
if not m: reasons.append("no 'E7 rollouts' line in the log")
if m and rewarded == 0: reasons.append("no rollout earned a raw reward > 0")
if n and errors / n > max_err: reasons.append(f"error lines {errors}/{n} > {max_err}")
if n and not_stop / n > max_not_stop: reasons.append(f"length-capped {not_stop}/{n} > {max_not_stop}; revisit max_turns before training")
print(json.dumps({
    "pass": not reasons, "reasons": reasons, "exit_code": exit_code, "oom": oom,
    "rollouts": n, "not_stop": not_stop, "frac_not_stop": not_stop / n if n else None,
    "rewarded_raw": rewarded, "rewarded_zeroed_by_length": zeroed, "raw_reward_mean": raw,
    "error_lines": errors,
}, indent=2))
EOF
cat "$SMOKE_DIR/smoke_summary.json"
python3 -c "import json,sys; sys.exit(0 if json.load(open('$SMOKE_DIR/smoke_summary.json'))['pass'] else 1)" ||
    fail "smoke did not pass; see $SMOKE_DIR/smoke_summary.json"

echo "E7 smoke PASSED"
[ "$SMOKE_ONLY" = "1" ] && { echo "SMOKE_ONLY=1, not starting the full run"; exit 0; }
[ "$AUTO_LAUNCH" = "1" ] || { echo "AUTO_LAUNCH=0, not starting the full run"; exit 0; }

# --- full run --------------------------------------------------------------
TRAIN_DIR="runs/$TRAIN_RUN_NAME"
mkdir -p "$TRAIN_DIR"
git rev-parse --short HEAD > "$TRAIN_DIR/CODE_SHA"
echo "E7 full run: 18 epochs -> $TRAIN_DIR/train.log"
RUN_NAME="$TRAIN_RUN_NAME" bash scripts/train_swe_smith_1k.sh 2>&1 | tee -a "$TRAIN_DIR/train.log"
exit "${PIPESTATUS[0]}"
