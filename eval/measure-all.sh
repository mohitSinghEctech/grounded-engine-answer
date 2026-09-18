#!/bin/bash
# Three groups of three runs, one behaviour change per group, each
# compared against the baseline it actually builds on. Written as a
# script rather than typed nine times so the configuration for each
# group is recorded next to its results.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="eval/runs/measure.log"
: > "$LOG"

say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

wait_healthy() {
  for _ in $(seq 1 45); do
    curl -sf http://localhost:8080/health >/dev/null 2>&1 && return 0
    sleep 2
  done
  say "!! service never became healthy"
  return 1
}

group() {
  local name=$1; shift
  say "=== group ${name} | config: $* ==="

  # shellcheck disable=SC2086
  env "$@" docker compose up -d tax-agent >>"$LOG" 2>&1
  wait_healthy || return 1

  # Prove the container really has the config before spending 30 minutes.
  say "    container: PIPELINE=$(docker compose exec -T tax-agent printenv PIPELINE 2>/dev/null) \
RETRY=$(docker compose exec -T tax-agent printenv GRAPH_RETRY_ON_FABRICATION 2>/dev/null) \
AGENT=$(docker compose exec -T tax-agent printenv GRAPH_AGENT_ON_COMPARISON 2>/dev/null)"

  for r in 1 2 3; do
    say "    ${name} run ${r} starting"
    .venv/bin/python eval/run.py --delay 3 --out "eval/runs/${name}-r${r}.csv" >>"$LOG" 2>&1
    say "    ${name} run ${r} done"
  done
}

say "### measuring three changes, three runs each"

group g1-graph PIPELINE=graph
group g2-retry PIPELINE=graph GRAPH_RETRY_ON_FABRICATION=true
group g3-agent PIPELINE=graph GRAPH_AGENT_ON_COMPARISON=true

say "### comparisons"

compare() {
  say "--- $1 vs $2 ---"
  .venv/bin/python eval/compare.py \
    --label "$1" eval/runs/"$1"-r*.csv \
    --label "$2" eval/runs/"$2"-r*.csv >>"$LOG" 2>&1
}

# Parity first: the port must not have changed behaviour before the two
# branches are judged against it.
say "--- linear vs g1-graph (parity) ---"
.venv/bin/python eval/compare.py \
  --label linear eval/runs/t3-progress-r*.csv \
  --label graph eval/runs/g1-graph-r*.csv >>"$LOG" 2>&1

compare g1-graph g2-retry
compare g1-graph g3-agent

# Leave the service in the default configuration, so nobody inherits a
# flag they did not set.
env PIPELINE=graph docker compose up -d tax-agent >>"$LOG" 2>&1

say "### all done"
