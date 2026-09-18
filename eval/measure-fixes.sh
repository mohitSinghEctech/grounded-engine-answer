#!/bin/bash
# Two groups of three, measuring what the first nine runs asked for.
#
# The baseline here is g3-agent, not the linear pipeline: that group is
# the best configuration measured so far (graph + agent branch), so it is
# what these changes have to improve on. Comparing against anything older
# would credit these fixes with the agent branch's win.
#
# g4 holds the code still and changes only the CORPUS - guidance pages
# re-labelled away from an extrapolable sequence, and maps_to_1961 filled
# for 319 of 552 sections instead of 4. Both reached the index through
# `make reindex`, so g4 against g3-agent isolates them.
#
# g5 then adds the corrected retry rule on top of g4. One change per
# group, which is the rule the NOT_IN_CORPUS attempt paid to learn.
set -u

cd "$(dirname "$0")/.." || exit 1
LOG="eval/runs/measure-fixes.log"
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

say "### measuring the corpus fixes, then the corrected retry rule"

group g4-corpus PIPELINE=graph GRAPH_AGENT_ON_COMPARISON=true
group g5-retry  PIPELINE=graph GRAPH_AGENT_ON_COMPARISON=true GRAPH_RETRY_ON_FABRICATION=true

say "### comparisons"

compare() {
  say "--- $1 vs $2 ---"
  .venv/bin/python eval/compare.py \
    --label "$1" eval/runs/"$1"-r*.csv \
    --label "$2" eval/runs/"$2"-r*.csv 2>&1 | tee -a "$LOG"
}

# The corpus changes against the configuration they were built on top of.
compare g3-agent g4-corpus
# The retry rule against the corpus it now runs on.
compare g4-corpus g5-retry

# Leave the service where the measurement says it should be.
env PIPELINE=graph GRAPH_AGENT_ON_COMPARISON=true \
  docker compose up -d tax-agent >>"$LOG" 2>&1

say "### all done"
