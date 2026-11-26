#!/usr/bin/env bash
set -euo pipefail

# Appendix G uses the exact 4k candidates and frozen router from the main policy grid.
[[ -r "${ANSWER_TOKENIZER:?Set ANSWER_TOKENIZER to the local answer-model tokenizer.json}" ]] || { echo "Cannot read ANSWER_TOKENIZER" >&2; exit 2; }
results=data/results/appendix
longmem=data/longmemeval/longmemeval_s.json
mkdir -p "$results"
python3 -m experiments.exp_rebuttal prepare-audit --data "$longmem" \
  --run-dir data/results/paper_controls --output "$results/frozen_4k.jsonl"
python3 -m experiments.exp_rebuttal rescore --answers "$results/frozen_4k.jsonl" --output "$results/scored_4k.jsonl"
python3 -m experiments.rebuttal_analysis replay "$results/scored_4k.jsonl" > "$results/selectors.json"
python3 -m experiments.rebuttal_analysis summarize "$results/scored_4k.jsonl" --judge independent_correct > "$results/frozen_router.json"

for experiment in joint token views history locomo
do
  data="$longmem"
  options=()
  case "$experiment" in
    token|history) options=(--tokenizer "$ANSWER_TOKENIZER") ;;
    locomo) data=data/locomo/locomo_test.json ;;
  esac
  directory="$results/$experiment"
  python3 -m experiments.exp_rebuttal generate "$experiment" --data "$data" --output "$directory" "${options[@]}"
  answers="$directory/answers.jsonl"
  field=is_correct
  if [[ "$experiment" == joint ]]; then
    python3 -m experiments.exp_rebuttal rescore --answers "$answers" --output "$directory/scored.jsonl"
    answers="$directory/scored.jsonl"
    field=independent_correct
  fi
  python3 -m experiments.rebuttal_analysis summarize "$answers" --judge "$field" > "$directory/summary.json"
done
