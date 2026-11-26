#!/usr/bin/env bash
set -euo pipefail

# Table 9: stratified subset audit of the matched pair.
python3 scripts/judge_validation_matched_control.py \
  --data_path data/longmemeval/longmemeval_s.json \
  --judge_model gpt-4o-mini \
  --budgets 1000 4000 6000 --n_per_budget 100 \
  --output_dir data/results/judge_validation_matched_control

# Table 10: full allocation-grid audit on the same 100 questions.
python3 -m experiments.exp_paper_controls rejudge \
  --inputs \
    data/results/paper_controls/raw/ratio25_b4000.jsonl \
    data/results/paper_controls/raw/ratio33_b4000.jsonl \
    data/results/paper_controls/raw/ratio67_b4000.jsonl \
    data/results/paper_controls/raw/ratio75_b4000.jsonl \
    data/results/paper_controls/raw/router_b4000.jsonl \
    data/results/paper_controls/raw/tracka1024_b4000.jsonl \
  --judge-model gpt-4o-mini --prompt-mode questionless \
  --sample-ids-file data/results/judge_validation_matched_control/audit_plan.json \
  --tag audit100 \
  --output-dir data/results/paper_controls/rejudge

# Table 11: question-inclusive check by the main judge without regenerating answers.
python3 -m experiments.exp_paper_controls rejudge \
  --inputs data/results/paper_controls/raw/ratio50_b*.jsonl data/results/paper_controls/raw/trackb1024_b*.jsonl \
  --judge-model deepseek-v4-flash --prompt-mode question_inclusive \
  --output-dir data/results/paper_controls/rejudge

# Table 12: match Track B to the realized Split length.
python3 -m experiments.exp_paper_controls length-match \
  --split-inputs data/results/paper_controls/raw/ratio50_b*.jsonl \
  --data-path data/longmemeval/longmemeval_s.json \
  --packing-mode whole_unit_skip \
  --output-dir data/results/paper_controls/length_matched
