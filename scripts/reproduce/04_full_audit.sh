#!/usr/bin/env bash
set -euo pipefail

# Table 8: independently judge the two sets of answers saved by 01_main.sh.
python3 scripts/judge_validation_matched_control.py \
  --data_path data/longmemeval/longmemeval_s.json \
  --judge_model gpt-4o-mini \
  --budgets 1000 2000 4000 6000 --n_per_budget 500 \
  --output_dir data/results/judge_validation_full
