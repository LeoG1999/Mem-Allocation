#!/usr/bin/env bash
set -euo pipefail

python3 -m experiments.exp_paper_controls reuse-main \
  --main data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json \
  --trackb data/results/trackb_deepseek/trackb_fullbudget_results.json

# Tables 2 and 6: chunk size and budget utilization.
python3 -m experiments.exp_paper_controls generate \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --budgets 1000 2000 4000 6000 \
  --conditions split256 trackb256 split512 trackb512 \
  --packing-mode whole_unit_skip \
  --output-dir data/results/paper_controls

# Table 3: use MiniLM for both retrieval views.
python3 -m experiments.exp_dense_matched \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --budgets 1000 2000 4000 6000 \
  --output-dir data/results/dense_matched
