#!/usr/bin/env bash
set -euo pipefail

# Table 23 and Figure 5: LoCoMo on the same fixed set of 200 questions.
python3 -m experiments.exp_budget_sweep \
  --benchmark locomo --data_path data/locomo/locomo_test.json \
  --model deepseek-v4-flash --judge_model deepseek-v4-flash \
  --max_samples 200 --seeds 42 --shuffle \
  --budgets 500 1000 2000 4000 \
  --strategies dual_track \
  --output_dir data/results/locomo_split

python3 -m experiments.exp_budget_sweep \
  --benchmark locomo --data_path data/locomo/locomo_test.json \
  --model deepseek-v4-flash --judge_model deepseek-v4-flash \
  --max_samples 200 --seeds 42 --shuffle \
  --budgets 500 1000 2000 4000 \
  --strategies bm25_chunk_trackb \
  --output_dir data/results/locomo_trackb

python3 -m experiments.exp_budget_sweep \
  --benchmark locomo --data_path data/locomo/locomo_test.json \
  --model deepseek-v4-flash --judge_model deepseek-v4-flash \
  --max_samples 200 --seeds 42 --shuffle \
  --budgets 1000 2000 4000 \
  --strategies recent_sessions bm25_session session_summary embedding_chunk \
  --output_dir data/results/locomo_references
