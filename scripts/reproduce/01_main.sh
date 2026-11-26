#!/usr/bin/env bash
set -euo pipefail

# Main LongMemEval-S comparison at 1k–6k (Tables 1, 5 and 7 and Figure 2).
config=configs/main_sweep.yaml
python3 -m experiments.exp_budget_sweep \
  --config "$config" \
  --model deepseek-v4-flash --judge_model deepseek-v4-flash \
  --data_path data/longmemeval/longmemeval_s.json \
  --max_samples 500 --seeds 42 --shuffle \
  --budgets 1000 2000 4000 6000 \
  --strategies recent_sessions bm25_session session_summary embedding_chunk bm25_chunk dual_track \
  --output_dir data/results/main_deepseek

python3 -m experiments.exp_trackb_fullbudget \
  --config "$config" \
  --model deepseek-v4-flash --judge_model deepseek-v4-flash \
  --data_path data/longmemeval/longmemeval_s.json \
  --max_samples 500 --seed 42 --budgets 1000 2000 4000 6000 \
  --packing_mode whole_unit_skip \
  --output_dir data/results/trackb_deepseek

python3 -m experiments.exp_budget_sweep \
  --config "$config" \
  --model Qwen/Qwen3.5-9B --judge_model deepseek-v4-flash \
  --data_path data/longmemeval/longmemeval_s.json \
  --max_samples 500 --seeds 42 --shuffle \
  --budgets 1000 2000 4000 6000 \
  --strategies recent_sessions bm25_session session_summary embedding_chunk bm25_chunk dual_track \
  --output_dir data/results/main_qwen

python3 -m experiments.exp_trackb_fullbudget \
  --config "$config" \
  --model Qwen/Qwen3.5-9B --judge_model deepseek-v4-flash \
  --data_path data/longmemeval/longmemeval_s.json \
  --max_samples 500 --seed 42 --budgets 1000 2000 4000 6000 \
  --packing_mode whole_unit_skip \
  --output_dir data/results/trackb_qwen

python3 scripts/paired_crossover_stats.py \
  --main data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json \
  --trackb data/results/trackb_deepseek/trackb_fullbudget_results.json \
  --bm25_chunk data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json \
  --budgets 1000 2000 4000 6000 \
  --output data/results/statistics/paired_crossover.json
