#!/usr/bin/env bash
set -euo pipefail

# Table 13 and Figure 4: Track A and Track B masking.
python3 -m experiments.exp_masking_n500 \
  --data_path data/longmemeval/longmemeval_s.json \
  --model deepseek-v4-flash --judge_model deepseek-v4-flash \
  --split-input data/results/paper_controls/raw/ratio50_b4000.jsonl \
  --budget 4000 --max_samples 500 \
  --conditions dual_track,dual_track_mask_a,dual_track_mask_b \
  --output_dir data/results/masking

python3 scripts/aggregate_masking_n500.py \
  --input_dir data/results/masking \
  --output data/results/masking/table13.json

# Table 14: answer-span coverage for both tracks.
python3 -m experiments.exp_orthogonality \
  --correctness_file data/results/main_deepseek/budget_sweep_longmemeval_deepseek-v4-flash.json \
  --strategy dual_track --budget 4000 \
  --data_path data/longmemeval/longmemeval_s.json \
  --model deepseek-v4-flash \
  --output data/results/coverage/table14.json

# Table 15 uses the paired main-sweep contexts.
python3 scripts/aggregate_paper_controls.py \
  --run-dir data/results/paper_controls \
  --audit-plan data/results/judge_validation_matched_control/audit_plan.json \
  --output data/results/paper_controls/tables.json
