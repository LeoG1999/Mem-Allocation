#!/usr/bin/env bash
set -euo pipefail

# Table 16: 12k, 24k, and full history.
python3 -m experiments.exp_extended_budget \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --budgets 12000 24000 \
  --packing-mode whole_unit_skip \
  --output-dir data/results/extended_budget
