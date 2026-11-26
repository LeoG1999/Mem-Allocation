#!/usr/bin/env bash
set -euo pipefail

# Tables 10 and 17–19 and Figure 3: static allocation, router, and oracle.
python3 -m experiments.exp_paper_controls generate \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --budgets 1000 2000 4000 6000 \
  --conditions ratio25 ratio33 \
  --packing-mode whole_unit_skip \
  --output-dir data/results/paper_controls

python3 -m experiments.exp_paper_controls generate \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --budgets 2000 4000 6000 \
  --conditions ratio75 \
  --packing-mode whole_unit_skip \
  --output-dir data/results/paper_controls

python3 -m experiments.exp_paper_controls generate \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --budgets 4000 \
  --conditions ratio67 \
  --packing-mode whole_unit_skip \
  --output-dir data/results/paper_controls

python3 -m experiments.exp_paper_controls route \
  --data-path data/longmemeval/longmemeval_s.json \
  --max-samples 500 --model deepseek-v4-flash \
  --output data/results/paper_controls/routes/route_b.jsonl

python3 -m experiments.exp_paper_controls generate \
  --data-path data/longmemeval/longmemeval_s.json \
  --conditions router --max-samples 500 --budgets 4000 \
  --routes data/results/paper_controls/routes/route_b.jsonl \
  --output-dir data/results/paper_controls
