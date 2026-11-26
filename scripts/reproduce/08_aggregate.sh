#!/usr/bin/env bash
set -euo pipefail

# Aggregate the packing, audit, coverage, and allocation controls.
python3 scripts/aggregate_paper_controls.py \
  --run-dir data/results/paper_controls \
  --audit-plan data/results/judge_validation_matched_control/audit_plan.json \
  --output data/results/paper_controls/tables.json

# Remaining paper tables.
python3 scripts/aggregate_paper_tables.py \
  --results-dir data/results \
  --output-dir data/results/paper_tables "$@"

# Figures 2--6.
python3 scripts/plot_paper_figures.py \
  --tables-dir data/results/paper_tables \
  --output-dir data/results/paper_figures
