#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  "") ;;
  --with-appendix) [[ -r "${ANSWER_TOKENIZER:?Set ANSWER_TOKENIZER to the local answer-model tokenizer.json}" ]] ;;
  *) echo "Usage: bash scripts/reproduce/run_all.sh [--with-appendix]" >&2; exit 2 ;;
esac

for script in \
  01_main.sh \
  02_chunk_and_dense.sh \
  03_allocation_router.sh \
  04_full_audit.sh \
  04_judges_and_length.sh \
  05_masking_and_coverage.sh \
  06_extended_budget.sh \
  07_locomo.sh
do
  bash "scripts/reproduce/${script}"
done

if [[ "${1:-}" == --with-appendix ]]; then
  bash scripts/reproduce/07_appendix.sh
fi
bash scripts/reproduce/08_aggregate.sh "$@"
