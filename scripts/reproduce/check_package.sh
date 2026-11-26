#!/usr/bin/env bash
set -euo pipefail

# Check the code package without calling any model API.
python3 -m compileall -q src experiments scripts tests
python3 tests/run_checks.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/smoke_imports.py

for script in scripts/reproduce/*.sh
do
  bash -n "${script}"
done

echo "Code package checks passed"
