#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin=""
if command -v python3.11 >/dev/null 2>&1; then
  python_bin="$(command -v python3.11)"
elif command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.11 is not installed; falling back to Python 3.12."
  python_bin="$(command -v python3.12)"
else
  echo "Python 3.11 or 3.12 is required." >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pre_commit install
.venv/bin/python -m ipykernel install --user --name transaction-forecasting --display-name 'Python (transaction-forecasting)'
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pre_commit run --all-files

echo "Setup complete. Activate with: source .venv/bin/activate"
