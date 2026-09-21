#!/usr/bin/env bash
# Python 3.11 is the tested interpreter; install in a fresh local virtual environment.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-tested.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_portable.py
