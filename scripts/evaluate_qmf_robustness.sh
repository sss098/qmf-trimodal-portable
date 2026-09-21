#!/usr/bin/env bash
# Evaluate existing final checkpoints; no training or optimizer updates.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$python_bin" reports/qmf_robustness.py "$@"
