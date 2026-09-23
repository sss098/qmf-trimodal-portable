#!/usr/bin/env bash
# Five matched experiments; no existing main-run outputs are overwritten.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$project_dir/.venv/bin/python}"
experiment="${1:-all}"
case "$experiment" in
  all) experiments=(us emg table us_table no_noise) ;;
  us|emg|table|us_table|no_noise) experiments=("$experiment") ;;
  *) echo "Usage: $0 [all|us|emg|table|us_table|no_noise]" >&2; exit 2 ;;
esac
for item in "${experiments[@]}"; do
  method=qmf
  case "$item" in us|emg|table) method=equal_late ;; esac
  "$python_bin" -m trimodal_joint.cli train \
    --config "configs/necessary_${item}_v1.yaml" --methods "$method" \
    --seeds 42,43,44 --folds 0,1,2,3,4 \
    --name "necessary_${item}_v1" --resume
done
