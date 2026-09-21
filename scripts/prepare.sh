#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$python_bin" scripts/check_portable.py
"$python_bin" -m trimodal_joint.cli prepare --config configs/noise_aug_v1.yaml "$@"
