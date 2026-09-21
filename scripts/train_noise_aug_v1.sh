#!/usr/bin/env bash
# Matched augmented equal fusion and QMF, in a separate run directory.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$python_bin" -m trimodal_joint.cli train \
  --config configs/noise_aug_v1.yaml --methods equal_late,qmf \
  --seeds 42,43,44 --folds 0,1,2,3,4 --name portable_noise_aug_v1 --resume "$@"
