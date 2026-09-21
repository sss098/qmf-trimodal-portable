#!/usr/bin/env bash
# Train both ranking ablations sequentially on the same GPU.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

"$python_bin" -m trimodal_joint.cli train \
  --config configs/base.yaml --methods qmf \
  --seeds 42,43,44 --folds 0,1,2,3,4 \
  --rank-mode positive_margin --rank-weight 0 \
  --name portable_rank0_v1 --resume

"$python_bin" -m trimodal_joint.cli train \
  --config configs/base.yaml --methods qmf \
  --seeds 42,43,44 --folds 0,1,2,3,4 \
  --rank-mode original --rank-weight 0.1 \
  --name portable_rank_original_v1 --resume
