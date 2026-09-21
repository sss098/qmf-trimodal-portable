#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
"$python_bin" -m trimodal_joint.cli train --config configs/base.yaml \
  --methods equal_late,qmf,tmc --seeds 42,43,44 --folds 0,1,2,3,4 \
  --name portable_base_v1 --resume "$@"
