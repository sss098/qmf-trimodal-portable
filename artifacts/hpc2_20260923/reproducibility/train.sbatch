#!/usr/bin/env bash
#SBATCH --job-name=qmf-necessary
#SBATCH --partition=gpuA800
#SBATCH --account=a_jiang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHON_BIN="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
status_file="logs/job-${SLURM_JOB_ID}.status"
trap 'code=$?; printf "%s\n" "$code" > "$status_file"' EXIT
nvidia-smi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
"$PYTHON_BIN" -m pytest -q tests
bash scripts/train_necessary_experiments.sh all
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
for name in ['us','emg','table','us_table','no_noise']:
    method='equal_late' if name in ['us','emg','table'] else 'qmf'
    root=Path(f'artifacts/runs/necessary_{name}_v1')/method
    summary=json.loads((root/'summary.json').read_text())
    assert summary['complete_five_fold'] and summary['n']==85 and not summary['smoke']
    assert len(list(root.glob('seed*/fold*/model.pt')))==15
print('All five experiments completed: 75 fold models, 85 OOF patients per experiment.')
PY
