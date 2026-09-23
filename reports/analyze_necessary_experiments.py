"""Patient-paired analysis of the five HPC experiments and retained main models."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qmf_diagnostics import bootstrap_indices, metric_arrays

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/hpc2_20260923/analysis"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ["us", "emg", "table", "us_table", "no_noise"]:
        method = "equal_late" if name in ["us", "emg", "table"] else "qmf"
        paths[name] = ROOT / f"artifacts/hpc2_20260923/runs/necessary_{name}_v1/{method}"
    for name, method in [("main", "qmf"), ("equal", "equal_late")]:
        paths[name] = ROOT / f"artifacts/runs/noise_aug_v1/{method}"
    anchor = None
    estimates, draws, sources, rows = {}, {}, {}, []
    for name, path in paths.items():
        source = path / "ensemble_predictions.csv"
        frame = pd.read_csv(source).sort_values("patient_id").reset_index(drop=True)
        key = frame[["patient_id", "fold", "label"]]
        if anchor is None:
            anchor = key
            y = frame.label.to_numpy()
            indices = bootstrap_indices(y, 10000, 20260923)
        pd.testing.assert_frame_equal(key, anchor)
        assert len(frame) == frame.patient_id.nunique() == 85
        summary = json.loads((path / "summary.json").read_text())
        assert summary["complete_five_fold"] and not summary["smoke"]
        p = frame.probability.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        estimates[name] = metric_arrays(y, p)
        draws[name] = metric_arrays(y[indices], p[indices])
        sources[name] = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        for metric, values in draws[name].items():
            lo, hi = np.quantile(values, [0.025, 0.975])
            rows.append(
                dict(
                    model=name,
                    metric=metric,
                    estimate=float(estimates[name][metric]),
                    lower=lo,
                    upper=hi,
                )
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "metrics_ci.csv", index=False)
    comparisons = []
    for control in ["table", "us_table", "no_noise", "equal", "us", "emg"]:
        for metric in draws["main"]:
            difference = draws["main"][metric] - draws[control][metric]
            lo, hi = np.quantile(difference, [0.025, 0.975])
            comparisons.append(
                dict(
                    comparison=f"main - {control}",
                    metric=metric,
                    difference=float(estimates["main"][metric] - estimates[control][metric]),
                    lower=lo,
                    upper=hi,
                )
            )
    pd.DataFrame(comparisons).to_csv(OUTPUT / "paired_differences_ci.csv", index=False)
    (OUTPUT / "provenance.json").write_text(
        json.dumps(
            dict(patients=85, bootstrap_repeats=10000, seed=20260923, sources=sources), indent=2
        )
    )
    print(
        pd.DataFrame(comparisons).query("metric in ['roc_auc', 'accuracy']").to_string(index=False)
    )


if __name__ == "__main__":
    main()
