"""Reproduce clean-data paired intervals and internal-branch metrics for the main report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from qmf_diagnostics import bootstrap_indices, interval, metric_arrays
from scipy.special import softmax

from trimodal_joint.io import save_json, sha256


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "reports/noise_aug_analysis"
    output.mkdir(exist_ok=True)
    frames, point, boot, sources = {}, {}, {}, {}
    anchor = None
    for run in ["formal_v1", "noise_aug_v1"]:
        for method in ["equal_late", "qmf"]:
            data = []
            for seed in [42, 43, 44]:
                relative = Path("artifacts/runs") / run / method / f"seed{seed}/oof_predictions.csv"
                path = project / relative
                frame = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
                keys = frame[["patient_id", "label", "fold"]]
                if anchor is None:
                    anchor = keys
                pd.testing.assert_frame_equal(keys, anchor)
                if len(frame) != 85 or frame.patient_id.nunique() != 85:
                    raise ValueError("Expected 85 unique OOF patients")
                data.append(frame)
                sources[str(relative)] = sha256(path)
            frames[run, method] = data
    y = anchor.label.to_numpy(int)
    indices = bootstrap_indices(y, 10000, 20260920)
    for key, data in frames.items():
        probability = np.mean([d.probability.to_numpy() for d in data], axis=0)
        point[key] = metric_arrays(y, probability[None])
        boot[key] = metric_arrays(y[indices], probability[indices])
    rows = []
    for a, b in [
        (("noise_aug_v1", "qmf"), ("formal_v1", "qmf")),
        (("noise_aug_v1", "qmf"), ("noise_aug_v1", "equal_late")),
    ]:
        for metric in point[a]:
            low, high = interval(boot[a][metric] - boot[b][metric])
            rows.append(
                dict(
                    comparison=f"{'/'.join(a)} minus {'/'.join(b)}",
                    metric=metric,
                    estimate=float(point[a][metric][0] - point[b][metric][0]),
                    ci95_low=low,
                    ci95_high=high,
                )
            )
    pd.DataFrame(rows).to_csv(output / "paired_clean_ci.csv", index=False)
    rows = []
    for run in ["formal_v1", "noise_aug_v1"]:
        for modality in ["us", "emg", "table"]:
            probability = np.mean(
                [
                    softmax(d[[f"{modality}_score_0", f"{modality}_score_1"]].to_numpy(), axis=1)[
                        :, 1
                    ]
                    for d in frames[run, "qmf"]
                ],
                axis=0,
            )
            values = metric_arrays(y, probability[None])
            rows.append(
                dict(run=run, modality=modality, **{k: float(v[0]) for k, v in values.items()})
            )
    pd.DataFrame(rows).to_csv(output / "internal_branch_metrics.csv", index=False)
    save_json(
        output / "metadata.json",
        dict(
            patients=85,
            bootstrap=10000,
            seed=20260920,
            scope="fixed OOF predictions; patient-paired bootstrap",
            source_sha256=sources,
        ),
    )
    print(json.dumps({"output": str(output), "patients": 85}))


if __name__ == "__main__":
    main()
