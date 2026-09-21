"""Controlled perturbations must preserve pairing and clean predictions."""

import numpy as np

from trimodal_joint.robustness import emg_noise, noise_for, perturb


def test_noise_is_paired_and_does_not_use_model_identity():
    a = noise_for((100, 6), "patient-a", "emg", 0)
    np.testing.assert_array_equal(a, noise_for((100, 6), "patient-a", "emg", 0))
    assert not np.array_equal(a, noise_for((100, 6), "patient-b", "emg", 0))
    assert not np.array_equal(a, noise_for((100, 6), "patient-a", "emg", 1))


def test_zero_level_and_nested_severity():
    x = np.ones((100, 6), dtype=np.float32)
    noise = noise_for(x.shape, "patient-a", "table", 0)
    np.testing.assert_array_equal(perturb(x, noise, 0), x)
    np.testing.assert_allclose(perturb(x, noise, 2) - x, 2 * (perturb(x, noise, 1) - x), atol=1e-6)
    np.testing.assert_array_equal(x, np.ones_like(x))


def test_emg_overlapping_windows_share_noise_samples():
    noise = emg_noise("patient-a", 0).reshape(5, 331, 100, 6)
    np.testing.assert_array_equal(noise[:, :-1, 30:], noise[:, 1:, :-30])


def test_summary_uses_patient_pairs_and_keeps_noise_repeats_separate(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd

    reports = Path(__file__).resolve().parents[1] / "reports"
    monkeypatch.syspath_prepend(str(reports))
    spec = importlib.util.spec_from_file_location(
        "robustness_report", reports / "qmf_robustness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for setting in ["equal_late", "original"]:
        for seed in [42, 43]:
            for repeat in range(2):
                for level in [0, 1]:
                    for pid, label in enumerate([0, 0, 1, 1]):
                        # Repetition one reverses ranks; averaging probabilities would give ties.
                        p = [0.1, 0.4, 0.6, 0.9][pid]
                        if repeat == 1 and level == 1:
                            p = 1 - p
                        rows.append(
                            dict(
                                setting=setting,
                                seed=seed,
                                modality="us",
                                level=level,
                                repeat=repeat,
                                patient_id=pid,
                                label=label,
                                probability=p,
                                quality=1 - level,
                                branch_loss=level,
                            )
                        )
    args = SimpleNamespace(
        output=tmp_path,
        noise_repeats=2,
        bootstrap=20,
        settings=["equal_late", "original"],
        smoke=False,
    )
    module.summarize(pd.DataFrame(rows), args)
    changes = pd.read_csv(tmp_path / "paired_changes_ci.csv")
    paired = changes[changes.comparison == "change_minus_equal_late_change"]
    np.testing.assert_array_equal(paired[["estimate", "ci95_low", "ci95_high"]], 0)
    quality = pd.read_csv(tmp_path / "quality_changes_ci.csv")
    assert set(quality[(quality.level == 1) & (quality.variable == "quality")].change) == {-1}
