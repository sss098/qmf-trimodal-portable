import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def module():
    path = Path(__file__).resolve().parents[1] / "reports/qmf_diagnostics.py"
    spec = importlib.util.spec_from_file_location("qmf_diagnostics", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_vector_metrics_match_sklearn_with_ties():
    m = module()
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.1, 0.9])
    result = m.metric_arrays(y, p[None])
    assert result["roc_auc"][0] == roc_auc_score(y, p)
    assert result["accuracy"][0] == 0.75
    assert result["sensitivity"][0] == 1
    assert result["specificity"][0] == 0.5


def test_stratified_bootstrap_and_identical_pair_difference():
    m = module()
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.6, 0.4, 0.7])
    idx = m.bootstrap_indices(y, 100, 42)
    assert np.all(y[idx].sum(1) == 3)
    result = m.metric_arrays(y[idx], p[idx])
    assert np.all(result["roc_auc"] - result["roc_auc"] == 0)


def test_quality_shift_preserves_branch_probability_but_changes_fusion():
    m = module()
    z = np.array([[[0.0, 1.0], [1.0, 0.0], [0.2, 0.1]]])
    shifted = z.copy()
    shifted[:, 0] += 5
    np.testing.assert_allclose(m.softmax(z[:, 0], axis=-1), m.softmax(shifted[:, 0], axis=-1))
    assert not np.allclose(m.qmf_probability(z), m.qmf_probability(shifted))


def test_constant_correlation_is_missing():
    m = module()
    assert m.correlations(np.ones(4), np.arange(4)) == (None, None)
