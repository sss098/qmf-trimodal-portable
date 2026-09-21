import pytest
import torch
from torch.nn import functional as F

from trimodal_joint.config import Config
from trimodal_joint.fusion import fuse
from trimodal_joint.losses import QualityHistory, total_loss


def test_original_ranking_matches_upstream_value_and_gradient():
    history = QualityHistory(4)
    history.sums[:] = torch.tensor([[0.2] * 3, [0.9] * 3, [0.2] * 3, [0.5] * 3])
    idx = torch.arange(4)
    q = torch.tensor([[0.1] * 3, [2.0] * 3, [0.4] * 3, [-0.3] * 3], requires_grad=True)
    h = history.sums.float()
    normalized = (h - h.min(0).values) / (h.max(0).values - h.min(0).values)
    target = (normalized - normalized.roll(-1, 0)).sign()
    margin = (normalized - normalized.roll(-1, 0)).abs()
    nonzero = target.clone()
    nonzero[nonzero == 0] = 1
    expected = sum(
        F.margin_ranking_loss(
            q[:, m], q.roll(-1, 0)[:, m] + margin[:, m] / nonzero[:, m], -target[:, m]
        )
        for m in range(3)
    )
    actual = history.rank(q, idx, mode="original")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, q, retain_graph=True)[0], torch.autograd.grad(expected, q)[0]
    )


def test_original_ranking_zero_history_is_finite():
    q = torch.randn(3, 3, requires_grad=True)
    value = QualityHistory(3).rank(q, torch.arange(3), mode="original")
    assert value.item() == 0
    value.backward()
    assert torch.isfinite(q.grad).all()


def test_zero_rank_keeps_dynamic_fusion_and_classification_gradients():
    z = torch.randn(4, 3, 2, requires_grad=True)
    y = torch.tensor([0, 1, 1, 0])
    w = torch.tensor([2.0, 1.0])
    out = fuse(z, "qmf")
    expected = F.cross_entropy(out["scores"], y, weight=w) + sum(
        F.cross_entropy(z[:, m], y, weight=w) for m in range(3)
    )
    actual = total_loss(
        z, out, "qmf", y, w, 1, torch.arange(4), None, rank_weight=0, rank_mode="original"
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z, retain_graph=True)[0], torch.autograd.grad(expected, z)[0]
    )


def test_rank_modes_have_different_margin_constraints():
    h = QualityHistory(2)
    h.sums[:] = torch.tensor([[0.0] * 3, [1.0] * 3])
    q = torch.zeros(2, 3)
    assert h.rank(q, torch.arange(2), mode="original").item() == 0
    assert h.rank(q, torch.arange(2), mode="positive_margin").item() == 3
    torch.testing.assert_close(
        h.rank(q, torch.arange(2)), h.rank(q, torch.arange(2), mode="positive_margin")
    )


def test_invalid_rank_mode_rejected():
    with pytest.raises(ValueError):
        Config(rank_mode="typo").validate()


def test_original_ranking_preserves_upstream_hinge_boundary_gradient():
    history = QualityHistory(2)
    history.sums[:] = torch.tensor([[0.0] * 3, [1.0] * 3])
    q = torch.tensor([[-1.0] * 3, [0.0] * 3], requires_grad=True)
    target = torch.tensor([-1.0, 1.0])
    expected = sum(
        F.margin_ranking_loss(q[:, m], q.roll(-1, 0)[:, m] + 1 / target, -target) for m in range(3)
    )
    actual = history.rank(q, torch.arange(2), mode="original")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, q, retain_graph=True)[0],
        torch.autograd.grad(expected, q)[0],
    )


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_invalid_rank_weight_rejected(weight):
    with pytest.raises(ValueError):
        Config(rank_weight=weight).validate()
