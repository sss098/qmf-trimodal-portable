import numpy as np
import torch


def test_package_exists():
    import importlib.util

    assert importlib.util.find_spec("trimodal_joint.fusion") is not None


def test_fusion_identity_and_gradients():
    from trimodal_joint.fusion import combine_evidence, fuse

    alpha = torch.tensor([[3.0, 2.0]])
    torch.testing.assert_close(combine_evidence(alpha, torch.ones_like(alpha)), alpha)
    for method in ["equal_late", "qmf", "tmc"]:
        z = torch.tensor([[[1.0, 2.0], [0.5, 1.0], [2.0, -1.0]]], requires_grad=True)
        out = fuse(z, method)
        torch.testing.assert_close(out["probabilities"].sum(1), torch.ones(1))
        (-out["probabilities"][:, 1].log()).mean().backward()
        assert torch.isfinite(z.grad).all()
        assert (z.grad.abs().sum(-1) > 0).all()
    z = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    torch.testing.assert_close(fuse(z, "equal_late")["scores"], torch.tensor([[3.0, 4.0]]))


def test_rank_direction():
    from trimodal_joint.losses import QualityHistory

    h = QualityHistory(2)
    h.update(torch.tensor([0, 1]), torch.tensor([[0.1] * 3, [0.9] * 3]))
    good = torch.tensor([[1.0] * 3, [0.0] * 3], requires_grad=True)
    bad = 1 - good
    assert h.rank(good, torch.tensor([0, 1])) < h.rank(bad, torch.tensor([0, 1]))
    empty = QualityHistory(2)
    assert empty.rank(good, torch.tensor([0, 1])).item() == 0


def test_table_training_only():
    from trimodal_joint.tabular import fit_table, transform_table

    x = np.array([[1.0, np.nan], [3.0, np.nan], [100.0, 5.0]])
    state = fit_table(x, np.array([0, 1]), ["a", "b"])
    assert state["names"] == ["a"]
    np.testing.assert_allclose(transform_table(x, state)[:2, 0], [-1, 1])


def test_emg_network_and_features():
    from trimodal_joint.emg_features import extract_features
    from trimodal_joint.emg_model import FTTransformerRaw

    model = FTTransformerRaw(n_channels=6, window_size=100, n_classes=2, feature_dim=144)
    assert sum(p.numel() for p in model.parameters()) == 572274
    x = np.random.default_rng(42).normal(size=(100, 6)).astype("float32")
    f = extract_features(x, 1000)
    assert f.shape == (144,) and np.isfinite(f).all()
    y = model(torch.from_numpy(x[None]), torch.from_numpy(f[None]))
    assert y.shape == (1, 2)
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_evidential_loss_finite():
    from trimodal_joint.losses import evidential_loss

    alpha = torch.ones(2, 2, requires_grad=True)
    value = evidential_loss(alpha, torch.tensor([0, 1]), 10, torch.ones(2))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(alpha.grad).all()


def test_tmc_matches_explicit_dempster_shafer():
    from trimodal_joint.fusion import combine_evidence

    a = torch.tensor([[2.0, 7.0], [3.0, 4.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[5.0, 2.0], [9.0, 2.0]], dtype=torch.float64, requires_grad=True)
    sa, sb = a.sum(-1, keepdim=True), b.sum(-1, keepdim=True)
    ba, bb = (a - 1) / sa, (b - 1) / sb
    ua, ub = 2 / sa, 2 / sb
    conflict = ba[:, 0:1] * bb[:, 1:2] + ba[:, 1:2] * bb[:, 0:1]
    belief = (ba * bb + ba * ub + bb * ua) / (1 - conflict)
    uncertainty = ua * ub / (1 - conflict)
    reference = belief * (2 / uncertainty) + 1
    result = combine_evidence(a, b)
    torch.testing.assert_close(result, reference)
    ga = torch.autograd.grad(result.sum(), a, retain_graph=True)[0]
    gb = torch.autograd.grad(reference.sum(), a)[0]
    torch.testing.assert_close(ga, gb)


def test_qmf_quality_path_is_not_detached():
    from trimodal_joint.fusion import fuse

    z = torch.tensor([[[1.0, 2.0], [0.2, 0.3], [0.4, 0.5]]], requires_grad=True)
    out = fuse(z, "qmf")["scores"].sum()
    gradient = torch.autograd.grad(out, z)[0]
    q = torch.logsumexp(z, -1) / 10
    expected = q.unsqueeze(-1) + z.sum(-1, keepdim=True) * z.softmax(-1) / 10
    torch.testing.assert_close(gradient, expected)
