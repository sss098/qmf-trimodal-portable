"""Protect formal_v1 numerical behavior while cleaning implementation details."""

import tarfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archives/formal_v1_code_f25138d975a7.tar.gz"


def archived_module(filename):
    with tarfile.open(ARCHIVE) as archive:
        source = archive.extractfile(f"src/trimodal_joint/{filename}").read()
    module = types.ModuleType("formal_reference")
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def test_cleanup_preserves_emg_parameters_outputs_and_gradients():
    from trimodal_joint.emg_model import FTTransformerRaw

    original = archived_module("emg_model.py").FTTransformerRaw
    args = dict(n_channels=6, window_size=100, n_classes=2, feature_dim=144)
    torch.manual_seed(17)
    reference = original(**args).eval()
    current = FTTransformerRaw(**args).eval()
    current.load_state_dict(reference.state_dict(), strict=True)
    x, f = torch.randn(2, 100, 6), torch.randn(2, 144)
    expected, actual = reference(x, f), current(x, f)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    for (n, p), (old_n, old_p) in zip(current.named_parameters(), reference.named_parameters()):
        assert n == old_n
        torch.testing.assert_close(p.grad, old_p.grad, rtol=0, atol=0)


def test_cleanup_preserves_reference_features():
    from trimodal_joint.emg_features import extract_features

    original = archived_module("emg_features.py").extract_features
    windows = np.random.default_rng(17).normal(size=(3, 100, 6)).astype(np.float32)
    for x in [*windows, np.zeros((100, 6), dtype=np.float32)]:
        np.testing.assert_array_equal(extract_features(x, 1000), original(x, 1000))


def test_default_qmf_loss_preserves_archived_outputs_gradients_and_history():
    from trimodal_joint.fusion import fuse
    from trimodal_joint.losses import QualityHistory, total_loss

    old = archived_module("losses.py")
    current_history, old_history = QualityHistory(6), old.QualityHistory(6)
    initial = torch.arange(18, dtype=torch.float64).reshape(6, 3) / 10
    current_history.sums[:] = initial
    old_history.sums[:] = initial
    torch.manual_seed(901)
    z = torch.randn(4, 3, 2, requires_grad=True)
    args = (
        z,
        fuse(z, "qmf"),
        "qmf",
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([1.7, 0.7]),
        3,
        torch.tensor([0, 3, 2, 5]),
    )
    expected = old.total_loss(*args, old_history)
    actual = total_loss(*args, current_history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z, retain_graph=True)[0],
        torch.autograd.grad(expected, z)[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(current_history.sums, old_history.sums, rtol=0, atol=0)
