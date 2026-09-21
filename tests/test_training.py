import numpy as np
import torch


def test_split_has_no_patient_overlap():
    from trimodal_joint.training import split_indices

    records = [{"fold": i % 5} for i in range(85)]
    for f in range(5):
        train, valid, test = split_indices(records, f)
        assert len(train) == 51 and len(valid) == len(test) == 17
        assert set(train).isdisjoint(valid) and set(train).isdisjoint(test)
        assert set(valid).isdisjoint(test)


def test_diagnostics_are_patient_level():
    from trimodal_joint.evaluation import metrics, prediction_row
    from trimodal_joint.fusion import fuse

    z = torch.tensor([[[0.1, 0.9], [0.2, 0.6], [0.4, 0.5]]])
    for method in ["equal_late", "qmf", "tmc"]:
        rows = []
        for i, y in enumerate([0, 1]):
            record = {"patient_id": str(i), "fold": 0, "label": y}
            rows.append(prediction_row(record, z, fuse(z, method), method))
        assert all(0 <= r["probability"] <= 1 for r in rows)
        assert np.isfinite(metrics(rows)["log_loss"])


def test_all_losses_reach_all_modalities():
    from trimodal_joint.fusion import fuse
    from trimodal_joint.losses import QualityHistory, total_loss

    for method in ["equal_late", "qmf", "tmc"]:
        z = torch.randn(4, 3, 2, requires_grad=True)
        loss = total_loss(
            z,
            fuse(z, method),
            method,
            torch.tensor([0, 1, 1, 0]),
            torch.ones(2),
            1,
            torch.arange(4),
            QualityHistory(4),
        )
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
        assert (z.grad.abs().sum((0, 2)) > 0).all()
