"""Exercise actual optimizer/checkpoint/RNG resume with a tiny injected network."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from trimodal_joint import training
from trimodal_joint.config import Config


class TinyModel(nn.Module):
    def __init__(self, table_dim, imagenet_weights):
        super().__init__()
        self.ultrasound = nn.Linear(2, 2)
        self.us_head = nn.Linear(2, 2)
        self.emg = nn.Sequential(nn.Dropout(0.2), nn.Linear(2, 2))
        self.tabular = nn.Linear(2, 2)

    def forward(self, image, windows, features, table):
        return torch.stack(
            [self.us_head(self.ultrasound(image)), self.emg(windows), self.tabular(table)], 1
        )

    def parameter_counts(self):
        return {n: sum(p.numel() for p in m.parameters()) for n, m in self.named_children()}


class TinyData:
    def __init__(self, cache, train):
        self.records = [{"patient_id": str(i), "label": i % 2} for i in range(6)]
        self.table = np.ones((6, 2), dtype=np.float32)
        self.state = {"train": train.tolist()}


class TinyDataset(Dataset):
    def __init__(self, data, train, count, noise=None):
        self.train = train

    def __len__(self):
        return len(self.train)

    def __getitem__(self, i):
        x = torch.from_numpy(np.random.normal(size=2).astype("float32"))
        return x, x + 1, x, x - 1, torch.tensor(int(self.train[i]) % 2), torch.tensor(i)


@pytest.mark.parametrize(
    "method,rank_mode,rank_weight",
    [
        ("equal_late", "positive_margin", 0.1),
        ("qmf", "positive_margin", 0.1),
        ("tmc", "positive_margin", 0.1),
        ("qmf", "original", 0.1),
        ("qmf", "positive_margin", 0.0),
    ],
)
def test_interrupted_resume_matches_uninterrupted(
    tmp_path, monkeypatch, method, rank_mode, rank_weight
):
    monkeypatch.setattr(training, "TrimodalModel", TinyModel)
    monkeypatch.setattr(training, "FoldData", TinyData)
    monkeypatch.setattr(training, "PatientDataset", TinyDataset)
    config = Config(
        device="cpu", epochs=2, batch_size=2, rank_mode=rank_mode, rank_weight=rank_weight
    )
    args = (config, Path("unused"), np.arange(6), np.array([], dtype=int))
    full, _, _ = training.fit(*args, tmp_path / "full", method, 42, 2, False, "same")
    real_save = training.save_checkpoint

    class Interrupted(Exception):
        pass

    def interrupt_after_first(path, state):
        real_save(path, state)
        if state["epoch"] == 1:
            raise Interrupted()

    monkeypatch.setattr(training, "save_checkpoint", interrupt_after_first)
    with pytest.raises(Interrupted):
        training.fit(*args, tmp_path / "resumed", method, 42, 2, False, "same")
    monkeypatch.setattr(training, "save_checkpoint", real_save)
    resumed, _, _ = training.fit(*args, tmp_path / "resumed", method, 42, 2, True, "same")
    for key, value in full.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)
    state = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    assert state["epoch"] == 2
