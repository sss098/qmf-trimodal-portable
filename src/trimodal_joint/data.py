"""Patient-balanced multimodal sampling and strictly training-fold preprocessing."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .augmentation import TrainingNoise
from .features import batch_features
from .io import fingerprint, save_json, sha256
from .tabular import fit_table, transform_table


def image_tensor(path: str, train: bool) -> torch.Tensor:
    with Image.open(path) as original:
        image = original.convert("RGB")
    image.thumbnail((224, 224), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (224, 224))
    canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
    if train:
        if random.random() < 0.5:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.92, 1.08))
        canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.90, 1.10))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy((array - mean) / std)


class FoldData:
    """Read-only raw windows plus training-fitted channel Z-score and table state."""

    def __init__(self, cache: Path, train: np.ndarray):
        self.meta = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        self.records = self.meta["records"]
        self.windows = np.load(cache / "windows.npy", mmap_mode="r")
        with np.load(cache / "table.npz") as file:
            raw = file["values"]
            names = file["names"].tolist()
        self.table_state = fit_table(raw, train, names)
        self.table = transform_table(raw, self.table_state)
        sum_c = np.zeros(6)
        square = np.zeros(6)
        total = 0
        for i in train:
            x = np.asarray(self.windows[i], dtype=np.float64)
            sum_c += x.sum(axis=(0, 1, 2))
            square += (x * x).sum(axis=(0, 1, 2))
            total += np.prod(x.shape[:-1])
        mean = sum_c / total
        var = square / total - mean**2
        self.mean = mean.astype(np.float32)
        self.std = (np.sqrt(np.maximum(var, 1e-12)) + 1e-8).astype(np.float32)
        self.state = {
            "train_indices": train.tolist(),
            "emg_mean": self.mean,
            "emg_std": self.std,
            "table": self.table_state,
        }
        # Features depend on train-only Z-score; never use another fold's cached features.
        signature = fingerprint(
            {
                "cache": self.meta["fingerprint"],
                "state": self.state,
                "features_code": sha256(Path(__file__).with_name("features.py")),
            }
        )
        feature_dir = cache / "features" / signature
        feature_dir.mkdir(parents=True, exist_ok=True)
        path = feature_dir / "features.npy"
        marker = feature_dir / "complete.json"
        if not marker.exists():
            temp = feature_dir / "features.tmp.npy"
            features = np.lib.format.open_memmap(
                temp, mode="w+", dtype=np.float32, shape=(len(self.records), 5, 331, 144)
            )
            for i in range(len(self.records)):
                windows = (self.windows[i] - self.mean) / self.std
                features[i] = batch_features(windows.reshape(-1, 100, 6)).reshape(5, 331, 144)
            features.flush()
            del features
            temp.replace(path)
            save_json(marker, {"sha256": sha256(path)})
        if sha256(path) != json.loads(marker.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError("Feature cache checksum mismatch")
        self.features = np.load(path, mmap_mode="r")

    def patient_emg(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return ((self.windows[index] - self.mean) / self.std).reshape(-1, 100, 6), self.features[
            index
        ].reshape(-1, 144)


class PatientDataset(Dataset):
    def __init__(
        self,
        data: FoldData,
        indices: np.ndarray,
        windows_per_segment: int,
        noise: TrainingNoise | None = None,
    ):
        self.data = data
        self.indices = indices
        self.count = windows_per_segment
        self.noise = noise or TrainingNoise()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple:
        index = int(self.indices[item])
        record = self.data.records[index]
        positions = np.stack([np.random.choice(331, self.count, replace=False) for _ in range(5)])
        segments = np.arange(5)[:, None]
        raw = self.data.windows[index][segments, positions]
        raw = ((raw - self.data.mean) / self.data.std).reshape(-1, 100, 6).copy()
        features = self.data.features[index][segments, positions].reshape(-1, 144).copy()
        image = image_tensor(random.choice(record["image_paths"]), True)
        image, raw, features, table = self.noise(
            image, raw, features, self.data.table[index], positions
        )
        return (
            image,
            torch.from_numpy(raw),
            torch.from_numpy(features),
            torch.from_numpy(table),
            torch.tensor(record["label"], dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
        )
