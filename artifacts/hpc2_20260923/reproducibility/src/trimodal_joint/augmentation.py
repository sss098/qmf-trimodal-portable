"""Single-modality input corruption for training, using the checkpointed NumPy RNG."""

from dataclasses import dataclass

import numpy as np
import torch

from .features import batch_features
from .raw import segment_windows


@dataclass(frozen=True)
class TrainingNoise:
    probability: float = 0.0
    us_max_std: float = 0.1
    emg_max_std: float = 1.0
    table_max_std: float = 1.0

    def __call__(self, image, windows, features, table, positions, rng=np.random):
        if self.probability == 0 or rng.random() >= self.probability:
            return image, windows, features, table
        modality = int(rng.choice(3))
        maximum = (self.us_max_std, self.emg_max_std, self.table_max_std)[modality]
        std = float(rng.uniform(0, maximum))
        return self.corrupt(image, windows, features, table, positions, modality, std, rng)

    @staticmethod
    def corrupt(image, windows, features, table, positions, modality, std, rng):
        if modality == 0:
            mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
            scale = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
            noise = torch.from_numpy(rng.normal(0, std, (1, 224, 224)).astype(np.float32))
            pixels = (image * scale + mean + noise).clamp(0, 1)
            image = (pixels - mean) / scale
        elif modality == 1:
            continuous = rng.normal(0, std, (5, 10000, 6)).astype(np.float32)
            noise = segment_windows(continuous)[np.arange(5)[:, None], positions]
            windows = windows + noise.reshape(-1, 100, 6)
            features = batch_features(windows)
        elif modality == 2:
            table = table + rng.normal(0, std, table.shape).astype(np.float32)
        else:
            raise ValueError(f"Unknown modality index: {modality}")
        return image, windows, features, table
