"""Strict, serializable experiment configuration."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    source_project: str = "data/source"
    clinical_excel: str = "data/clinical.xlsx"
    raw_root: str = "data/raw"
    imagenet_weights: str = "weights/resnet50-11ad3fa6.pth"
    output: str = "artifacts"
    epochs: int = 100
    patience: int = 12
    batch_size: int = 6
    windows_per_segment: int = 8
    lr: float = 2e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 1e-3
    rank_weight: float = 0.1
    rank_mode: str = "positive_margin"
    split_seed: int = 20260911
    device: str = "cuda"
    threads: int = 2
    noise_probability: float = 0.0
    us_noise_max_std: float = 0.1
    emg_noise_max_std: float = 1.0
    table_noise_max_std: float = 1.0

    def validate(self) -> None:
        if not math.isfinite(self.noise_probability) or not 0 <= self.noise_probability <= 1:
            raise ValueError("noise_probability must lie in [0,1]")
        for key in ["us_noise_max_std", "emg_noise_max_std", "table_noise_max_std"]:
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not math.isfinite(self.rank_weight):
            raise ValueError("rank_weight must be finite")
        if self.rank_mode not in ("positive_margin", "original"):
            raise ValueError("rank_mode must be positive_margin or original")
        for key in ["epochs", "patience", "batch_size", "windows_per_segment", "threads"]:
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.epochs > 100:
            raise ValueError("Maximum training budget is 100 epochs")
        if self.windows_per_segment > 331:
            raise ValueError("windows_per_segment exceeds available windows")
        for key in ["lr", "encoder_lr"]:
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f"Invalid {key}")
        if self.weight_decay < 0 or self.rank_weight < 0 or self.device not in ("cpu", "cuda"):
            raise ValueError("Invalid regularization or device")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | None) -> Config:
    values = {} if path is None else yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Config must be a YAML mapping")
    unknown = set(values) - set(Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config = Config(**values)
    config.validate()
    return config
