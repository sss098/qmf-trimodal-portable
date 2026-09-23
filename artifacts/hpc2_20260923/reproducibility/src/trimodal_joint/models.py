"""Three jointly trainable branches with common two-class raw-score interface."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50

from .emg_model import FTTransformerRaw


class TrimodalModel(nn.Module):
    def __init__(self, table_dim: int, imagenet_weights: str, modalities: str = "us,emg,table"):
        super().__init__()
        self.modalities = tuple(modalities.split(","))
        self.ultrasound = resnet50(weights=None)
        path = Path(imagenet_weights)
        if not path.is_file():
            raise FileNotFoundError(f"ImageNet weights missing: {path}")
        self.ultrasound.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True), strict=True
        )
        self.ultrasound.fc = nn.Identity()
        self.us_head = nn.Sequential(nn.Dropout(0.35), nn.Linear(2048, 2))
        self.emg = FTTransformerRaw(n_channels=6, window_size=100, n_classes=2, feature_dim=144)
        self.tabular = nn.Sequential(
            nn.Linear(table_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

        # Initialize in the original order so retained branches keep the same seed state.
        # Remove inactive modules entirely: they cannot train or enter checkpoints.
        if "us" not in self.modalities:
            del self.ultrasound, self.us_head
        if "emg" not in self.modalities:
            del self.emg
        if "table" not in self.modalities:
            del self.tabular

    def us_scores(self, image: torch.Tensor) -> torch.Tensor:
        return self.us_head(self.ultrasound(image))

    def forward(
        self,
        image: torch.Tensor,
        windows: torch.Tensor,
        features: torch.Tensor,
        table: torch.Tensor,
    ) -> torch.Tensor:
        scores = []
        if "us" in self.modalities:
            scores.append(self.us_scores(image))
        if "emg" in self.modalities:
            batch, count = windows.shape[:2]
            scores.append(
                self.emg(windows.reshape(-1, 100, 6), features.reshape(-1, 144))
                .reshape(batch, count, 2)
                .mean(1)
            )
        if "table" in self.modalities:
            scores.append(self.tabular(table))
        return torch.stack(scores, 1)

    def parameter_counts(self) -> dict[str, int]:
        return {
            name: sum(p.numel() for p in module.parameters())
            for name, module in self.named_children()
        }
