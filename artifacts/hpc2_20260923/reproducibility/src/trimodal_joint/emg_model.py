"""Adapted from the user-provided per-channel FT-Transformer; see provenance."""

import torch
from torch import nn

N_CLASSES = 2


class ChannelTokenizer(nn.Module):
    """Map each channel waveform (T) into a token embedding."""

    def __init__(self, window_size: int, d_token: int):
        super().__init__()
        self.proj = nn.Linear(window_size, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return self.proj(x)


class FTTransformerRaw(nn.Module):
    """Transformer that fuses per-channel handcrafted features into channel tokens.

    Fusion strategy:
      - handcrafted features have shape (B, N_channels * per_chan_feat)
      - reshape -> (B, N_channels, per_chan_feat)
      - map each per-channel feature vector to a token embedding of size d_token
      - add this embedding to the corresponding channel token (element-wise)
      - proceed with cls_token, pos_embedding and Transformer encoder
    """

    def __init__(
        self,
        n_channels: int,
        window_size: int,
        n_classes: int = N_CLASSES,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.2,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.tokenizer = ChannelTokenizer(window_size, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_channels + 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # per-channel feature mapper (if feature_dim > 0 and divisible by n_channels)
        self.per_channel_mapper = None
        self.per_chan_feat = 0
        fused_dim = d_token
        if feature_dim > 0:
            if feature_dim % n_channels != 0:
                raise ValueError(
                    "feature_dim must be divisible by n_channels for per-channel fusion"
                )
            self.per_chan_feat = feature_dim // n_channels
            # map per-channel feature vector -> d_token embedding
            self.per_channel_mapper = nn.Sequential(
                nn.LayerNorm(self.per_chan_feat),
                nn.Linear(self.per_chan_feat, d_token),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_token, d_token),
            )
        # classification head (operates on cls token only)
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, n_classes),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        tokens = self.tokenizer(x)  # (B, C, d_token)

        if self.per_channel_mapper is not None:
            B = features.size(0)
            # features shape: (B, C * per_chan_feat) -> reshape
            feat = features.view(B, self.n_channels, self.per_chan_feat)
            feat = feat.to(tokens.dtype)
            feat_tok = self.per_channel_mapper(feat)  # (B, C, d_token)
            tokens = tokens + feat_tok

        # prepend cls token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, C+1, d_token)
        tokens = tokens + self.pos_embedding
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]

        return self.head(cls_out)
