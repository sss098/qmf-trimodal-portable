"""Vectorized numerical equivalent of the original 24 features per channel."""

import numpy as np
import pywt

from .emg_features import BANDS


def batch_features(windows: np.ndarray) -> np.ndarray:
    """Return [windows,144] with channel-major ordering; input [windows,100,6]."""
    x = np.asarray(windows, dtype=np.float32).transpose(0, 2, 1)
    if x.shape[1:] != (6, 100):
        raise ValueError(f"Expected [N,100,6], got {windows.shape}")
    diff = np.diff(x, axis=-1)
    time = [
        np.sqrt(np.mean(x * x, -1)),
        np.mean(np.abs(x), -1),
        np.max(np.abs(x), -1),
        (((x[..., :-1] >= 0) != (x[..., 1:] >= 0)) & (np.abs(diff) > 1e-6)).sum(-1),
        np.abs(diff).sum(-1),
        (
            ((diff[..., :-1] >= 0) != (diff[..., 1:] >= 0))
            & (np.abs(np.diff(diff, axis=-1)) > 1e-6)
        ).sum(-1),
    ]
    freqs = np.fft.rfftfreq(100, 1 / 1000)
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    total = power.sum(-1) + 1e-12
    band = [power[..., (freqs >= a) & (freqs < b)].sum(-1) for a, b in BANDS]
    centroid = (power * freqs).sum(-1) / total
    # searchsorted(side='left'), including the guarded all-zero case.
    median_index = (
        (np.cumsum(power, axis=-1) < (0.5 * total)[..., None]).sum(-1).clip(max=len(freqs) - 1)
    )
    fft = band + [centroid, freqs[power.argmax(-1)], freqs[median_index], centroid]
    frames = np.stack([x[..., start : start + 64] * np.hanning(64) for start in (0, 32)], axis=-2)
    stft_power = np.abs(np.fft.rfft(frames, n=128, axis=-1)) ** 2
    sf = np.fft.rfftfreq(128, 1 / 1000)
    stft = [stft_power[..., (sf >= a) & (sf < b)].mean(axis=(-1, -2)) for a, b in BANDS]
    wave = [np.sum(c * c, -1) for c in pywt.wavedec(x, "db4", level=3, axis=-1)]
    out = np.stack(time + fft + stft + wave, -1).reshape(len(x), 144).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite handcrafted features")
    return out
