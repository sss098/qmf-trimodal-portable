"""Scalar reference for parity tests; production uses features.batch_features.

The shared frequency bands and 24-feature ordering belong to formal_v1's
input contract, including the repeated spectral centroid/mean-power frequency.
"""

import numpy as np
import pywt

BANDS = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 500)]
STFT_WIN, STFT_HOP, STFT_NFFT = 64, 32, 128
WAVELET_NAME, WAVELET_LEVEL = "db4", 3


def _zero_crossings(x: np.ndarray, eps: float = 1e-6) -> int:
    x1 = x[:-1]
    x2 = x[1:]
    signs = (x1 >= 0) != (x2 >= 0)
    return int(np.sum(signs & (np.abs(x1 - x2) > eps)))


def _slope_sign_changes(x: np.ndarray, eps: float = 1e-6) -> int:
    dx = np.diff(x)
    dx1 = dx[:-1]
    dx2 = dx[1:]
    signs = (dx1 >= 0) != (dx2 >= 0)
    return int(np.sum(signs & (np.abs(dx1 - dx2) > eps)))


def _fft_features(x: np.ndarray, fs: float) -> np.ndarray:
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(x)
    power = (np.abs(fft_vals) ** 2).astype(np.float64)

    band_energies = []
    for f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energies.append(power[mask].sum())

    power_sum = power.sum() + 1e-12
    centroid = float((freqs * power).sum() / power_sum)
    peak_freq = float(freqs[np.argmax(power)])

    cumsum = np.cumsum(power)
    median_freq = float(freqs[min(int(np.searchsorted(cumsum, 0.5 * power_sum)), len(freqs) - 1)])
    mean_power_freq = centroid

    return np.array(
        band_energies + [centroid, peak_freq, median_freq, mean_power_freq], dtype=np.float32
    )


def _stft_features(x: np.ndarray, fs: float) -> np.ndarray:
    n = x.shape[0]
    if n < STFT_WIN:
        return np.zeros((len(BANDS),), dtype=np.float32)

    frames = []
    for start in range(0, n - STFT_WIN + 1, STFT_HOP):
        frame = x[start : start + STFT_WIN]
        win = np.hanning(STFT_WIN)
        frames.append(frame * win)

    mags = []
    for frame in frames:
        spec = np.fft.rfft(frame, n=STFT_NFFT)
        mags.append(np.abs(spec) ** 2)

    mags = np.stack(mags, axis=0)
    freqs = np.fft.rfftfreq(STFT_NFFT, d=1.0 / fs)

    band_energy = []
    for f_lo, f_hi in BANDS:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_energy.append(mags[:, mask].mean())

    return np.array(band_energy, dtype=np.float32)


def _wavelet_energy(x: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec(x, WAVELET_NAME, level=WAVELET_LEVEL)
    energies = [np.sum(c**2) for c in coeffs]
    return np.array(energies, dtype=np.float32)


def extract_features(window: np.ndarray, fs: float) -> np.ndarray:
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch].astype(np.float32)

        rms = float(np.sqrt(np.mean(x**2)))
        mav = float(np.mean(np.abs(x)))
        peak = float(np.max(np.abs(x)))
        zc = float(_zero_crossings(x))
        wl = float(np.sum(np.abs(np.diff(x))))
        ssc = float(_slope_sign_changes(x))

        fft_feats = _fft_features(x, fs)
        stft_feats = _stft_features(x, fs)
        wav_feats = _wavelet_energy(x)

        feats.append(
            np.concatenate(
                [
                    np.array([rms, mav, peak, zc, wl, ssc], dtype=np.float32),
                    fft_feats,
                    stft_feats,
                    wav_feats,
                ]
            )
        )

    return np.concatenate(feats, axis=0).astype(np.float32)
