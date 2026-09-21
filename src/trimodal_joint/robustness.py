"""Frozen-checkpoint inference under paired, single-modality Gaussian corruption."""

import hashlib

import numpy as np
import torch

from .data import image_tensor
from .features import batch_features
from .raw import segment_windows

MODALITIES = ("us", "emg", "table")


def noise_for(shape, patient_id, modality, repeat):
    key = f"qmf-corruption-v1:{patient_id}:{modality}:{repeat}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def perturb(values, noise, level):
    if not np.isfinite(level) or level < 0:
        raise ValueError("Noise level must be finite and nonnegative")
    return (values + level * noise).astype(np.float32)


def emg_noise(patient_id, repeat):
    # Generate once on the continuous segments so overlapping windows agree.
    return segment_windows(noise_for((5, 10000, 6), patient_id, "emg", repeat)).reshape(-1, 100, 6)


@torch.no_grad()
def branch_scores(model, record, raw_windows, table, state, modality, level, repeat, device):
    """Use saved refit preprocessing; never fit transforms on test patients."""
    pid = record["patient_id"]
    if modality == "us":
        scores = []
        mean = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
        for start in range(0, len(record["image_paths"]), 2):
            images = []
            for index in range(start, min(start + 2, len(record["image_paths"]))):
                image = image_tensor(record["image_paths"][index], False).numpy()
                if level:
                    pixel = image * std + mean
                    noise = noise_for((1, 224, 224), pid, f"us:{index}", repeat)
                    pixel = np.clip(perturb(pixel, noise, 0.1 * level), 0, 1)
                    image = (pixel - mean) / std
                images.append(torch.from_numpy(image))
            scores.append(model.us_scores(torch.stack(images).to(device)))
        return torch.cat(scores).mean(0)
    if modality == "emg":
        windows = ((raw_windows - state["emg_mean"]) / state["emg_std"]).reshape(-1, 100, 6)
        if level:
            windows = perturb(windows, emg_noise(pid, repeat), level)
        features = batch_features(windows)
        scores = []
        for start in range(0, len(windows), 256):
            x = torch.from_numpy(windows[start : start + 256].copy()).to(device)
            f = torch.from_numpy(features[start : start + 256].copy()).to(device)
            scores.append(model.emg(x, f))
        return torch.cat(scores).reshape(5, 331, 2).mean(1).mean(0)
    if modality == "table":
        if level:
            # Draw in the original column space to preserve pairing if folds drop columns.
            noise = noise_for((state["table_raw_dim"],), pid, "table", repeat)
            table = perturb(table, noise[state["table"]["keep"]], level)
        return model.tabular(torch.from_numpy(table[None].copy()).to(device))[0]
    raise ValueError(f"Unknown modality: {modality}")
