"""Train-only corruption must retain feature consistency and clean defaults."""

import numpy as np
import torch

from trimodal_joint.augmentation import TrainingNoise
from trimodal_joint.features import batch_features


def sample():
    positions = np.tile(np.array([0, 1]), (5, 1))
    windows = np.ones((10, 100, 6), dtype=np.float32)
    return (
        torch.zeros(3, 224, 224),
        windows,
        batch_features(windows),
        np.zeros(3, np.float32),
        positions,
    )


def test_disabled_augmentation_preserves_inputs_and_rng():
    values = sample()
    rng = np.random.RandomState(42)
    before = rng.get_state()[1].copy()
    result = TrainingNoise()(*values, rng=rng)
    for x, y in zip(result, values[:4]):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(before, rng.get_state()[1])


def test_emg_corruption_recomputes_features_and_preserves_overlap():
    image, windows, features, table, positions = sample()
    noise = TrainingNoise(probability=1)
    # Force the EMG branch while retaining an actual seeded Gaussian draw.
    result = noise.corrupt(
        image,
        windows,
        features,
        table,
        positions,
        modality=1,
        std=0.5,
        rng=np.random.RandomState(42),
    )
    np.testing.assert_array_equal(result[0], image)
    np.testing.assert_array_equal(result[3], table)
    np.testing.assert_allclose(result[2], batch_features(result[1]))
    w = result[1].reshape(5, 2, 100, 6)
    np.testing.assert_allclose(w[:, 0, 30:], w[:, 1, :-30])
    np.testing.assert_array_equal(windows, np.ones_like(windows))


def test_image_corruption_stays_in_pixel_bounds_and_shares_channels():
    values = sample()
    result = TrainingNoise().corrupt(*values, modality=0, std=0.1, rng=np.random.RandomState(2))
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    pixels = result[0] * std + mean
    assert pixels.min() >= -1e-7 and pixels.max() <= 1 + 1e-7
    for i in [1, 2, 3]:
        np.testing.assert_array_equal(result[i], values[i])


def test_augmented_rng_restore_reproduces_next_sample():
    from trimodal_joint.io import random_state, restore_random

    values = sample()
    augment = TrainingNoise(probability=1)
    state = random_state()
    first = augment(*values)
    restore_random(state)
    second = augment(*values)
    for x, y in zip(first, second):
        np.testing.assert_array_equal(x, y)


def test_table_corruption_does_not_change_other_modalities():
    values = sample()
    result = TrainingNoise().corrupt(*values, modality=2, std=1, rng=np.random.RandomState(2))
    assert not np.array_equal(result[3], values[3])
    for i in [0, 1, 2]:
        np.testing.assert_array_equal(result[i], values[i])
