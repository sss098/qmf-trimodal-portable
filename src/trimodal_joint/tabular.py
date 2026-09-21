"""Fold-local v1.4 tabular preprocessing."""

import numpy as np


def fit_table(x, indices, names, max_missing=0.5):
    train = x[indices]
    keep = np.flatnonzero(
        (np.isnan(train).mean(0) <= max_missing)
        & np.array([len(np.unique(c[np.isfinite(c)])) > 1 for c in train.T])
    )
    if not len(keep):
        raise ValueError("No usable training columns")
    subset = train[:, keep]
    median = np.nanmedian(subset, axis=0)
    filled = np.where(np.isnan(subset), median, subset)
    mean = filled.mean(0)
    std = filled.std(0)
    std = np.where(std < 1e-6, 1, std)
    return {
        "keep": keep,
        "median": median,
        "mean": mean,
        "std": std,
        "names": [names[i] for i in keep],
        "dropped": [names[i] for i in range(len(names)) if i not in keep],
        "fit_indices": np.asarray(indices),
    }


def transform_table(x, state):
    a = x[:, state["keep"]]
    out = ((np.where(np.isnan(a), state["median"], a) - state["mean"]) / state["std"]).astype(
        np.float32
    )
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite tabular inputs")
    return out
