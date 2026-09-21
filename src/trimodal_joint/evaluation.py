"""All-image/all-window patient predictions and out-of-fold metrics."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from .data import image_tensor
from .fusion import fuse


def metrics(rows: list[dict]) -> dict:
    y = np.array([r["label"] for r in rows])
    p = np.array([r["probability"] for r in rows])
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probabilities")
    pred = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prediction_row(record: dict, branches: torch.Tensor, out: dict, method: str) -> dict:
    p = float(out["probabilities"][0, 1])
    row = {
        "patient_id": record["patient_id"],
        "fold": record["fold"],
        "label": record["label"],
        "probability": p,
        "prediction": int(p >= 0.5),
    }
    for m, name in enumerate(["us", "emg", "table"]):
        for c in range(2):
            row[f"{name}_score_{c}"] = float(branches[0, m, c])
        if method == "qmf":
            row[f"{name}_quality"] = float(out["confidence"][0, m])
        if method == "tmc":
            alpha = out["branch_alpha"][0, m]
            row[f"{name}_uncertainty"] = float(2 / alpha.sum())
            for c in range(2):
                row[f"{name}_alpha_{c}"] = float(alpha[c])
    if method == "tmc":
        for c in range(2):
            row[f"fusion_alpha_{c}"] = float(out["alpha"][0, c])
        row["fusion_uncertainty"] = float(out["uncertainty"][0])
    return row


@torch.no_grad()
def evaluate(model, data, indices, method: str, device: torch.device) -> list[dict]:
    model.eval()
    rows = []
    for i in indices:
        record = data.records[int(i)]
        us = []
        # At most two ROI images per inference batch; each patient has equal final weight.
        for start in range(0, len(record["image_paths"]), 2):
            image = torch.stack(
                [image_tensor(p, False) for p in record["image_paths"][start : start + 2]]
            ).to(device)
            us.append(model.us_scores(image))
        us = torch.cat(us).mean(0, keepdim=True)
        windows, features = data.patient_emg(int(i))
        scores = []
        for start in range(0, len(windows), 256):
            x = torch.from_numpy(windows[start : start + 256].copy()).to(device)
            f = torch.from_numpy(features[start : start + 256].copy()).to(device)
            scores.append(model.emg(x, f))
        emg = torch.cat(scores).reshape(5, 331, 2).mean(1).mean(0, keepdim=True)
        table = model.tabular(torch.from_numpy(data.table[i : i + 1]).to(device))
        branches = torch.stack([us, emg, table], 1)
        out = fuse(branches, method)
        rows.append(prediction_row(record, branches, out, method))
    return rows
