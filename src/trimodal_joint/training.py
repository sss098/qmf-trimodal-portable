"""Patient-held-out selection/refit with atomic epoch-boundary resume."""

import gc
import json
import logging
import platform
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .augmentation import TrainingNoise
from .config import Config
from .data import FoldData, PatientDataset
from .evaluation import evaluate, metrics
from .fusion import fuse
from .io import (
    fingerprint,
    random_state,
    restore_random,
    save_checkpoint,
    save_json,
    seed_all,
    sha256,
    write_csv,
)
from .losses import QualityHistory, total_loss
from .models import TrimodalModel
from .prepare import code_fingerprint

LOG = logging.getLogger(__name__)


def split_indices(records: list[dict], fold: int) -> tuple:
    folds = np.array([r["fold"] for r in records])
    return (
        np.flatnonzero((folds != fold) & (folds != (fold + 1) % 5)),
        np.flatnonzero(folds == (fold + 1) % 5),
        np.flatnonzero(folds == fold),
    )


def fit(
    config: Config,
    cache: Path,
    train: np.ndarray,
    valid: np.ndarray,
    folder: Path,
    method: str,
    seed: int,
    epochs: int,
    resume: bool,
    run_signature: str,
):
    """Return current model; selection chooses only E, refit starts from scratch."""
    seed_all(seed, config.threads)
    data = FoldData(cache, train)
    device = torch.device(config.device)
    model = TrimodalModel(data.table.shape[1], config.imagenet_weights).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.ultrasound.parameters(), "lr": config.encoder_lr},
            {
                "params": list(model.us_head.parameters())
                + list(model.emg.parameters())
                + list(model.tabular.parameters()),
                "lr": config.lr,
            },
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    labels = np.array([data.records[i]["label"] for i in train])
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training fold must contain both classes")
    weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device=device)
    loader = DataLoader(
        PatientDataset(
            data,
            train,
            config.windows_per_segment,
            TrainingNoise(
                config.noise_probability,
                config.us_noise_max_std,
                config.emg_noise_max_std,
                config.table_noise_max_std,
            ),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    quality = QualityHistory(len(train)) if method == "qmf" else None
    last = folder / "last.pt"
    best = float("inf")
    selected = 0
    stale = 0
    history = []
    start = 1
    phase_signature = fingerprint(
        {"run": run_signature, "train": train.tolist(), "valid": valid.tolist(), "epochs": epochs}
    )
    if last.exists():
        if not resume:
            raise FileExistsError(f"{last}: use --resume or another run name")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["signature"] != phase_signature:
            raise ValueError("Checkpoint configuration/data/code mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    opt_state[key] = value.to(device)
        if quality is not None:
            quality.sums = state["quality_history"]
        history = state["history"]
        best = state["best"]
        selected = state["selected"]
        stale = state["stale"]
        start = state["epoch"] + 1
        restore_random(state["rng"])
    else:
        save_json(folder / "preprocessing.json", data.state)
        save_json(
            folder / "split.json",
            {
                "train": [data.records[i]["patient_id"] for i in train],
                "validation": [data.records[i]["patient_id"] for i in valid],
            },
        )
        save_json(
            folder / "parameters.json",
            {
                "counts": model.parameter_counts(),
                "total": sum(p.numel() for p in model.parameters()),
                "class_weights": weights.cpu().tolist(),
            },
        )
    if not (len(valid) and stale >= config.patience):
        for epoch in range(start, epochs + 1):
            model.train()
            losses = []
            for image, windows, features, table, y, local_index in loader:
                optimizer.zero_grad(set_to_none=True)
                branches = model(
                    image.to(device), windows.to(device), features.to(device), table.to(device)
                )
                out = fuse(branches, method)
                loss = total_loss(
                    branches,
                    out,
                    method,
                    y.to(device),
                    weights,
                    epoch,
                    local_index,
                    quality,
                    config.rank_weight,
                    config.rank_mode,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5, error_if_nonfinite=True
                )
                if epoch == 1 and not losses:
                    missing = [
                        name
                        for name, p in model.named_parameters()
                        if p.requires_grad and p.grad is None
                    ]
                    if missing:
                        raise RuntimeError(f"Untrained parameters: {missing}")
                optimizer.step()
                losses.append(float(loss.detach()))
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "last_gradient_norm": float(norm),
            }
            if len(valid):
                score = metrics(evaluate(model, data, valid, method, device))
                row.update({f"validation_{k}": v for k, v in score.items()})
                if score["log_loss"] < best:
                    best = score["log_loss"]
                    selected = epoch
                    stale = 0
                else:
                    stale += 1
            else:
                selected = epoch
            history.append(row)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "selected": selected,
                "stale": stale,
                "best": best,
                "history": history,
                "rng": random_state(),
                "quality_history": quality.sums if quality else None,
                "signature": phase_signature,
                "preprocessing": data.state,
                "table_dim": data.table.shape[1],
            }
            save_checkpoint(last, state)
            save_json(folder / "history.json", history)
            save_json(
                folder / "selection.json",
                {
                    "selected_epoch": selected,
                    "best_validation_log_loss": best if len(valid) else None,
                },
            )
            LOG.info(
                "%s epoch=%d loss=%.5f selected=%d", folder, epoch, row["train_loss"], selected
            )
            if len(valid) and stale >= config.patience:
                break
    if selected < 1:
        raise RuntimeError("No completed training epochs")
    return model, data, selected


def train_runs(
    config: Config,
    cache: Path,
    methods: list[str],
    seeds: list[int],
    folds: list[int],
    smoke: bool,
    resume: bool,
    name: str,
) -> None:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    meta = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    records = meta["records"]
    root = Path(config.output) / "runs" / name
    shared = {
        "config": config.to_dict(),
        "dataset": meta["fingerprint"],
        "code": code_fingerprint(),
        "weights": sha256(Path(config.imagenet_weights)),
        "smoke": smoke,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if config.device == "cuda" else "cpu",
        },
    }
    all_summaries = {}
    for method in methods:
        method_rows = []
        seed_scores = {}
        for seed in seeds:
            run = root / method / f"seed{seed}"
            contract = {**shared, "method": method, "seed": seed}
            path = run / "config.json"
            if path.exists() and json.loads(path.read_text(encoding="utf-8")) != contract:
                raise ValueError(f"Run configuration changed: {run}; choose a new --name")
            save_json(path, contract)
            signature = fingerprint(contract)
            oof = []
            for fold in folds:
                folder = run / f"fold{fold}"
                done = folder / "predictions.json"
                if done.exists():
                    if not resume:
                        raise FileExistsError(f"{done}: use --resume")
                    oof.extend(json.loads(done.read_text(encoding="utf-8")))
                    continue
                train, valid, test = split_indices(records, fold)
                model, data, selected = fit(
                    config,
                    cache,
                    train,
                    valid,
                    folder / "selection",
                    method,
                    seed + fold,
                    1 if smoke else config.epochs,
                    resume,
                    signature,
                )
                del model, data
                gc.collect()
                if config.device == "cuda":
                    torch.cuda.empty_cache()
                train = np.array(sorted(set(train) | set(valid)), dtype=int)
                model, data, _ = fit(
                    config,
                    cache,
                    train,
                    np.array([], dtype=int),
                    folder / "refit",
                    method,
                    seed + fold,
                    selected,
                    resume,
                    signature,
                )
                rows = evaluate(model, data, test, method, torch.device(config.device))
                save_checkpoint(
                    folder / "model.pt",
                    {
                        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "table_dim": data.table.shape[1],
                        "preprocessing": data.state,
                        "contract": contract,
                        "selected_epochs": selected,
                    },
                )
                save_json(
                    folder / "test_split.json", {"test": [records[i]["patient_id"] for i in test]}
                )
                save_json(folder / "metrics.json", metrics(rows))
                save_json(done, rows)
                oof.extend(rows)
                del model, data
                gc.collect()
                if config.device == "cuda":
                    torch.cuda.empty_cache()
            oof.sort(key=lambda r: r["patient_id"])
            if len({r["patient_id"] for r in oof}) != len(oof):
                raise ValueError("Duplicate OOF patients")
            write_csv(run / "oof_predictions.csv", oof)
            score = metrics(oof)
            save_json(run / "summary.json", {"metrics": score, "n": len(oof), "smoke": smoke})
            method_rows.append(oof)
            seed_scores[str(seed)] = score
        ids = [[r["patient_id"] for r in rows] for rows in method_rows]
        if any(x != ids[0] for x in ids):
            raise ValueError("Seed OOF IDs differ")
        probabilities = np.mean([[r["probability"] for r in rows] for rows in method_rows], axis=0)
        ensemble = [
            {k: r[k] for k in ("patient_id", "fold", "label")}
            | {"probability": float(p), "prediction": int(p >= 0.5)}
            for r, p in zip(method_rows[0], probabilities)
        ]
        write_csv(root / method / "ensemble_predictions.csv", ensemble)
        report = {
            "per_seed": seed_scores,
            "ensemble": metrics(ensemble),
            "n": len(ensemble),
            "smoke": smoke,
            "complete_five_fold": sorted(folds) == list(range(5)),
            "mean": {
                k: float(np.mean([v[k] for v in seed_scores.values()]))
                for k in seed_scores[str(seeds[0])]
            },
            "sample_sd": {
                k: float(np.std([v[k] for v in seed_scores.values()], ddof=1))
                if len(seeds) > 1
                else None
                for k in seed_scores[str(seeds[0])]
            },
        }
        save_json(root / method / "summary.json", report)
        all_summaries[method] = report
    save_json(root / "comparison.json", all_summaries)
