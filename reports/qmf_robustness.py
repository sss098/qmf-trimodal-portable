"""Paired corruption evaluation of four frozen fusion settings; no training."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qmf_diagnostics import bootstrap_indices, correlations, interval, metric_arrays, render_table
from scipy.special import logsumexp

from trimodal_joint.fusion import fuse
from trimodal_joint.io import fingerprint, save_json, seed_all, sha256
from trimodal_joint.models import TrimodalModel
from trimodal_joint.robustness import MODALITIES, branch_scores
from trimodal_joint.tabular import transform_table

RUNS = {
    "equal_late": ("formal_v1", "equal_late"),
    "positive_margin": ("formal_v1", "qmf"),
    "rank0": ("qmf_rank0_v1", "qmf"),
    "original": ("qmf_rank_original_v1", "qmf"),
}


def diagnostics(frame, setting, seed):
    rows = []
    for fold, group in frame.groupby("fold"):
        y = group.label.to_numpy(int)
        for modality in MODALITIES:
            z = group[[f"{modality}_score_0", f"{modality}_score_1"]].to_numpy()
            q = logsumexp(z, axis=1) / 10
            loss = logsumexp(z, axis=1) - z[np.arange(len(y)), y]
            pearson, spearman = correlations(q, loss)
            rows.append(
                dict(
                    setting=setting,
                    seed=seed,
                    fold=fold,
                    modality=modality,
                    n=len(y),
                    pearson=pearson,
                    spearman=spearman,
                    covariance=float(np.cov(q, loss, ddof=0)[0, 1]),
                    q_mean=float(q.mean()),
                    q_std=float(q.std()),
                    q_mean_minus_one_third=float(q.mean() - 1 / 3),
                )
            )
    return rows


def run_inference(args):
    all_rows, diagnostic_rows, sources = [], [], {}
    anchor = None
    for setting in args.settings:
        run, method = RUNS[setting]
        if args.run_name:
            run = args.run_name
        for seed in args.seeds:
            root = args.project / "artifacts/runs" / run / method / f"seed{seed}"
            contract = json.loads((root / "config.json").read_text())
            cache = args.project / "artifacts/prepared" / contract["dataset"]
            meta = json.loads((cache / "manifest.json").read_text())
            if meta["fingerprint"] != contract["dataset"]:
                raise ValueError("Dataset fingerprint mismatch")
            raw = np.load(cache / "windows.npy", mmap_mode="r")
            with np.load(cache / "table.npz") as file:
                raw_table = file["values"]
            records = {r["patient_id"]: (i, r) for i, r in enumerate(meta["records"])}
            frame = pd.read_csv(root / "oof_predictions.csv").sort_values("patient_id")
            frame = frame[frame.fold.isin(args.folds)].reset_index(drop=True)
            if frame.patient_id.duplicated().any():
                raise ValueError("Duplicate OOF patient")
            key = frame[["patient_id", "fold", "label"]]
            if anchor is None:
                anchor = key
            pd.testing.assert_frame_equal(key, anchor)
            diagnostic_rows.extend(diagnostics(frame, setting, seed))
            sources[str(root / "oof_predictions.csv")] = sha256(root / "oof_predictions.csv")
            if args.diagnostics_only:
                continue
            for fold in args.folds:
                folder = root / f"fold{fold}"
                checkpoint = folder / "model.pt"
                digest = sha256(checkpoint)
                sources[str(checkpoint)] = digest
                token = fingerprint(
                    dict(
                        checkpoint=digest,
                        levels=args.levels,
                        repeats=args.noise_repeats,
                        smoke=args.smoke,
                        code=sha256(Path(__file__)),
                        inference=sha256(Path(branch_scores.__code__.co_filename)),
                    )
                )
                destination = args.output / "predictions" / setting / f"seed{seed}_fold{fold}.json"
                if destination.exists():
                    previous = json.loads(destination.read_text())
                    if previous["signature"] != token:
                        raise ValueError(f"Resume configuration changed: {destination}")
                    all_rows.extend(previous["rows"])
                    print(f"Reused {setting} seed={seed} fold={fold}", flush=True)
                    continue
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if state["contract"] != contract:
                    raise ValueError("Checkpoint contract mismatch")
                preprocess = state["preprocessing"]
                preprocess["table_raw_dim"] = raw_table.shape[1]
                table = transform_table(raw_table, preprocess["table"])
                model = TrimodalModel(state["table_dim"], contract["config"]["imagenet_weights"])
                model.load_state_dict(state["model"], strict=True)
                del state
                model.to(args.device).eval()
                test_ids = json.loads((folder / "test_split.json").read_text())["test"]
                selected = frame[frame.fold == fold]
                if set(test_ids) != set(selected.patient_id):
                    raise ValueError("Saved test split differs from OOF predictions")
                if args.smoke:
                    selected = selected.groupby("label", group_keys=False).head(1)
                rows = []
                for _, saved in selected.iterrows():
                    i, record = records[saved.patient_id]
                    if i in preprocess["train_indices"] or record["label"] != saved.label:
                        raise ValueError("Patient leakage or label mismatch")
                    clean = torch.stack(
                        [
                            branch_scores(
                                model, record, raw[i], table[i], preprocess, m, 0, 0, args.device
                            )
                            for m in MODALITIES
                        ]
                    )[None]
                    expected = np.array(
                        [[saved[f"{m}_score_{c}"] for c in range(2)] for m in MODALITIES]
                    )
                    np.testing.assert_allclose(clean[0].cpu(), expected, rtol=2e-4, atol=2e-4)
                    clean_probability = float(fuse(clean, method)["probabilities"][0, 1])
                    np.testing.assert_allclose(
                        clean_probability, saved.probability, rtol=2e-4, atol=2e-5
                    )
                    for m, modality in enumerate(MODALITIES):
                        for repeat in range(args.noise_repeats):
                            for level in args.levels:
                                z = clean.clone()
                                if level:
                                    z[0, m] = branch_scores(
                                        model,
                                        record,
                                        raw[i],
                                        table[i],
                                        preprocess,
                                        modality,
                                        level,
                                        repeat,
                                        args.device,
                                    )
                                q = torch.logsumexp(z[0, m], 0) / 10
                                loss = torch.logsumexp(z[0, m], 0) - z[0, m, int(saved.label)]
                                rows.append(
                                    dict(
                                        setting=setting,
                                        seed=seed,
                                        fold=fold,
                                        patient_id=saved.patient_id,
                                        label=int(saved.label),
                                        modality=modality,
                                        repeat=repeat,
                                        level=level,
                                        probability=float(fuse(z, method)["probabilities"][0, 1]),
                                        quality=float(q),
                                        branch_loss=float(loss),
                                    )
                                )
                save_json(destination, dict(signature=token, rows=rows))
                all_rows.extend(rows)
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"Finished {setting} seed={seed} fold={fold}, patients={len(selected)}",
                    flush=True,
                )
    pd.DataFrame(diagnostic_rows).to_csv(args.output / "fold_quality.csv", index=False)
    return pd.DataFrame(all_rows), sources


def summarize(frame, args):
    # Ensemble model seeds, but measure each noise realization separately before averaging metrics.
    group = ["setting", "modality", "level", "repeat", "patient_id", "label"]
    ensemble = frame.groupby(group, as_index=False)[
        ["probability", "quality", "branch_loss"]
    ].mean()
    ensemble.to_csv(args.output / "ensemble_predictions.csv", index=False)
    metrics_rows, delta_rows, quality_rows = [], [], []
    points, boots = {}, {}
    for (setting, modality, level), part in ensemble.groupby(["setting", "modality", "level"]):
        pivot = part.pivot(index="patient_id", columns="repeat", values="probability").sort_index()
        labels = part.groupby("patient_id").label.first().reindex(pivot.index).to_numpy(int)
        if pivot.isna().any().any() or len(pivot.columns) != args.noise_repeats:
            raise ValueError("Missing paired noise predictions")
        indices = bootstrap_indices(labels, args.bootstrap, 20260920)
        point, boot = [], []
        for repeat in pivot.columns:
            p = pivot[repeat].to_numpy()
            point.append(metric_arrays(labels, p[None]))
            boot.append(metric_arrays(labels[indices], p[indices]))
        key = setting, modality, level
        points[key] = {k: float(np.mean([x[k][0] for x in point])) for k in point[0]}
        boots[key] = {k: np.mean([x[k] for x in boot], axis=0) for k in boot[0]}
        for metric, estimate in points[key].items():
            low, high = interval(boots[key][metric])
            metrics_rows.append(
                dict(
                    setting=setting,
                    modality=modality,
                    level=level,
                    metric=metric,
                    estimate=estimate,
                    ci95_low=low,
                    ci95_high=high,
                )
            )
        # Patient-paired changes; average repeats and seeds without treating them as new patients.
        baseline = ensemble[
            (ensemble.setting == setting) & (ensemble.modality == modality) & (ensemble.level == 0)
        ]
        for variable in ["quality", "branch_loss"]:
            current = part.groupby("patient_id")[variable].mean().reindex(pivot.index).to_numpy()
            clean = baseline.groupby("patient_id")[variable].mean().reindex(pivot.index).to_numpy()
            change = current - clean
            low, high = interval(change[indices].mean(1))
            quality_rows.append(
                dict(
                    setting=setting,
                    modality=modality,
                    level=level,
                    variable=variable,
                    mean=float(current.mean()),
                    change=float(change.mean()),
                    ci95_low=low,
                    ci95_high=high,
                )
            )
    for key in points:
        setting, modality, level = key
        clean_key = setting, modality, 0.0
        for metric in points[key]:
            # Signed change: negative means degradation for AUC/accuracy, positive for loss.
            delta = points[key][metric] - points[clean_key][metric]
            distribution = boots[key][metric] - boots[clean_key][metric]
            low, high = interval(distribution)
            delta_rows.append(
                dict(
                    setting=setting,
                    modality=modality,
                    level=level,
                    comparison="corrupted_minus_clean",
                    metric=metric,
                    estimate=delta,
                    ci95_low=low,
                    ci95_high=high,
                )
            )
            if setting != "equal_late" and "equal_late" in args.settings:
                eq, eq_clean = ("equal_late", modality, level), ("equal_late", modality, 0.0)
                difference = delta - (points[eq][metric] - points[eq_clean][metric])
                low, high = interval(distribution - (boots[eq][metric] - boots[eq_clean][metric]))
                delta_rows.append(
                    dict(
                        setting=setting,
                        modality=modality,
                        level=level,
                        comparison="change_minus_equal_late_change",
                        metric=metric,
                        estimate=difference,
                        ci95_low=low,
                        ci95_high=high,
                    )
                )
    performance = pd.DataFrame(metrics_rows)
    performance.to_csv(args.output / "metrics_ci.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.output / "paired_changes_ci.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(args.output / "quality_changes_ci.csv", index=False)
    shown = performance[performance.metric.isin(["roc_auc", "balanced_accuracy"])]
    text = "# QMF 模态质量下降实验\n\n"
    text += "固定已训练模型；每次只扰动一种输入，保留完整患者级聚合。\n\n"
    text += "每个噪声重复先集成模型种子，再计算指标，最后平均重复指标。"
    text += "置信区间对患者配对重采样，不把种子或噪声重复当成独立患者。\n\n"
    text += render_table(
        shown, ["setting", "modality", "level", "metric", "estimate", "ci95_low", "ci95_high"]
    )
    text += "\n\npaired_changes_ci.csv 中 change_minus_equal_late_change 表示 QMF 的性能变化"
    text += "减去等权融合的变化；对 AUC、准确率等指标，正值表示 QMF 下降较少。"
    text += "对 Brier 和对数损失则负值更好。区间包含零表示差异仍不明确。\n\n"
    text += "quality_changes_ci.csv 检查分支损失是否增大、q 是否下降。"
    text += "fold_quality.csv 给出各独立模型的相关性、协方差和 q 均值；"
    text += "均值与 1/3 的差只是描述性检查，不能据此认定满足总体期望条件。\n"
    if args.smoke:
        text = "# SMOKE：仅检查流程，不用于正式结论\n\n" + text
    (args.output / "report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--settings", nargs="+", choices=RUNS)
    parser.add_argument("--run-name", help="Evaluate equal_late/qmf from a new training run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--levels", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2])
    parser.add_argument("--noise-repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Invalid --run-name")
    if args.settings is None:
        args.settings = ["equal_late", "positive_margin"] if args.run_name else list(RUNS)
    if args.run_name and not set(args.settings) <= {"equal_late", "positive_margin"}:
        parser.error("--run-name supports equal_late and positive_margin only")
    if (
        0 not in args.levels
        or any(not np.isfinite(x) or x < 0 for x in args.levels)
        or args.noise_repeats < 1
        or args.bootstrap < 1
    ):
        parser.error("Include level 0; use finite nonnegative levels and positive repeat counts")
    for name in ["settings", "seeds", "folds", "levels"]:
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate --{name}")
    if not set(args.folds) <= set(range(5)):
        parser.error("Folds must be 0 through 4")
    if args.output is None:
        suffix = "qmf_robustness_smoke" if args.smoke else "qmf_robustness"
        if args.run_name:
            suffix += "_" + args.run_name
        if args.diagnostics_only:
            suffix = "qmf_ranking_diagnostics"
            if args.run_name:
                suffix += "_" + args.run_name
        args.output = args.project / "reports" / suffix
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(20260920, 2)
    rows, sources = run_inference(args)
    if not args.diagnostics_only:
        summarize(rows, args)
    save_json(
        args.output / "metadata.json",
        dict(
            run_name=args.run_name,
            settings=args.settings,
            seeds=args.seeds,
            folds=args.folds,
            levels=args.levels,
            noise_repeats=args.noise_repeats,
            bootstrap=args.bootstrap,
            smoke=args.smoke,
            diagnostics_only=args.diagnostics_only,
            source_sha256=sources,
            protocol="single-modality Gaussian corruption; frozen OOF models; no retraining",
            units="US: level*0.1 pixel SD; EMG/table: level training-standardized SD",
            uncertainty="patient bootstrap conditional on trained models and fixed noise realizations",
        ),
    )
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
