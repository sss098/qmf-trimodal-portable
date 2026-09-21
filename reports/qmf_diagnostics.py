"""Patient-level diagnostics of retained QMF predictions; no model training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import pearsonr, rankdata, spearmanr

BRANCHES = ("us", "emg", "table")
METHODS = ("equal_late", "qmf", "tmc")
SEEDS = (42, 43, 44)


def correlations(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def metric_arrays(y, p):
    """Evaluate rows of patient probabilities, including average ranks for AUC ties."""
    y = np.broadcast_to(y, p.shape)
    positive = y.sum(-1)
    negative = y.shape[-1] - positive
    pred = p >= 0.5
    tp = (pred * y).sum(-1)
    tn = ((~pred) * (1 - y)).sum(-1)
    sensitivity = tp / positive
    specificity = tn / negative
    ranks = rankdata(p, axis=-1)
    eps = np.finfo(float).eps
    clipped = np.clip(p, eps, 1 - eps)
    return {
        "roc_auc": ((ranks * y).sum(-1) - positive * (positive + 1) / 2) / (positive * negative),
        "accuracy": (tp + tn) / y.shape[-1],
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": 2 * tp / (2 * tp + (negative - tn) + (positive - tp)),
        "brier": ((p - y) ** 2).mean(-1),
        "log_loss": -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean(-1),
    }


def bootstrap_indices(y, repeats, seed):
    """Keep class counts fixed, resampling whole patients with replacement."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    # Same multinomial draws as analyze_formal_v1.py for reproducible prior AUC CIs.
    wp = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=repeats)
    wn = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=repeats)
    return np.stack(
        [np.concatenate([np.repeat(pos, p), np.repeat(neg, n)]) for p, n in zip(wp, wn)]
    )


def qmf_probability(z):
    q = logsumexp(z, axis=-1) / 10
    return softmax((z * q[..., None]).sum(-2), axis=-1)[..., 1]


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def load_predictions(root):
    frames = {}
    anchor = None
    sources = {}
    for method in METHODS:
        for seed in SEEDS:
            path = root / method / f"seed{seed}" / "oof_predictions.csv"
            d = pd.read_csv(path).sort_values("patient_id").reset_index(drop=True)
            key = d[["patient_id", "fold", "label"]]
            if anchor is None:
                anchor = key
            pd.testing.assert_frame_equal(key, anchor)
            if len(d) != 85 or d.patient_id.nunique() != 85:
                raise ValueError("Expected 85 distinct patients")
            if not np.isfinite(d.select_dtypes("number").to_numpy()).all():
                raise ValueError("Nonfinite predictions")
            if not d.probability.between(0, 1).all():
                raise ValueError("Invalid probability")
            frames[method, seed] = d
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return frames, anchor, sources


def quality_records(frames):
    rows = []
    for seed in SEEDS:
        d = frames["qmf", seed]
        y = d.label.to_numpy(int)
        for m in BRANCHES:
            z = d[[f"{m}_score_0", f"{m}_score_1"]].to_numpy()
            q = d[f"{m}_quality"].to_numpy()
            np.testing.assert_allclose(q, logsumexp(z, axis=1) / 10, rtol=2e-6, atol=1e-7)
            loss = logsumexp(z, axis=1) - z[np.arange(len(y)), y]
            prob = softmax(z, axis=1)[:, 1]
            r = d[["patient_id", "fold", "label"]].copy()
            r["seed"] = seed
            r["modality"] = m
            for name, value in dict(
                logit0=z[:, 0],
                logit1=z[:, 1],
                offset=z.mean(1),
                gap=z[:, 1] - z[:, 0],
                abs_gap=np.abs(z[:, 1] - z[:, 0]),
                quality=q,
                loss=loss,
                probability=prob,
                correct=(prob >= 0.5) == y,
            ).items():
                r[name] = value
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def correlation_tables(records):
    rows = []
    for (seed, modality), d in records.groupby(["seed", "modality"]):
        for fold in [-1, *range(5)]:
            f = d if fold == -1 else d[d.fold == fold]
            for label in [-1, 0, 1]:
                group = f if label == -1 else f[f.label == label]
                for target in ["loss", "offset", "abs_gap"]:
                    pearson, spearman = correlations(
                        group.quality.to_numpy(), group[target].to_numpy()
                    )
                    rows.append(
                        dict(
                            seed=seed,
                            modality=modality,
                            fold=fold,
                            label=label,
                            target=target,
                            n=len(group),
                            pearson=pearson,
                            spearman=spearman,
                        )
                    )
    return pd.DataFrame(rows)


def scale_table(records):
    rows = []
    for (seed, modality), d in records.groupby(["seed", "modality"]):
        for fold in [-1, *range(5)]:
            f = d if fold == -1 else d[d.fold == fold]
            for correct in [-1, 0, 1]:
                group = f if correct == -1 else f[f.correct == bool(correct)]
                if group.empty:
                    continue
                for variable in ["logit0", "logit1", "offset", "gap", "abs_gap", "quality", "loss"]:
                    v = group[variable].to_numpy()
                    quantiles = np.quantile(v, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
                    rows.append(
                        dict(
                            seed=seed,
                            modality=modality,
                            fold=fold,
                            correct=correct,
                            variable=variable,
                            n=len(v),
                            mean=v.mean(),
                            std=v.std(),
                            **dict(
                                zip(["min", "p05", "p25", "median", "p75", "p95", "max"], quantiles)
                            ),
                        )
                    )
    return pd.DataFrame(rows)


def performance_tables(frames, y, indices):
    summary, differences = [], []
    point, boot = {}, {}
    for method in METHODS:
        p = np.mean([frames[method, seed].probability.to_numpy() for seed in SEEDS], axis=0)
        point[method] = metric_arrays(y, p[None])
        boot[method] = metric_arrays(y[indices], p[indices])
        for metric, values in boot[method].items():
            low, high = interval(values)
            summary.append(
                dict(
                    method=method,
                    metric=metric,
                    estimate=point[method][metric][0],
                    ci95_low=low,
                    ci95_high=high,
                )
            )
    for a, b in [("qmf", "equal_late"), ("tmc", "qmf"), ("tmc", "equal_late")]:
        for metric in boot[a]:
            low, high = interval(boot[a][metric] - boot[b][metric])
            differences.append(
                dict(
                    comparison=f"{a} minus {b}",
                    metric=metric,
                    estimate=point[a][metric][0] - point[b][metric][0],
                    ci95_low=low,
                    ci95_high=high,
                )
            )
    return pd.DataFrame(summary), pd.DataFrame(differences)


def shift_diagnostics(frames, y):
    arrays = []
    for seed in SEEDS:
        d = frames["qmf", seed]
        arrays.append(
            np.stack([d[[f"{m}_score_0", f"{m}_score_1"]].to_numpy() for m in BRANCHES], axis=1)
        )
        np.testing.assert_allclose(qmf_probability(arrays[-1]), d.probability, rtol=2e-6, atol=2e-7)
    base = np.mean([qmf_probability(z) for z in arrays], axis=0)
    rows = []
    for m, name in enumerate(BRANCHES):
        for shift in [-5.0, -1.0, 1.0, 5.0]:
            ps = []
            for z in arrays:
                new = z.copy()
                new[:, m, :] += shift
                np.testing.assert_allclose(
                    softmax(new[:, m], axis=-1), softmax(z[:, m], axis=-1), atol=1e-14
                )
                ps.append(qmf_probability(new))
            p = np.mean(ps, axis=0)
            metrics = metric_arrays(y, p[None])
            rows.append(
                dict(
                    modality=name,
                    shift=shift,
                    q_change=shift / 10,
                    changed_predictions=int(((p >= 0.5) != (base >= 0.5)).sum()),
                    mean_abs_probability_change=float(np.abs(p - base).mean()),
                    **{k: v[0] for k, v in metrics.items()},
                )
            )
    return pd.DataFrame(rows)


def render_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                f"{row[c]:.4f}" if isinstance(row[c], (float, np.floating)) else str(row[c])
                for c in columns
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(out, corr, performance, differences, shifts, repeats):
    text = [
        "# QMF 质量与统计补充分析",
        "使用 formal_v1 已保存的患者折外预测。三种方法共用 85 名患者；每种方法的最终概率取三个种子的平均。",
        "## 1. q 与损失的关系",
        "Pearson 衡量线性关系，Spearman 衡量排序关系，预期均为负。下表先给出每个种子汇总五折的结果：",
    ]
    overall = corr[(corr.fold == -1) & (corr.label == -1) & (corr.target == "loss")]
    text.append(render_table(overall, ["seed", "modality", "n", "pearson", "spearman"]))
    within = corr[(corr.fold >= 0) & (corr.label == -1) & (corr.target == "loss")]
    rows = []
    for m, d in within.groupby("modality"):
        rows.append(
            dict(
                modality=m,
                folds=len(d),
                negative_pearson=int((d.pearson < 0).sum()),
                negative_spearman=int((d.spearman < 0).sum()),
                median_pearson=d.pearson.median(),
                median_spearman=d.spearman.median(),
            )
        )
    text.extend(
        [
            "每个种子有 5 个分别训练的模型。以下统计 15 个模型各自测试折内的相关性，避免混合模型尺度；这 15 个系数用于描述稳定性，同一患者的三个种子不作为独立样本。",
            render_table(pd.DataFrame(rows), list(rows[0])),
            "按正负类拆分的结果见 correlations.csv（label=0/1）；fold=-1、label=-1 分别表示汇总五折、合并类别。每折仅 17 人，重点看方向和稳定性。",
        ]
    )
    text.extend(
        [
            "## 2. 分类分数尺度",
            "offset 为两个类别分数的均值，abs_gap 为两类分数差的绝对值。二分类质量可写成 q = [offset + log(2 cosh(gap/2))]/10，因此 q 同时包含整体偏移和类别分离程度。",
            "下面列出每个种子汇总五折时，q 与整体偏移的相关性：",
            render_table(
                corr[(corr.fold == -1) & (corr.label == -1) & (corr.target == "offset")],
                ["seed", "modality", "pearson", "spearman"],
            ),
            "scale_distribution.csv 按种子、模态、折及预测正确/错误输出分数与 q 的均值、标准差和分位数；correlations.csv 同时包含 q 与 abs_gap 的关系。",
            "### 平移敏感性",
            "给指定模态的两个分数同时加常数，单分支概率与交叉熵严格不变，q 增加常数的十分之一。下表显示三种子集成后的融合预测变化。此检查衡量结构上的尺度敏感性，不用测试集选择新偏移参数。",
            render_table(
                shifts,
                [
                    "modality",
                    "shift",
                    "changed_predictions",
                    "mean_abs_probability_change",
                    "roc_auc",
                    "accuracy",
                ],
            ),
        ]
    )
    text.extend(
        [
            "## 3. 患者级 95% 置信区间",
            f"按正负类分层、有放回抽取患者 {repeats:,} 次，取 2.5% 和 97.5% 分位数。各方法使用完全相同的抽样索引；三个种子先集成，独立单位是患者。区间针对固定折外预测，不包含重新训练的不确定性。",
            render_table(performance, ["method", "metric", "estimate", "ci95_low", "ci95_high"]),
            "### 配对差值",
            render_table(
                differences, ["comparison", "metric", "estimate", "ci95_low", "ci95_high"]
            ),
            "AUC、准确率、敏感度、特异度、平衡准确率和 F1 越高越好；Brier 分数与对数损失越低越好。这里报告探索性区间，不作多重比较显著性宣称。",
            "## 4. 尚需训练的消融",
            "当前结果可以检查 q 与损失的关系及尺度敏感性。排序项本身的贡献需要比较无排序、原排序、当前正间隔排序；训练命令见 docs/qmf_ablations.md。",
        ]
    )
    (out / "report.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project / "artifacts/runs/formal_v1")
    parser.add_argument("--output", type=Path, default=project / "reports/qmf_diagnostics")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    frames, anchor, sources = load_predictions(args.root)
    y = anchor.label.to_numpy(int)
    records = quality_records(frames)
    corr = correlation_tables(records)
    scales = scale_table(records)
    indices = bootstrap_indices(y, args.replicates, args.seed)
    performance, differences = performance_tables(frames, y, indices)
    shifts = shift_diagnostics(frames, y)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in dict(
        patient_quality=records,
        correlations=corr,
        scale_distribution=scales,
        metrics_ci=performance,
        paired_differences_ci=differences,
        shift_sensitivity=shifts,
    ).items():
        data.to_csv(args.output / f"{name}.csv", index=False)
    metadata = dict(
        patients=len(y),
        seeds=SEEDS,
        replicates=args.replicates,
        bootstrap_seed=args.seed,
        unit="patient",
        method="class-stratified paired percentile bootstrap",
        scope="fixed ensemble OOF predictions; no retraining",
        source_sha256=sources,
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(args.output, corr, performance, differences, shifts, args.replicates)
    print(f"Wrote diagnostics for {len(y)} patients to {args.output}")


if __name__ == "__main__":
    main()
