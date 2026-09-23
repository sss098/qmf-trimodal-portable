"""Submission figures from verified OOF predictions; no synthetic observations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from make_figures import AMBER, BLUE, INK, TEAL, finish, title
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
SRC = ROOT / "sources"
ANALYSIS = PROJECT / "artifacts/hpc2_20260923/analysis"
CI = pd.read_csv(ANALYSIS / "metrics_ci.csv")
DIFF = pd.read_csv(ANALYSIS / "paired_differences_ci.csv")
COLORS = {
    "main": BLUE,
    "table": TEAL,
    "equal": "#87939F",
    "us": "#7295B3",
    "emg": AMBER,
    "us_table": "#8976AA",
    "no_noise": "#AB8274",
}
LABELS = {
    "main": "Trimodal dynamic",
    "table": "Tabular only",
    "equal": "Trimodal equal",
    "us": "Ultrasound only",
    "emg": "sEMG only",
    "us_table": "Ultrasound + tabular",
    "no_noise": "Without added noise",
}


def load(name):
    if name in ["main", "equal"]:
        method = "qmf" if name == "main" else "equal_late"
        p = PROJECT / f"artifacts/runs/noise_aug_v1/{method}/ensemble_predictions.csv"
    else:
        method = "equal_late" if name in ["us", "emg", "table"] else "qmf"
        p = (
            PROJECT
            / f"artifacts/hpc2_20260923/runs/necessary_{name}_v1/{method}/ensemble_predictions.csv"
        )
    d = pd.read_csv(p).sort_values("patient_id").reset_index(drop=True)
    assert len(d) == d.patient_id.nunique() == 85
    d.to_csv(SRC / f"final_{name}_predictions.csv", index=False)
    return d


def classification():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.65))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.10, top=0.94, wspace=0.30, hspace=0.58)
    for name in ["main", "table", "equal"]:
        d = load(name)
        x, y, _ = roc_curve(d.label, d.probability)
        axs[0, 0].plot(
            x,
            y,
            lw=1.6,
            color=COLORS[name],
            label=f"{LABELS[name]} ({roc_auc_score(d.label, d.probability):.3f})",
        )
    ax = axs[0, 0]
    ax.plot([0, 1], [0, 1], ls="--", color="#C6CDD2", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=6)
    title(ax, "a", "Discrimination")
    for ax, name, letter in zip(
        [axs[0, 1], axs[1, 0], axs[1, 1]], ["main", "table", "equal"], "bcd"
    ):
        d = load(name)
        cm = confusion_matrix(d.label, d.probability >= 0.5, labels=[0, 1])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{norm[i, j]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if norm[i, j] > 0.65 else INK,
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            xlabel="Predicted class",
            ylabel="True class",
        )
        title(ax, letter, LABELS[name])
    finish(fig, "fig03_final_classification")


def contributions():
    names = ["us", "emg", "table", "us_table", "main"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.12, top=0.93, wspace=0.78, hspace=0.62)
    for j, name in enumerate(names):
        r = CI[(CI.model == name) & (CI.metric == "roc_auc")].iloc[0]
        axs[0, 0].errorbar(
            r.estimate,
            j,
            xerr=[[r.estimate - r.lower], [r.upper - r.estimate]],
            fmt="o",
            color=COLORS[name],
            capsize=2,
            ms=4,
            lw=1.2,
        )
        for metric, off, mark in [("sensitivity", -0.12, "o"), ("specificity", 0.12, "s")]:
            r = CI[(CI.model == name) & (CI.metric == metric)].iloc[0]
            axs[0, 1].errorbar(
                r.estimate,
                j + off,
                xerr=[[max(0, r.estimate - r.lower)], [max(0, r.upper - r.estimate)]],
                fmt=mark,
                color=COLORS[name],
                capsize=2,
                ms=3.5,
                lw=0.9,
                mfc=COLORS[name] if mark == "o" else "white",
            )
    for ax in axs[0]:
        ax.set(
            yticks=range(len(names)),
            yticklabels=[LABELS[n] for n in names],
            ylim=(4.6, -0.6),
            xlim=(-0.03, 1.03),
        )
    axs[0, 0].axvline(0.5, color="#B7C1CA", ls="--", lw=0.8)
    axs[0, 0].set_xlabel("ROC AUC and 95% CI")
    axs[0, 1].set_xlabel("Rate and 95% CI")
    title(axs[0, 0], "a", "Independent modality comparisons")
    title(axs[0, 1], "b", "Sensitivity and specificity")
    from matplotlib.lines import Line2D

    axs[0, 1].legend(
        handles=[
            Line2D([], [], color=INK, marker="o", ls="", label="Sensitivity"),
            Line2D([], [], color=INK, marker="s", mfc="white", ls="", label="Specificity"),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.42),
        ncol=2,
        fontsize=6,
    )
    controls = ["table", "us_table", "no_noise", "equal"]
    for ax, metric, letter, label in [
        (axs[1, 0], "roc_auc", "c", "AUC difference"),
        (axs[1, 1], "accuracy", "d", "Accuracy difference (percentage points)"),
    ]:
        factor = 100 if metric == "accuracy" else 1
        for j, name in enumerate(controls):
            r = DIFF[(DIFF.comparison == f"main - {name}") & (DIFF.metric == metric)].iloc[0]
            ax.errorbar(
                r.difference * factor,
                j,
                xerr=[[(r.difference - r.lower) * factor], [(r.upper - r.difference) * factor]],
                fmt="o",
                color=COLORS[name],
                capsize=3,
                ms=4,
                lw=1.2,
            )
        ax.axvline(0, color="#A6B2BB", ls="--", lw=0.9)
        ax.set(
            yticks=range(4),
            yticklabels=["vs " + LABELS[n] for n in controls],
            ylim=(3.6, -0.6),
            xlabel=label,
        )
        if metric == "roc_auc":
            ax.set_xlim(-0.035, 0.055)
        else:
            ax.set_xlim(-2, 16)
        title(ax, letter, "Dynamic fusion: paired effect")
    finish(fig, "fig04_final_contributions")


def robustness():
    data = pd.read_csv(SRC / "enhanced_metrics_ci.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.74, wspace=0.36)
    for ax, mod, label, letter in zip(
        axs, ["us", "emg", "table"], ["Ultrasound", "sEMG", "Tabular"], "abc"
    ):
        for setting, name in [("positive_margin", "main"), ("equal_late", "equal")]:
            d = data[
                (data.setting == setting) & (data.modality == mod) & (data.metric == "roc_auc")
            ].sort_values("level")
            ax.plot(d.level, d.estimate, "o-", color=COLORS[name], lw=1.6, ms=3, label=LABELS[name])
            ax.fill_between(d.level, d.ci95_low, d.ci95_high, color=COLORS[name], alpha=0.12, lw=0)
        ax.set(
            xlabel="Perturbation level", ylabel="ROC AUC", ylim=(0.55, 1.02), xticks=[0, 0.5, 1, 2]
        )
        title(ax, letter, label)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.53, 0.995))
    finish(fig, "fig06_final_robustness")


if __name__ == "__main__":
    classification()
    contributions()
    robustness()
    CI.to_csv(SRC / "final_metrics_ci.csv", index=False)
    DIFF.to_csv(SRC / "final_paired_differences_ci.csv", index=False)
