"""Reproduce manuscript figures from frozen patient-level result snapshots."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT, QA = ROOT / "sources", ROOT / "figures", ROOT / "notes/figure_qa"
QA.mkdir(exist_ok=True)
sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, TEAL, AMBER, GRAY = "#2676B8", "#328C86", "#BF913F", "#937E70"
INK = "#243747"
MODS = ["us", "emg", "table"]
NAMES = ["Ultrasound", "sEMG", "Tabular"]
COLORS = [BLUE, AMBER, TEAL]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.edgecolor": "#7C8992", "text.color": INK,
    "axes.labelcolor": INK, "legend.frameon": False,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})


def finish(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=QA / f"{name}.alignment.json", strict=True
    )
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold", pad=10)
    ax.set_axisbelow(True)


def bootstrap_auc(y, probabilities, draws):
    """Vectorized rank AUC with ties, using a shared patient bootstrap."""
    yy = np.asarray(y)[draws]
    pp = np.asarray(probabilities)[draws]
    ranks = rankdata(pp, axis=1)
    n1 = yy.sum(1)
    return ((ranks * yy).sum(1) - n1 * (n1 + 1) / 2) / (n1 * (yy.shape[1] - n1))


def patient_draws(y):
    rng = np.random.default_rng(20260922)
    return np.concatenate([
        rng.choice(np.flatnonzero(np.asarray(y) == label), (10000, int((np.asarray(y) == label).sum())))
        for label in [0, 1]
    ], axis=1)


def classification(pred, ci):
    clean = pred[(pred.modality == "us") & (pred.level == 0) & (pred["repeat"] == 0)]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.7))
    fig.subplots_adjust(left=.10, right=.97, bottom=.10, top=.94, wspace=.45, hspace=.58)
    for setting, name, color in [("positive_margin", "Dynamic fusion", BLUE), ("equal_late", "Equal-weight fusion", TEAL)]:
        d = clean[clean.setting == setting]
        assert len(d) == 85 and d.patient_id.nunique() == 85
        x, y, _ = roc_curve(d.label, d.probability)
        axs[0, 0].plot(x, y, color=color, lw=1.6, label=f"{name}: {roc_auc_score(d.label, d.probability):.3f}")
    ax = axs[0, 0]
    ax.plot([0, 1], [0, 1], color="#C1C8CD", ls="--", lw=.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=6.5)
    title(ax, "a", "Patient-level ROC curves")
    metrics = ["accuracy", "sensitivity", "specificity", "balanced_accuracy"]
    for setting, color, off in [("positive_margin", BLUE, -.1), ("equal_late", TEAL, .1)]:
        d = ci[(ci.setting == setting) & (ci.modality == "us") & (ci.level == 0)].set_index("metric").loc[metrics]
        axs[0, 1].errorbar(d.estimate, np.arange(4) + off,
                         xerr=[d.estimate-d.ci95_low, d.ci95_high-d.estimate],
                         fmt="o", color=color, capsize=2, ms=3.5, lw=1)
    ax = axs[0, 1]
    ax.set(yticks=range(4), yticklabels=["Accuracy", "Sensitivity", "Specificity", "Balanced acc."],
           xlim=(.25, 1.03), xlabel="Estimate and 95% CI")
    ax.invert_yaxis()
    title(ax, "b", "Threshold-based performance")
    for ax, setting, label, letter in [(axs[1, 0], "positive_margin", "Dynamic fusion", "c"),
                                      (axs[1, 1], "equal_late", "Equal-weight fusion", "d")]:
        d = clean[clean.setting == setting]
        cm = confusion_matrix(d.label, d.probability >= .5, labels=[0, 1])
        normalized = cm / cm.sum(1, keepdims=True)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({normalized[i,j]:.1%})", ha="center", va="center",
                        fontsize=9, color="white" if normalized[i,j] > .65 else INK)
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["Negative", "Positive"],
               yticklabels=["Negative", "Positive"], xlabel="Predicted class", ylabel="True class")
        title(ax, letter, label)
    finish(fig, "fig03_classification")


def robustness(ci):
    old = pd.read_csv(SRC / "original_metrics_ci.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.73, wspace=.35)
    for ax, mod, name, letter in zip(axs, MODS, NAMES, "abc"):
        for frame, setting, label, color, ls in [
            (ci, "positive_margin", "Dynamic + enhanced training", BLUE, "-"),
            (ci, "equal_late", "Equal + enhanced training", TEAL, "-"),
            (old, "positive_margin", "Dynamic + earlier training", GRAY, "--"),
        ]:
            d = frame[(frame.setting == setting) & (frame.modality == mod) & (frame.metric == "roc_auc")].sort_values("level")
            ax.plot(d.level, d.estimate, "o", ls=ls, color=color, lw=1.4, ms=3, label=label)
            ax.fill_between(d.level, d.ci95_low, d.ci95_high, color=color, alpha=.10, lw=0)
        ax.set(xlabel="Perturbation level", ylabel="ROC AUC", ylim=(.35, 1.02), xticks=[0,.5,1,2])
        title(ax, letter, name)
    h,l=axs[0].get_legend_handles_labels()
    fig.legend(h,l,loc="upper center",bbox_to_anchor=(.52,.995),ncol=2,fontsize=6.8)
    finish(fig, "fig04_robustness")


def quality(pred):
    fq = pd.read_csv(SRC / "enhanced_fold_quality.csv")
    fq = fq[fq.setting == "positive_margin"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.1))
    fig.subplots_adjust(left=.11, right=.98, bottom=.10, top=.93, wspace=.36, hspace=.61)
    for ax, metric, letter in [(axs[0,0], "spearman", "a"), (axs[0,1], "pearson", "b")]:
        for j,(mod,col) in enumerate(zip(MODS,COLORS)):
            vals=fq[fq.modality==mod].sort_values(["seed","fold"])[metric]
            ax.boxplot(vals, positions=[j], widths=.48, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, alpha=.18, edgecolor=col),
                       medianprops=dict(color=INK, linewidth=1.2),
                       whiskerprops=dict(color=col, linewidth=.8), capprops=dict(color=col, linewidth=.8))
            ax.scatter(j+np.linspace(-.12,.12,len(vals)),vals,s=13,color=col,alpha=.8)
            ax.plot([j-.22,j+.22],[vals.median()]*2,color=INK,lw=1.3)
        ax.axhline(0,color="#AAB4BC",ls="--",lw=.8)
        ax.set(xticks=range(3),xticklabels=NAMES,ylabel="Within-fold correlation",ylim=(-1.05,.9))
        title(ax,letter,f"Quality–loss: {metric.capitalize()}")
    for ax, metric, label, letter in [(axs[1,0],"quality","Mean quality coefficient","c"),
                                     (axs[1,1],"branch_loss","Mean branch cross-entropy","d")]:
        for mod,name,color in zip(MODS,NAMES,COLORS):
            d=pred[(pred.setting=="positive_margin") & (pred.modality==mod)]
            avg=d.groupby("level")[metric].mean()
            ax.plot(avg.index,avg.values,"o-",color=color,lw=1.5,ms=3,label=name)
        ax.set(xlabel="Perturbation level",ylabel=label,xticks=[0,.5,1,2])
        ax.set_ylim((0,.5) if metric=="quality" else (.5,1.12))
        title(ax,letter,"Quality response" if metric=="quality" else "Loss response")
    axs[1,0].legend(loc="upper right",fontsize=6.5)
    finish(fig,"fig05_quality")


def components():
    raw=pd.read_csv(SRC/"enhanced_branch_predictions.csv")
    patient=raw.groupby("patient_id")["label"].first().sort_index()
    probs={}
    for mod in MODS:
        raw[mod+"_prob"]=expit(raw[mod+"_score_1"]-raw[mod+"_score_0"])
        probs[mod]=raw.groupby("patient_id")[mod+"_prob"].mean().reindex(patient.index).to_numpy()
    probs["fusion"]=raw.groupby("patient_id").probability.mean().reindex(patient.index).to_numpy()
    draws=patient_draws(patient)
    rows=[]
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
    fig.subplots_adjust(left=.12,right=.98,bottom=.22,top=.86,wspace=.75)
    for j,(mod,name,color) in enumerate(zip(MODS+["fusion"],NAMES+["Fused"],COLORS+[BLUE])):
        estimate=roc_auc_score(patient,probs[mod]);values=bootstrap_auc(patient,probs[mod],draws)
        lo,hi=np.quantile(values,[.025,.975])
        if mod == "fusion":
            archived=pd.read_csv(SRC/"enhanced_metrics_ci.csv")
            archived=archived[(archived.setting=="positive_margin") & (archived.modality=="us") & (archived.level==0) & (archived.metric=="roc_auc")].iloc[0]
            lo,hi=archived.ci95_low,archived.ci95_high
        rows.append(dict(output=mod,auc=estimate,ci_low=lo,ci_high=hi))
        axs[0].barh(j,estimate,color=color,alpha=.6,height=.55,edgecolor=color,linewidth=.6)
        axs[0].errorbar(estimate,j,xerr=[[estimate-lo],[hi-estimate]],fmt="o",color=color,capsize=3,ms=4,lw=1.2)
    axs[0].set(yticks=range(4),yticklabels=NAMES+["Fused"],xlim=(0,1.0),xlabel="ROC AUC and 95% CI")
    axs[0].invert_yaxis();axs[0].axvline(.5,color="#C3C9CE",ls="--",lw=.8)
    title(axs[0],"a","Outputs of the joint model")
    ab={}
    for name in ["no_ranking","original_ranking","positive_margin"]:
        d=pd.read_csv(SRC/f"ranking_{name}.csv").set_index("patient_id").reindex(patient.index)
        assert (d.label.to_numpy()==patient.to_numpy()).all()
        ab[name]=d.probability.to_numpy()
    reference=bootstrap_auc(patient,ab["positive_margin"],draws)
    diffs=[]
    for j,(key,name) in enumerate([("no_ranking","vs no ranking"),("original_ranking","vs original ranking")]):
        samples=reference-bootstrap_auc(patient,ab[key],draws)
        estimate=roc_auc_score(patient,ab["positive_margin"])-roc_auc_score(patient,ab[key])
        lo,hi=np.quantile(samples,[.025,.975]);diffs.append(dict(comparison=key,estimate=estimate,ci_low=lo,ci_high=hi))
        axs[1].errorbar(estimate,j,xerr=[[estimate-lo],[hi-estimate]],fmt="o",color=BLUE,capsize=3,ms=4,lw=1.2)
    axs[1].axvline(0,color="#B8C1C7",ls="--",lw=.8)
    axs[1].set(yticks=[0,1],yticklabels=["vs no ranking","vs original ranking"],ylim=(-.7,1.7),
               xlim=(-.05,.06),xticks=[-.04,0,.04],xlabel="AUC difference and paired 95% CI")
    axs[1].invert_yaxis();title(axs[1],"b","Positive-margin ranking")
    pd.DataFrame(rows).to_csv(SRC/"branch_auc_ci.csv",index=False)
    pd.DataFrame(diffs).to_csv(SRC/"ranking_auc_differences_ci.csv",index=False)
    finish(fig,"fig06_components")


def scale_and_seed():
    raw=pd.read_csv(SRC/"enhanced_branch_predictions.csv")
    rows=[]
    for mod in MODS:
        for (seed,fold),d in raw.groupby(["seed","fold"]):
            rows.append(dict(modality=mod,seed=seed,fold=fold,
                             quality=d[mod+"_quality"].median(),
                             offset=((d[mod+"_score_0"]+d[mod+"_score_1"])/2).median(),
                             gap=(d[mod+"_score_1"]-d[mod+"_score_0"]).abs().median()))
    frame=pd.DataFrame(rows);frame.to_csv(SRC/"enhanced_scale_medians.csv",index=False)
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.65))
    fig.subplots_adjust(left=.08,right=.985,bottom=.22,top=.82,wspace=.43)
    for ax,metric,label,letter in zip(axs,["quality","offset","gap"],["Quality coefficient","Common logit offset","Absolute class-logit gap"],"abc"):
        for j,(mod,col) in enumerate(zip(MODS,COLORS)):
            v=frame[frame.modality==mod][metric]
            ax.boxplot(v, positions=[j], widths=.48, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col, alpha=.18, edgecolor=col),
                       medianprops=dict(color=INK, linewidth=1.2),
                       whiskerprops=dict(color=col, linewidth=.8), capprops=dict(color=col, linewidth=.8))
            ax.scatter(j+np.linspace(-.12,.12,len(v)),v,s=12,color=col,alpha=.8)
            ax.plot([j-.23,j+.23],[v.median()]*2,color=INK,lw=1.3)
        ax.set(xticks=range(3),xticklabels=["US","sEMG","Table"],ylabel="Model-level median")
        title(ax,letter,label)
    finish(fig,"figS02_scale")
    d=pd.read_csv(SRC/"seed_metrics.csv")
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
    fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.84,wspace=.35)
    markers=["o","s","^"]
    for ax,metric,label,letter in zip(axs,["roc_auc","accuracy"],["ROC AUC","Accuracy"],"ab"):
        for seed,m in zip([42,43,44],markers):
            vals=d[d.seed==seed].set_index("method").loc[["equal_late","qmf"],metric]
            ax.plot([0,1],vals,marker=m,ms=4,color=[GRAY,TEAL,BLUE][seed-42],lw=1,label=f"Seed {seed}")
        ax.set(xticks=[0,1],xticklabels=["Equal-weight fusion","Dynamic fusion"],ylabel=label,xlim=(-.25,1.25),ylim=(.74,.96))
        title(ax,letter,label)
    axs[1].legend(loc="upper left",fontsize=6.5)
    finish(fig,"figS03_seed_stability")


def main():
    pred=pd.read_csv(SRC/"enhanced_predictions.csv")
    ci=pd.read_csv(SRC/"enhanced_metrics_ci.csv")
    classification(pred,ci);components();quality(pred);robustness(ci);scale_and_seed()
    print("Rendered six quantitative figure sets with source data and alignment audits.")


if __name__ == "__main__":
    main()
