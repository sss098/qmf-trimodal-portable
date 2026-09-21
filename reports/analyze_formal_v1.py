"""Recompute the retained formal_v1 audit and paired bootstrap report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    root = project / "artifacts/runs/formal_v1"
    results = {}
    all_contracts = []
    aligned = None
    for method in ["equal_late", "qmf", "tmc"]:
        ensemble = (
            pd.read_csv(root / method / "ensemble_predictions.csv")
            .sort_values("patient_id")
            .reset_index(drop=True)
        )
        anchor = ensemble[["patient_id", "fold", "label"]]
        if aligned is None:
            aligned = anchor
        else:
            pd.testing.assert_frame_equal(aligned, anchor)
        if len(ensemble) != 85 or ensemble.patient_id.nunique() != 85:
            raise ValueError("Bad OOF")
        branches = {m: [] for m in ["us", "emg", "table"]}
        epochs = []
        single = []
        qs = []
        for seed in [42, 43, 44]:
            run = root / method / f"seed{seed}"
            c = json.loads((run / "config.json").read_text())
            all_contracts.append(c)
            if c["smoke"]:
                raise ValueError("Smoke included")
            df = (
                pd.read_csv(run / "oof_predictions.csv")
                .sort_values("patient_id")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(df[["patient_id", "fold", "label"]], aligned)
            single.append(df.probability.to_numpy())
            for branch in branches:
                if method == "tmc":
                    a = df[[f"{branch}_alpha_0", f"{branch}_alpha_1"]].to_numpy()
                    p = a[:, 1] / a.sum(1)
                else:
                    p = softmax(df[[f"{branch}_score_0", f"{branch}_score_1"]].to_numpy(), axis=1)[
                        :, 1
                    ]
                branches[branch].append(p)
            if method == "qmf":
                qs.append(df[["us_quality", "emg_quality", "table_quality"]].to_numpy())
            for fold in range(5):
                d = run / f"fold{fold}"
                e = json.loads((d / "selection/selection.json").read_text())["selected_epoch"]
                h = json.loads((d / "selection/history.json").read_text())
                refit = json.loads((d / "refit/selection.json").read_text())["selected_epoch"]
                if refit != e:
                    raise ValueError("refit epoch mismatch")
                split = json.loads((d / "selection/split.json").read_text())
                test = json.loads((d / "test_split.json").read_text())["test"]
                if set(test) & (set(split["train"]) | set(split["validation"])):
                    raise ValueError("leakage")
                epochs.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "selected": e,
                        "stopped": len(h),
                        "first_val_loss": h[0]["validation_log_loss"],
                        "best_val_loss": h[e - 1]["validation_log_loss"],
                        "last_val_loss": h[-1]["validation_log_loss"],
                        "train_first": h[0]["train_loss"],
                        "train_last": h[-1]["train_loss"],
                    }
                )
        np.testing.assert_allclose(
            np.mean(single, axis=0), ensemble.probability, rtol=1e-12, atol=1e-12
        )
        y = ensemble.label.to_numpy()
        result = {
            "ensemble": json.loads((root / method / "summary.json").read_text())["ensemble"],
            "branch_auc_ensemble": {
                b: roc_auc_score(y, np.mean(ps, axis=0)) for b, ps in branches.items()
            },
            "branch_acc_ensemble": {
                b: accuracy_score(y, np.mean(ps, axis=0) >= 0.5) for b, ps in branches.items()
            },
            "epochs": epochs,
        }
        if qs:
            q = np.concatenate(qs)
            result["quality_mean"] = q.mean(0).tolist()
            result["quality_range"] = [q.min(0).tolist(), q.max(0).tolist()]
        results[method] = result
    base = {k: v for k, v in all_contracts[0].items() if k not in ["method", "seed"]}
    if any(
        {k: v for k, v in c.items() if k not in ["method", "seed"]} != base for c in all_contracts
    ):
        raise ValueError("Mismatched configurations")
    print(
        "AUDIT all45 complete; same config/code/weights/dataset; unique aligned85; refit epochs agree"
    )
    for m, r in results.items():
        print(m, "branchauc", r["branch_auc_ensemble"], "branchacc", r["branch_acc_ensemble"])
        e = pd.DataFrame(r["epochs"])
        print(
            "epochs selected",
            e.selected.tolist(),
            "median",
            e.selected.median(),
            "min",
            e.selected.min(),
            "max",
            e.selected.max(),
            "stopped100",
            int((e.stopped == 100).sum()),
        )
        if "quality_mean" in r:
            print("qmean", r["quality_mean"], "range", r["quality_range"])
    # Stratified paired bootstrap, fixed OOF probabilities, not retraining uncertainty.
    probs = {
        m: pd.read_csv(root / m / "ensemble_predictions.csv")
        .sort_values("patient_id")
        .probability.to_numpy()
        for m in results
    }
    y = aligned.label.to_numpy()
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(20260915)
    b = 10000
    wp = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)), size=b)
    wn = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)), size=b)
    aucs = {}
    for m, p in probs.items():
        dif = p[pos, None] - p[None, neg]
        comparison = (dif > 0).astype(float) + 0.5 * (dif == 0)
        aucs[m] = np.einsum("bi,ij,bj->b", wp, comparison, wn, optimize=True) / (
            len(pos) * len(neg)
        )
    paired = []
    for a, c in [("qmf", "equal_late"), ("tmc", "equal_late"), ("tmc", "qmf")]:
        delta = aucs[a] - aucs[c]
        ca = (probs[a] >= 0.5) == y
        cb = (probs[c] >= 0.5) == y
        row = {
            "comparison": a + " minus " + c,
            "auc_difference": roc_auc_score(y, probs[a]) - roc_auc_score(y, probs[c]),
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "a_correct_b_wrong": int((ca & ~cb).sum()),
            "a_wrong_b_correct": int((~ca & cb).sum()),
        }
        paired.append(row)
        print("PAIR", row)
    report = {
        "audit": "All 45 outer runs complete, identical config/data/code/weights, aligned85 unique OOF patients, disjoint patient splits, selection/refit epochs matched, ensemble probabilities verified",
        "methods": results,
        "paired_bootstrap": paired,
        "bootstrap": {
            "replicates": b,
            "seed": 20260915,
            "scope": "class-stratified paired patient bootstrap, fixed ensemble OOF predictions; no retraining, no multiplicity correction",
        },
    }
    (project / "reports/formal_v1_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
