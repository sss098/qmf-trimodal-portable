# ESWA manuscript

Revision date: 23 September 2026. Target: **Expert Systems with Applications**.

- [English manuscript](ESWA_manuscript.md)
- [Supplementary material](ESWA_supplementary.md)
- [Highlights](ESWA_highlights.md)
- [Submission assessment and remaining author inputs](notes/ESWA_submission_review.md)
- [Reference-led organization](notes/reference_revision_final.md)
- [Supplementary Data S1: all 95 candidate fields](sources/Supplementary_Data_S1_fields.csv)

The manuscript follows M3S as its principal organizational reference, with CAMAF and MMHA informing modality, ablation and robustness analyses. It contains 25 references, six main figures and six supplementary figures. The main experimental comparisons include three independently trained modalities, ultrasound–tabular fusion, trimodal equal weighting, dynamic fusion without added noise, and the full dynamic model.

## Main figures

| Figure | Purpose | File |
|---|---|---|
| 1 | Patient alignment, sampling and classifier evaluation | [PNG](figures/fig00_patient_workflow.png) |
| 2 | Encoders and quality-aware fusion | [Redraw reference](figures/fig01_framework_redraw_v2.png) |
| 3 | ROC curves and confusion matrices | [PNG](figures/fig03_final_classification.png), [PDF](figures/fig03_final_classification.pdf) |
| 4 | Independent modalities and paired ablations | [PNG](figures/fig04_final_contributions.png), [PDF](figures/fig04_final_contributions.pdf) |
| 5 | Quality–loss correlations and perturbation responses | [PNG](figures/fig05_quality.png), [PDF](figures/fig05_quality.pdf) |
| 6 | Matched fusion methods under modality perturbations | [PNG](figures/fig06_final_robustness.png), [PDF](figures/fig06_final_robustness.pdf) |

The supplementary figures cover preprocessing, logit scales, seed variation, the sEMG encoder, gradient propagation, and historical branch/ranking analyses. Explanatory illustrations retain the original manuscript style and serve as manual redraw references. Quantitative figures use real saved predictions and a shared blue, teal and gold style. Quantitative exports include PDF/SVG and 600 dpi PNG/TIFF.

## Reproduction and records

`scripts/make_submission_figures.py` reproduces the three updated quantitative figures; `scripts/make_figures.py` produces the retained diagnostic panels. The scripts use Python, matplotlib, pandas, scipy and scikit-learn; panel alignment checks use the installed nature-figure skill. Final metric and paired-difference snapshots are in `sources/final_metrics_ci.csv` and `sources/final_paired_differences_ci.csv`. Perturbation paired intervals are in `sources/perturbation_paired_changes_ci.csv`.

HPC results and 75 final fitted classifiers are stored in `../artifacts/hpc2_20260923/`, with transfer verification in its `analysis/transfer_verification.json`. This revision did not train additional models or alter classification code. Earlier manuscript versions are preserved in `archives/before_submission_revision_20260923/`; the frozen review packet is distinct from the corrected current manuscript.

Hospital, recruitment, ethics and author declaration placeholders remain for confirmation. The submission assessment records the remaining evidence issues. Older Chinese drafts and editorial notes are historical working material; the English files linked above are the current revision.
