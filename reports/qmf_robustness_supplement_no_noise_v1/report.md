# QMF 模态质量下降实验

固定已训练模型；每次只扰动一种输入，保留完整患者级聚合。

每个噪声重复先集成模型种子，再计算指标，最后平均重复指标。置信区间对患者配对重采样，不把种子或噪声重复当成独立患者。

| setting | modality | level | metric | estimate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| positive_margin | emg | 0.0000 | roc_auc | 0.8710 | 0.7784 | 0.9453 |
| positive_margin | emg | 0.0000 | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| positive_margin | emg | 0.2500 | roc_auc | 0.8717 | 0.7791 | 0.9446 |
| positive_margin | emg | 0.2500 | balanced_accuracy | 0.7044 | 0.5933 | 0.8131 |
| positive_margin | emg | 0.5000 | roc_auc | 0.8347 | 0.7352 | 0.9182 |
| positive_margin | emg | 0.5000 | balanced_accuracy | 0.7163 | 0.6076 | 0.8194 |
| positive_margin | emg | 1.0000 | roc_auc | 0.8375 | 0.7389 | 0.9201 |
| positive_margin | emg | 1.0000 | balanced_accuracy | 0.7327 | 0.6272 | 0.8299 |
| positive_margin | emg | 2.0000 | roc_auc | 0.8429 | 0.7454 | 0.9236 |
| positive_margin | emg | 2.0000 | balanced_accuracy | 0.7381 | 0.6325 | 0.8356 |
| positive_margin | table | 0.0000 | roc_auc | 0.8710 | 0.7784 | 0.9453 |
| positive_margin | table | 0.0000 | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| positive_margin | table | 0.2500 | roc_auc | 0.8637 | 0.7735 | 0.9360 |
| positive_margin | table | 0.2500 | balanced_accuracy | 0.7125 | 0.6121 | 0.8137 |
| positive_margin | table | 0.5000 | roc_auc | 0.8506 | 0.7613 | 0.9243 |
| positive_margin | table | 0.5000 | balanced_accuracy | 0.7216 | 0.6255 | 0.8177 |
| positive_margin | table | 1.0000 | roc_auc | 0.8163 | 0.7307 | 0.8913 |
| positive_margin | table | 1.0000 | balanced_accuracy | 0.6808 | 0.5939 | 0.7696 |
| positive_margin | table | 2.0000 | roc_auc | 0.7529 | 0.6636 | 0.8352 |
| positive_margin | table | 2.0000 | balanced_accuracy | 0.6019 | 0.5316 | 0.6787 |
| positive_margin | us | 0.0000 | roc_auc | 0.8710 | 0.7784 | 0.9453 |
| positive_margin | us | 0.0000 | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| positive_margin | us | 0.2500 | roc_auc | 0.8817 | 0.7957 | 0.9504 |
| positive_margin | us | 0.2500 | balanced_accuracy | 0.7672 | 0.6585 | 0.8678 |
| positive_margin | us | 0.5000 | roc_auc | 0.8822 | 0.7966 | 0.9509 |
| positive_margin | us | 0.5000 | balanced_accuracy | 0.7916 | 0.6883 | 0.8866 |
| positive_margin | us | 1.0000 | roc_auc | 0.5057 | 0.3762 | 0.6337 |
| positive_margin | us | 1.0000 | balanced_accuracy | 0.5210 | 0.4271 | 0.6220 |
| positive_margin | us | 2.0000 | roc_auc | 0.5238 | 0.4071 | 0.6406 |
| positive_margin | us | 2.0000 | balanced_accuracy | 0.5000 | 0.5000 | 0.5000 |

paired_changes_ci.csv 中 change_minus_equal_late_change 表示 QMF 的性能变化减去等权融合的变化；对 AUC、准确率等指标，正值表示 QMF 下降较少。对 Brier 和对数损失则负值更好。区间包含零表示差异仍不明确。

quality_changes_ci.csv 检查分支损失是否增大、q 是否下降。fold_quality.csv 给出各独立模型的相关性、协方差和 q 均值；均值与 1/3 的差只是描述性检查，不能据此认定满足总体期望条件。
