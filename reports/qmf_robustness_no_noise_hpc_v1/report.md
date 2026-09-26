> 本结果为旧HPC no-noise模型，以原HPC缓存复现。s031肌电预处理与本地主实验不同，不能作为严格单因素噪声消融；正式比较请使用补充训练命令生成的 supplement_no_noise_v1。

# QMF 模态质量下降实验

固定已训练模型；每次只扰动一种输入，保留完整患者级聚合。

每个噪声重复先集成模型种子，再计算指标，最后平均重复指标。置信区间对患者配对重采样，不把种子或噪声重复当成独立患者。

| setting | modality | level | metric | estimate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| positive_margin | emg | 0.0000 | roc_auc | 0.8710 | 0.7868 | 0.9390 |
| positive_margin | emg | 0.0000 | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| positive_margin | emg | 0.2500 | roc_auc | 0.8742 | 0.7896 | 0.9418 |
| positive_margin | emg | 0.2500 | balanced_accuracy | 0.7286 | 0.6199 | 0.8317 |
| positive_margin | emg | 0.5000 | roc_auc | 0.8476 | 0.7560 | 0.9247 |
| positive_margin | emg | 0.5000 | balanced_accuracy | 0.7381 | 0.6318 | 0.8356 |
| positive_margin | emg | 1.0000 | roc_auc | 0.8485 | 0.7553 | 0.9254 |
| positive_margin | emg | 1.0000 | balanced_accuracy | 0.7518 | 0.6518 | 0.8436 |
| positive_margin | emg | 2.0000 | roc_auc | 0.8548 | 0.7651 | 0.9299 |
| positive_margin | emg | 2.0000 | balanced_accuracy | 0.7437 | 0.6438 | 0.8380 |
| positive_margin | table | 0.0000 | roc_auc | 0.8710 | 0.7868 | 0.9390 |
| positive_margin | table | 0.0000 | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| positive_margin | table | 0.2500 | roc_auc | 0.8677 | 0.7852 | 0.9343 |
| positive_margin | table | 0.2500 | balanced_accuracy | 0.7017 | 0.5999 | 0.8040 |
| positive_margin | table | 0.5000 | roc_auc | 0.8593 | 0.7779 | 0.9257 |
| positive_margin | table | 0.5000 | balanced_accuracy | 0.6999 | 0.6048 | 0.7976 |
| positive_margin | table | 1.0000 | roc_auc | 0.8230 | 0.7419 | 0.8932 |
| positive_margin | table | 1.0000 | balanced_accuracy | 0.6518 | 0.5688 | 0.7369 |
| positive_margin | table | 2.0000 | roc_auc | 0.7541 | 0.6674 | 0.8343 |
| positive_margin | table | 2.0000 | balanced_accuracy | 0.6226 | 0.5512 | 0.6996 |
| positive_margin | us | 0.0000 | roc_auc | 0.8710 | 0.7868 | 0.9390 |
| positive_margin | us | 0.0000 | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| positive_margin | us | 0.2500 | roc_auc | 0.8801 | 0.7964 | 0.9481 |
| positive_margin | us | 0.2500 | balanced_accuracy | 0.7481 | 0.6399 | 0.8512 |
| positive_margin | us | 0.5000 | roc_auc | 0.8801 | 0.7959 | 0.9481 |
| positive_margin | us | 0.5000 | balanced_accuracy | 0.7752 | 0.6665 | 0.8727 |
| positive_margin | us | 1.0000 | roc_auc | 0.4897 | 0.3724 | 0.6071 |
| positive_margin | us | 1.0000 | balanced_accuracy | 0.5000 | 0.5000 | 0.5000 |
| positive_margin | us | 2.0000 | roc_auc | 0.5238 | 0.4071 | 0.6406 |
| positive_margin | us | 2.0000 | balanced_accuracy | 0.5000 | 0.5000 | 0.5000 |

paired_changes_ci.csv 中 change_minus_equal_late_change 表示 QMF 的性能变化减去等权融合的变化；对 AUC、准确率等指标，正值表示 QMF 下降较少。对 Brier 和对数损失则负值更好。区间包含零表示差异仍不明确。

quality_changes_ci.csv 检查分支损失是否增大、q 是否下降。fold_quality.csv 给出各独立模型的相关性、协方差和 q 均值；均值与 1/3 的差只是描述性检查，不能据此认定满足总体期望条件。
