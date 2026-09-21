# QMF 模态质量下降实验

固定已训练模型；每次只扰动一种输入，保留完整患者级聚合。

每个噪声重复先集成模型种子，再计算指标，最后平均重复指标。置信区间对患者配对重采样，不把种子或噪声重复当成独立患者。

| setting | modality | level | metric | estimate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| equal_late | emg | 0.0000 | roc_auc | 0.8836 | 0.7966 | 0.9537 |
| equal_late | emg | 0.0000 | balanced_accuracy | 0.7584 | 0.6529 | 0.8615 |
| equal_late | emg | 0.2500 | roc_auc | 0.8904 | 0.8048 | 0.9579 |
| equal_late | emg | 0.2500 | balanced_accuracy | 0.7584 | 0.6529 | 0.8615 |
| equal_late | emg | 0.5000 | roc_auc | 0.8922 | 0.8072 | 0.9596 |
| equal_late | emg | 0.5000 | balanced_accuracy | 0.7584 | 0.6529 | 0.8615 |
| equal_late | emg | 1.0000 | roc_auc | 0.9032 | 0.8226 | 0.9654 |
| equal_late | emg | 1.0000 | balanced_accuracy | 0.7367 | 0.6280 | 0.8398 |
| equal_late | emg | 2.0000 | roc_auc | 0.9074 | 0.8354 | 0.9638 |
| equal_late | emg | 2.0000 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| equal_late | table | 0.0000 | roc_auc | 0.8836 | 0.7966 | 0.9537 |
| equal_late | table | 0.0000 | balanced_accuracy | 0.7584 | 0.6529 | 0.8615 |
| equal_late | table | 0.2500 | roc_auc | 0.8871 | 0.8048 | 0.9518 |
| equal_late | table | 0.2500 | balanced_accuracy | 0.7313 | 0.6288 | 0.8344 |
| equal_late | table | 0.5000 | roc_auc | 0.8742 | 0.7948 | 0.9385 |
| equal_late | table | 0.5000 | balanced_accuracy | 0.7241 | 0.6352 | 0.8153 |
| equal_late | table | 1.0000 | roc_auc | 0.8312 | 0.7527 | 0.8990 |
| equal_late | table | 1.0000 | balanced_accuracy | 0.7015 | 0.6210 | 0.7820 |
| equal_late | table | 2.0000 | roc_auc | 0.7464 | 0.6615 | 0.8247 |
| equal_late | table | 2.0000 | balanced_accuracy | 0.6199 | 0.5547 | 0.6922 |
| equal_late | us | 0.0000 | roc_auc | 0.8836 | 0.7966 | 0.9537 |
| equal_late | us | 0.0000 | balanced_accuracy | 0.7584 | 0.6529 | 0.8615 |
| equal_late | us | 0.2500 | roc_auc | 0.8843 | 0.8020 | 0.9511 |
| equal_late | us | 0.2500 | balanced_accuracy | 0.7775 | 0.6717 | 0.8806 |
| equal_late | us | 0.5000 | roc_auc | 0.8836 | 0.7973 | 0.9528 |
| equal_late | us | 0.5000 | balanced_accuracy | 0.7775 | 0.6717 | 0.8806 |
| equal_late | us | 1.0000 | roc_auc | 0.8866 | 0.7980 | 0.9556 |
| equal_late | us | 1.0000 | balanced_accuracy | 0.7562 | 0.6502 | 0.8595 |
| equal_late | us | 2.0000 | roc_auc | 0.8906 | 0.8046 | 0.9572 |
| equal_late | us | 2.0000 | balanced_accuracy | 0.7616 | 0.6553 | 0.8671 |
| positive_margin | emg | 0.0000 | roc_auc | 0.8850 | 0.7994 | 0.9523 |
| positive_margin | emg | 0.0000 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| positive_margin | emg | 0.2500 | roc_auc | 0.8855 | 0.8004 | 0.9528 |
| positive_margin | emg | 0.2500 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| positive_margin | emg | 0.5000 | roc_auc | 0.8869 | 0.8029 | 0.9533 |
| positive_margin | emg | 0.5000 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| positive_margin | emg | 1.0000 | roc_auc | 0.8920 | 0.8100 | 0.9558 |
| positive_margin | emg | 1.0000 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| positive_margin | emg | 2.0000 | roc_auc | 0.8978 | 0.8191 | 0.9600 |
| positive_margin | emg | 2.0000 | balanced_accuracy | 0.7938 | 0.6907 | 0.8945 |
| positive_margin | table | 0.0000 | roc_auc | 0.8850 | 0.7994 | 0.9523 |
| positive_margin | table | 0.0000 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| positive_margin | table | 0.2500 | roc_auc | 0.8857 | 0.8062 | 0.9479 |
| positive_margin | table | 0.2500 | balanced_accuracy | 0.7504 | 0.6497 | 0.8507 |
| positive_margin | table | 0.5000 | roc_auc | 0.8780 | 0.8020 | 0.9390 |
| positive_margin | table | 0.5000 | balanced_accuracy | 0.7278 | 0.6344 | 0.8193 |
| positive_margin | table | 1.0000 | roc_auc | 0.8427 | 0.7676 | 0.9072 |
| positive_margin | table | 1.0000 | balanced_accuracy | 0.6961 | 0.6237 | 0.7694 |
| positive_margin | table | 2.0000 | roc_auc | 0.7756 | 0.6968 | 0.8476 |
| positive_margin | table | 2.0000 | balanced_accuracy | 0.6443 | 0.5802 | 0.7133 |
| positive_margin | us | 0.0000 | roc_auc | 0.8850 | 0.7994 | 0.9523 |
| positive_margin | us | 0.0000 | balanced_accuracy | 0.7802 | 0.6771 | 0.8832 |
| positive_margin | us | 0.2500 | roc_auc | 0.8859 | 0.8029 | 0.9521 |
| positive_margin | us | 0.2500 | balanced_accuracy | 0.7775 | 0.6717 | 0.8806 |
| positive_margin | us | 0.5000 | roc_auc | 0.8827 | 0.7983 | 0.9509 |
| positive_margin | us | 0.5000 | balanced_accuracy | 0.7586 | 0.6502 | 0.8617 |
| positive_margin | us | 1.0000 | roc_auc | 0.8869 | 0.8027 | 0.9537 |
| positive_margin | us | 1.0000 | balanced_accuracy | 0.7616 | 0.6529 | 0.8647 |
| positive_margin | us | 2.0000 | roc_auc | 0.8871 | 0.8029 | 0.9537 |
| positive_margin | us | 2.0000 | balanced_accuracy | 0.7616 | 0.6529 | 0.8647 |

paired_changes_ci.csv 中 change_minus_equal_late_change 表示 QMF 的性能变化减去等权融合的变化；对 AUC、准确率等指标，正值表示 QMF 下降较少。对 Brier 和对数损失则负值更好。区间包含零表示差异仍不明确。

quality_changes_ci.csv 检查分支损失是否增大、q 是否下降。fold_quality.csv 给出各独立模型的相关性、协方差和 q 均值；均值与 1/3 的差只是描述性检查，不能据此认定满足总体期望条件。
