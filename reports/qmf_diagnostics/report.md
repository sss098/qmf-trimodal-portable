# QMF 质量与统计补充分析

使用 formal_v1 已保存的患者折外预测。三种方法共用 85 名患者；每种方法的最终概率取三个种子的平均。

## 1. q 与损失的关系

Pearson 衡量线性关系，Spearman 衡量排序关系，预期均为负。下表先给出每个种子汇总五折的结果：

| seed | modality | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| 42 | emg | 85 | 0.0073 | -0.3745 |
| 42 | table | 85 | -0.2325 | -0.8568 |
| 42 | us | 85 | 0.0548 | -0.4614 |
| 43 | emg | 85 | 0.1044 | -0.2546 |
| 43 | table | 85 | -0.0827 | -0.7813 |
| 43 | us | 85 | 0.0511 | -0.4664 |
| 44 | emg | 85 | 0.2569 | -0.2879 |
| 44 | table | 85 | -0.1098 | -0.8333 |
| 44 | us | 85 | -0.1172 | -0.5985 |

每个种子有 5 个分别训练的模型。以下统计 15 个模型各自测试折内的相关性，避免混合模型尺度；这 15 个系数用于描述稳定性，同一患者的三个种子不作为独立样本。

| modality | folds | negative_pearson | negative_spearman | median_pearson | median_spearman |
| --- | --- | --- | --- | --- | --- |
| emg | 15 | 8 | 13 | -0.0767 | -0.4142 |
| table | 15 | 13 | 15 | -0.2820 | -0.8554 |
| us | 15 | 11 | 15 | -0.0960 | -0.5637 |

按正负类拆分的结果见 correlations.csv（label=0/1）；fold=-1、label=-1 分别表示汇总五折、合并类别。每折仅 17 人，重点看方向和稳定性。

## 2. 分类分数尺度

offset 为两个类别分数的均值，abs_gap 为两类分数差的绝对值。二分类质量可写成 q = [offset + log(2 cosh(gap/2))]/10，因此 q 同时包含整体偏移和类别分离程度。

下面列出每个种子汇总五折时，q 与整体偏移的相关性：

| seed | modality | pearson | spearman |
| --- | --- | --- | --- |
| 42 | emg | 0.9400 | 0.9629 |
| 42 | table | 0.6003 | 0.5412 |
| 42 | us | 0.9395 | 0.8985 |
| 43 | emg | 0.9687 | 0.9853 |
| 43 | table | 0.5217 | 0.4870 |
| 43 | us | 0.8902 | 0.8787 |
| 44 | emg | 0.9695 | 0.9858 |
| 44 | table | 0.6985 | 0.6463 |
| 44 | us | 0.6946 | 0.6457 |

scale_distribution.csv 按种子、模态、折及预测正确/错误输出分数与 q 的均值、标准差和分位数；correlations.csv 同时包含 q 与 abs_gap 的关系。

### 平移敏感性

给指定模态的两个分数同时加常数，单分支概率与交叉熵严格不变，q 增加常数的十分之一。下表显示三种子集成后的融合预测变化。此检查衡量结构上的尺度敏感性，不用测试集选择新偏移参数。

| modality | shift | changed_predictions | mean_abs_probability_change | roc_auc | accuracy |
| --- | --- | --- | --- | --- | --- |
| us | -5.0000 | 17 | 0.1196 | 0.8885 | 0.7765 |
| us | -1.0000 | 3 | 0.0209 | 0.8738 | 0.8235 |
| us | 1.0000 | 0 | 0.0193 | 0.8668 | 0.8353 |
| us | 5.0000 | 4 | 0.0807 | 0.8534 | 0.8353 |
| emg | -5.0000 | 7 | 0.0507 | 0.8773 | 0.8000 |
| emg | -1.0000 | 1 | 0.0092 | 0.8717 | 0.8471 |
| emg | 1.0000 | 0 | 0.0087 | 0.8668 | 0.8353 |
| emg | 5.0000 | 0 | 0.0397 | 0.8548 | 0.8353 |
| table | -5.0000 | 33 | 0.3080 | 0.3058 | 0.5647 |
| table | -1.0000 | 0 | 0.0360 | 0.8597 | 0.8353 |
| table | 1.0000 | 2 | 0.0272 | 0.8738 | 0.8353 |
| table | 5.0000 | 5 | 0.0854 | 0.8850 | 0.8235 |

## 3. 患者级 95% 置信区间

按正负类分层、有放回抽取患者 10,000 次，取 2.5% 和 97.5% 分位数。各方法使用完全相同的抽样索引；三个种子先集成，独立单位是患者。区间针对固定折外预测，不包含重新训练的不确定性。

| method | metric | estimate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| equal_late | roc_auc | 0.8541 | 0.7518 | 0.9376 |
| equal_late | accuracy | 0.8235 | 0.7529 | 0.8941 |
| equal_late | sensitivity | 0.9355 | 0.8710 | 0.9839 |
| equal_late | specificity | 0.5217 | 0.3043 | 0.7391 |
| equal_late | balanced_accuracy | 0.7286 | 0.6199 | 0.8318 |
| equal_late | f1 | 0.8855 | 0.8372 | 0.9302 |
| equal_late | brier | 0.1287 | 0.0980 | 0.1618 |
| equal_late | log_loss | 0.4083 | 0.3251 | 0.5024 |
| qmf | roc_auc | 0.8703 | 0.7805 | 0.9432 |
| qmf | accuracy | 0.8353 | 0.7647 | 0.9059 |
| qmf | sensitivity | 0.9355 | 0.8710 | 0.9839 |
| qmf | specificity | 0.5652 | 0.3478 | 0.7826 |
| qmf | balanced_accuracy | 0.7504 | 0.6417 | 0.8534 |
| qmf | f1 | 0.8923 | 0.8438 | 0.9375 |
| qmf | brier | 0.1218 | 0.0856 | 0.1619 |
| qmf | log_loss | 0.3766 | 0.2771 | 0.4887 |
| tmc | roc_auc | 0.9004 | 0.8282 | 0.9593 |
| tmc | accuracy | 0.8118 | 0.7529 | 0.8706 |
| tmc | sensitivity | 0.9677 | 0.9194 | 1.0000 |
| tmc | specificity | 0.3913 | 0.2174 | 0.6087 |
| tmc | balanced_accuracy | 0.6795 | 0.5789 | 0.7826 |
| tmc | f1 | 0.8824 | 0.8451 | 0.9185 |
| tmc | brier | 0.1219 | 0.0939 | 0.1515 |
| tmc | log_loss | 0.3827 | 0.3150 | 0.4548 |

### 配对差值

| comparison | metric | estimate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| qmf minus equal_late | roc_auc | 0.0161 | -0.0196 | 0.0554 |
| qmf minus equal_late | accuracy | 0.0118 | 0.0000 | 0.0353 |
| qmf minus equal_late | sensitivity | 0.0000 | 0.0000 | 0.0000 |
| qmf minus equal_late | specificity | 0.0435 | 0.0000 | 0.1304 |
| qmf minus equal_late | balanced_accuracy | 0.0217 | 0.0000 | 0.0652 |
| qmf minus equal_late | f1 | 0.0068 | 0.0000 | 0.0211 |
| qmf minus equal_late | brier | -0.0068 | -0.0180 | 0.0053 |
| qmf minus equal_late | log_loss | -0.0316 | -0.0620 | 0.0018 |
| tmc minus qmf | roc_auc | 0.0302 | -0.0021 | 0.0708 |
| tmc minus qmf | accuracy | -0.0235 | -0.0824 | 0.0235 |
| tmc minus qmf | sensitivity | 0.0323 | 0.0000 | 0.0806 |
| tmc minus qmf | specificity | -0.1739 | -0.3478 | -0.0435 |
| tmc minus qmf | balanced_accuracy | -0.0708 | -0.1578 | 0.0025 |
| tmc minus qmf | f1 | -0.0100 | -0.0424 | 0.0253 |
| tmc minus qmf | brier | 0.0001 | -0.0154 | 0.0147 |
| tmc minus qmf | log_loss | 0.0061 | -0.0486 | 0.0518 |
| tmc minus equal_late | roc_auc | 0.0463 | 0.0056 | 0.0982 |
| tmc minus equal_late | accuracy | -0.0118 | -0.0706 | 0.0471 |
| tmc minus equal_late | sensitivity | 0.0323 | 0.0000 | 0.0806 |
| tmc minus equal_late | specificity | -0.1304 | -0.3043 | 0.0435 |
| tmc minus equal_late | balanced_accuracy | -0.0491 | -0.1441 | 0.0459 |
| tmc minus equal_late | f1 | -0.0031 | -0.0388 | 0.0344 |
| tmc minus equal_late | brier | -0.0067 | -0.0185 | 0.0045 |
| tmc minus equal_late | log_loss | -0.0255 | -0.0664 | 0.0097 |

AUC、准确率、敏感度、特异度、平衡准确率和 F1 越高越好；Brier 分数与对数损失越低越好。这里报告探索性区间，不作多重比较显著性宣称。

## 4. 尚需训练的消融

当前结果可以检查 q 与损失的关系及尺度敏感性。排序项本身的贡献需要比较无排序、原排序、当前正间隔排序；训练命令见 docs/qmf_ablations.md。
