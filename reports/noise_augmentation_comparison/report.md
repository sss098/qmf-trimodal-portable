# 噪声增强配对比较

Matched local preprocessing; conditional patient-paired comparison.

difference =（Enhanced 扰动后 − 干净）−（no-noise 扰动后 − 干净）。AUC、准确率等指标为正表示 Enhanced 下降较少；log loss、Brier 为负表示更好。先对三个模型种子集成，每次噪声重复分别计算指标，再平均三次重复；区间按85名患者配对重采样。
