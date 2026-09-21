# QMF 排序消融：设置与运行

本次只改变质量排序项。三个分支、质量函数、融合公式、原始分组、五折名单及其他训练参数均沿用 formal_v1。

| 组别 | rank_mode | rank_weight | 结果目录 |
| --- | --- | --- | --- |
| 当前正间隔排序 | positive_margin | 0.1 | 已有 artifacts/runs/formal_v1/qmf |
| 去掉排序项 | positive_margin | 0 | artifacts/runs/qmf_rank0_v1/qmf |
| 原 QMF RGB-D 排序 | original | 0.1 | artifacts/runs/qmf_rank_original_v1/qmf |

零系数组保留 QMF 动态融合与四项分类监督，跳过排序及历史更新。original 组使用原仓库 MarginRankingLoss 的输入偏移和目标方向，保留零历史范围保护及单患者批次处理。默认 positive_margin 与原 formal_v1 数值行为一致。

## 一次运行两组

```bash
cd /你的路径/qmf_trimodal_portable
bash scripts/train_qmf_ablations.sh
```

脚本依次运行两组，不会同时抢占显卡；默认使用 `/home/wp24/miniconda3/envs/mix/bin/python`，并显式加载本项目源码。中断后执行同一命令，可恢复配置、数据和代码一致的实验。

## 分别运行

```bash
cd /你的路径/qmf_trimodal_portable
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 第一组：只去掉质量排序监督。
/home/wp24/miniconda3/envs/mix/bin/python -m trimodal_joint.cli train \
  --config configs/base.yaml --methods qmf \
  --seeds 42,43,44 --folds 0,1,2,3,4 \
  --rank-mode positive_margin --rank-weight 0 \
  --name qmf_rank0_v1 --resume

# 第二组：使用原仓库的排序损失。
/home/wp24/miniconda3/envs/mix/bin/python -m trimodal_joint.cli train \
  --config configs/base.yaml --methods qmf \
  --seeds 42,43,44 --folds 0,1,2,3,4 \
  --rank-mode original --rank-weight 0.1 \
  --name qmf_rank_original_v1 --resume
```

每组 3 个种子 × 5 折，共 15 个最终模型；两组合计 30 个。每折仍先用 51 人训练、17 人验证确定轮数，再重新初始化并用 68 人重训，最后测试 17 人。最大 100 轮、早停耐心 12，批次 6；超声主干学习率 10⁻⁵，其余参数 2×10⁻⁴，AdamW 权重衰减 10⁻³。超声主干使用 ImageNet 初始化，肌电、表格与新分类头随机初始化，三路全部参数联合更新。

模型与结果沿用现有文件格式：fold*/model.pt、selection/ 和 refit/、各 seed 的 oof_predictions.csv，以及方法目录下的 ensemble_predictions.csv 和 summary.json。

## 现有结果的补充分析

```bash
/home/wp24/miniconda3/envs/mix/bin/python reports/qmf_diagnostics.py
```

输出集中在 `reports/qmf_diagnostics/`：

- `report.md`：可直接阅读的分析报告。
- `patient_quality.csv`：每患者、每种子、每分支的分数、质量及交叉熵。
- `correlations.csv`：按种子、折和类别划分的 Pearson/Spearman；fold=-1 表示合并五折，label=-1 表示合并类别。
- `scale_distribution.csv`：分数、分数均值、类别差距、质量及损失的分位数，含正确/错误预测分组。
- `shift_sensitivity.csv`：共同平移两类分数后，单分支概率不变而融合预测发生的变化。
- `metrics_ci.csv`、`paired_differences_ci.csv`：10,000 次患者级分层配对 Bootstrap 的指标与差值 95% CI。
- `metadata.json`：数据指纹、随机种子与统计口径。

置信区间以固定折外预测为基础，未重复训练。所有方法使用相同患者抽样；同一患者的多个种子先集成，不作为独立样本。

已有 formal_v1 结果可直接作为正间隔基线。新增排序配置改变了源码指纹，请使用上述新实验名称；原实验的续训仍需使用 archives 中对应的旧代码快照。本次没有启动正式训练。
