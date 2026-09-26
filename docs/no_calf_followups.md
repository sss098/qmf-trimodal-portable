# 去小腿围的三组补充实验

一条命令串行运行全部训练，并在结束后生成患者配对比较：

```bash
cd /home/wp24/mix/qmf_trimodal_joint_v1_4 && bash scripts/train_no_calf_followups.sh all
```

预检但不训练：`bash scripts/train_no_calf_followups.sh --dry-run`。中断后重跑原命令；每折检查点仅在代码、配置和数据指纹一致时续训。新实验共三组，每组3种子×5折，**45个最终模型**。

| 组别 | 输入 | 融合 | 输出目录 |
|---|---|---|---|
| Clinical-only no-calf | 临床表格38个候选变量 | 单分支恒等映射 | `artifacts/runs/supplement_clinical_no_calf_v1/equal_late/` |
| QMF no-calf | 超声、肌电、表格93个候选变量 | QMF | `artifacts/runs/supplement_trimodal_no_calf_v1/qmf/` |
| Equal no-calf | 同上 | 等权融合 | `artifacts/runs/supplement_trimodal_no_calf_v1/equal_late/` |

两项移除的变量均为`小腿围_患侧`、`小腿围_健侧`。Clinical-only同时排除所有`血液_`开头变量。各折先限制候选变量，再仅用训练患者拟合缺失筛选、填补及标准化；保存原始列索引供推理使用。三模态两组共用同一配置，但网络分别训练。

实验复用现有85人标签、同一冻结缓存、五折划分、种子42/43/44及Enhanced协议：100轮选轮预算、patience=100、验证log loss选E，重新初始化并在68人开发集训练E轮。未修改标签，也未重新生成主实验缓存。两个训练配置与Enhanced参考配置的差异仅限模态和表格特征组。

训练完成后自动输出`reports/no_calf_followups/metrics_ci.csv`和`paired_differences_ci.csv`，包含以下患者配对比较：

- Clinical-only有小腿围 vs Clinical-only无小腿围；
- 原三模态QMF vs QMF无小腿围；
- 原三模态等权融合 vs Equal无小腿围；
- QMF无小腿围 vs Equal无小腿围。

比较使用85人三种子平均OOF概率，10000次患者配对bootstrap；区间未校正多重比较，按探索性结果解释。原有combined表格去小腿围实验仍保留在`supplement_no_calf_v1`中，不与这三组混淆。

三组现已全部训练完成，共45个最终折模型。配对结果和解释见`reports/no_calf_followups/report.md`。此前已执行`--dry-run`、69项pytest测试、Ruff与shell语法检查。
