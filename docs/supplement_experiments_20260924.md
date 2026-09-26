# 补充实验判断与执行（2026-09-24）

## 一条命令运行全部新增训练

```bash
cd /home/wp24/mix/qmf_trimodal_joint_v1_4 && bash scripts/train_supplement_all.sh all
```

当前会话只做标签核查、冻结模型推理、代码验证和命令预检；未启动真实队列的新模型训练。
该命令串行执行 **90 个神经网络最终折模型 + 30 个传统表格最终折模型，共 120 个**。
每组为种子 42/43/44 × 五折。选轮和调参产生的中间拟合不计入最终模型数。
最后自动进行新版 no-noise 扰动推理及患者配对比较。中断后重跑同一命令即可；配置、代码或数据改变时会拒绝续训，避免混合结果。

只检查、不训练：`bash scripts/train_supplement_all.sh --dry-run`。
默认解释器 `/home/wp24/miniconda3/envs/mix/bin/python`，可由 `PYTHON_BIN` 覆盖。
本机已安装 XGBoost 3.0.5；重建环境用 `requirements-supplement.txt`。

## 逐项判断

| 项目 | 判断及实现 | 新增最终模型 |
|---|---|---:|
| DXA + 握力对账 | 必须审计，保留用户指定目录标签。已核对85人，并导出条件性AWGS结果；缺原始DXA报告，不能声称已验证原始DXA诊断 | 0 |
| Enhanced QMF，rank_weight=0 | 必须。旧rank0为patience=12且无额外噪声，不能与Enhanced直接比较 | 15 |
| no-noise，相同robustness | 必须。已为旧HPC模型补推理；另发现HPC肌电缓存不一致，严格噪声消融需在冻结的本地主实验缓存上重训 | 15 |
| Enhanced TMC | 强烈建议，已接入与Enhanced QMF相同的输入、划分、优化与选轮预算；保留TMC自身融合和损失 | 15 |
| Clinical-only / blood-only / combined | 强烈建议。新增clinical和blood，复用已完成且协议核对一致的combined表格MLP | 30 |
| 去掉两项小腿围 | 强烈建议。新增combined表格MLP的no_calf版本；检验表格基线对小腿围的依赖 | 15 |
| Elastic-net logistic + XGBoost | 建议。各使用combined候选特征、相同外层患者划分与开发集refit | 30 |
| Missing modality | 已做，不训练；基于独立编码器的OOF logits逐一移除分支，按剩余分支重新融合 | 0 |
| Clinically realistic corruption | 暂缓。缺少设备误差/脱落机制依据，不把任意高斯噪声或二元临床字段加噪称为临床真实扰动 | 0 |

去小腿围实验针对独立表格模型，不等于证明完整三模态QMF完全不依赖小腿围。若论文要作后一个声明，需要另训三模态去小腿围对照；本轮不增加该额外声明或训练。

## 标签与AWGS核查

用户明确指定 `/home/wp24/mix/更新数据/肌少症` 中的标签为准，不改标签。
重新读取B超和肌电目录名，按患者编号与原始Excel ID映射核对：**85/85一致，62阳性、23阴性**。
预检发现任何标签冲突会停止，不会静默改标。原五折划分不变。

当前Excel的分组表头为“低肌肉量组/非低肌肉量组”；目录名为“肌少症/非肌少症”。
按已记录的肌量指数应用DXA界值（男<7.0、女<5.4 kg/m²）得到62名低肌量者；
按握力界值（男<28、女<18 kg）得到63名低握力者；53人同时满足两项。
其余9名低肌量者握力正常，但缺AWGS认可的步速/SPPB/五次坐起数据，完整AWGS状态未定，不能直接判阴性。
有条件可判定的76人与目录标签一致。

本地找到的是Excel已录肌量指数，尚无可核验的原始DXA四肢瘦体重报告和测量方法证明。
因此这次完成的是字段对账与条件性规则重算，**不是原始DXA测量复算**。
握力记录也需原测量方案证明；“能否独立行走”不能替代AWGS体能指标。
规则来源：[AWGS 2019原始共识](https://pubmed.ncbi.nlm.nih.gov/32033882/)。

报告：`reports/label_audit_20260924/report.md`；逐例结果：同目录`patients.csv`。

## no-noise为何额外增加重训

旧模型在HPC训练，本地Enhanced主模型在本机训练。相同源文件、相同代码和依赖版本，并未生成完全相同的肌电张量。

- HPC与本地85名患者的ID/标签/折划分相同；表格文件SHA256完全相同。
- s031滤波后记录含211110个时间点，其中134057个唯一值，36541个时间值重复，最多重复12次，且顺序并非单调。
- 原实现用默认`np.argsort`再取重复时间戳的首条记录；默认非稳定排序在两台机器上保留了不同的重复采样点。
- 直接比较两机所选原始行索引：quicksort摘要不同，而stable排序摘要一致；原时间数组摘要相同。
- s031窗口最大绝对差为0.470433235；其余患者最大差不超过1.2e-7。
- 用本地缓存复现旧no-noise种子43/fold0时，s031肌电logits偏差达0.0082455，触发原有容差检查。

本轮没有修改原始预处理规则、重建主模型或替换旧缓存。新训练入口用`configs/supplement_data_lock.json`锁定主实验缓存的文件校验值，再执行原预处理来源校验。
所有新增模型复用该冻结缓存，确保输入与主实验相同。换机器时应携带原冻结缓存，不能自行重算后当作同一实验。
若以后统一改成稳定去重，应新建数据版本并重新训练所有相关主模型与对照，不能只修改一个对照。

旧HPC模型推理使用从HPC取回的原始缓存；表格与影像逐文件校验后仅映射本地路径。
证据在 `artifacts/hpc2_20260923/prepared/077bf22c53439c3abef2090859108df864eb1d3861fb556ae1b69550ad79b9b1/relocation.json`。
旧模型结果与Enhanced的比较仅作描述；最终噪声消融用新训练的`supplement_no_noise_v1`。

## 协议与防泄漏

神经网络沿用Enhanced协议：100轮选轮预算、patience=100、验证log loss选E；重新初始化，在68人开发集训练E轮；测试集17人。
AdamW、batch_size=6、学习率、衰减、类别权重、图像/肌电聚合均沿用原实现。
QMF rank0只去ranking项；TMC只切换方法；no-noise只关闭额外噪声，保留原图像增强。
表格三个新组在原95个候选变量上预先限制允许的列，再由训练集拟合缺失筛选、填补、标准化。
候选特征数为clinical 40、blood 55、combined 95、no_calf 93；各折实际保留列由训练子集决定。
保存原始列索引，因此推理与训练使用同一组列。clinical/blood/no_calf的额外表格噪声概率仍为1/6，与既有combined一致。

传统模型使用相同的51/17/17划分：51人拟合预处理和候选模型，以17人验证集log loss选超参数；随后在68人重新拟合预处理与模型，再评估17人测试集。
不在完整85人上拟合填补、缩放或选择超参数。两个传统模型不加合成噪声，属于标准表格基线，不宣称训练增强相同。

- Elastic-net：SAGA，class_weight=balanced；C∈{0.01,0.1,1}，l1_ratio∈{0.1,0.5,0.9}，最多10000次迭代；不收敛会报错。
- XGBoost：CPU hist、learning_rate=0.05、max_depth∈{1,2,3}、n_estimators∈{50,100}、reg_lambda∈{1,10}，其余参数固定；类别比值仅从拟合子集计算。
- 比较使用85人的OOF三种子平均概率；10000次患者配对bootstrap。种子、折、噪声重复均不作为独立患者。区间条件于已拟合模型，多组比较按探索性分析解释。

API依据：[scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。

## 输出

新增训练输出：`artifacts/runs/supplement_<组名>_v1/`；传统模型：`artifacts/runs/supplement_classical_v1/`。
Combined复用：`artifacts/hpc2_20260923/runs/necessary_table_v1/equal_late/`。

全部训练完成后自动生成：

- `reports/supplement_comparison/`：主模型、rank0、TMC、no-noise与表格各组的配对差值。
- `reports/qmf_robustness_supplement_no_noise_v1/`：新版no-noise完整扰动曲线。
- `reports/noise_augmentation_comparison/`：Enhanced与新版no-noise的配对性能下降差值。

本次无需训练的结果：`reports/label_audit_20260924/`、`reports/missing_modalities_v1/`、旧HPC no-noise推理报告。

新增代码会改变训练代码指纹，旧实验不能用更新后的源代码强行续训；已完成结果及原模型保留，新实验使用独立命名空间。


## 已完成的缺失模态结果

Enhanced QMF三种子OOF集成AUC为0.884993。移除表格分支后为0.568022，
相对变化-0.316971（患者配对95%区间：-0.460028至-0.175298）。
移除超声后AUC为0.887097、移除肌电后为0.889201；这两项变化的区间均包含零。
这支持进一步分析表格信息来源；仅能解释当前冻结网络的分支移除行为，不能替代独立重训的模态贡献实验。

## 验证

全部68项pytest测试通过；Ruff检查及格式检查通过；两条shell入口语法检查通过。
`train_supplement_all.sh --dry-run`已在真实数据上通过标签、参考缓存及协议预检。
测试中的模型拟合使用小型合成数据；未启动85人队列的新增训练。


旧HPC no-noise推理已完成全部15模型、85患者、5个噪声强度和3次重复。
结果：`reports/qmf_robustness_no_noise_hpc_v1/report.md`；
与Enhanced的描述性配对比较：`reports/legacy_noise_augmentation_comparison/report.md`。
可用`bash scripts/evaluate_legacy_no_noise.sh`复现；不会重新训练模型。
