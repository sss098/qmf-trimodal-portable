# 2026-09-16 无损整理记录

## 保留的正式实验

`artifacts/runs/formal_v1`的45组实验保留不变：45份最终model.pt、45份selection/last.pt、45份refit/last.pt及全部划分、预处理、配置、历史、预测、指标。正式文件共754个，清理前后文件集合、大小、修改时间一致；JSON/CSV另核对SHA256一致。保留正式数据缓存e56e5bf4...及全部正式特征缓存。

## 改了什么

- raw.resolve_signal移除未使用的recorded参数，调用及测试同步更新；仍在同患者采集子目录要求唯一CSV，行为不变。
- emg_features移除无人调用的批量包装、永久为True的依赖开关及不可达分支；保留标量参考实现供数值对照。
- emg_model移除重复赋值。网络结构、参数名、参数形状和数值行为不变。
- 分析脚本统一格式，增加main入口，项目路径相对脚本定位。重新运行后JSON统计结果与归档逐项相同。
- 修正文档过时状态，补充QMF排序margin与原仓库不同的事实；不修改该损失以保持formal_v1定义。

fusion.py、losses.py、models.py、training.py、生产features.py、数据处理配置均未改变。没有删144维输入中的重复频谱指标，没有取消特征映射分支。

## 删除内容

- artifacts/runs/verification01_smoke
- artifacts/runs/verification02_smoke
- artifacts/prepared/a145812450270338f073d6fc91d9777ba05a1d138fbc1e35662037d74c2fa757
- 可重新生成的Python/pytest/Ruff临时缓存。

上述3个实验产物目录合计释放约3.883 GiB，清单见cleanup_deleted.json。删除前已核对旧缓存不被任何formal_v1配置引用；短测JSON/CSV等轻量证据归档为archives/old_smoke_metadata.tar.gz，未保留其大检查点和数组。没有删除正式续训检查点、最终模型、当前数据缓存、原始采集数据、原QMF/TMC仓库或MATLAB资料。

## 原始代码快照与旧检查点

原始完整代码及测试、配置、文档、分析脚本和第三方参考位于：

`archives/formal_v1_code_f25138d975a7.tar.gz`

SHA256见archives/snapshot.json；对应formal_v1原代码指纹：

`f25138d975a7eec00bd9f8398d296d493f29546ea50d47a360b3e176d2ad9f12`

**清理后的源码即使数值等价，也会改变严格的全包指纹。没有伪造旧指纹或放宽resume校验。** 当前分析脚本可以直接分析formal_v1，但不能用当前源码对原实验执行resume。需要旧源码时可解压到独立目录（不覆盖当前代码）：

```bash
cd /home/wp24/mix/qmf_trimodal_joint_v1_4
mkdir -p ../qmf_formal_v1_snapshot
tar -xzf archives/formal_v1_code_f25138d975a7.tar.gz -C ../qmf_formal_v1_snapshot
PYTHONPATH=/home/wp24/mix/qmf_formal_v1_snapshot/src \
  /home/wp24/miniconda3/envs/mix/bin/python -m trimodal_joint.cli train \
  --config /home/wp24/mix/qmf_formal_v1_snapshot/configs/base.yaml \
  --methods equal_late,qmf,tmc --seeds 42,43,44 --folds 0,1,2,3,4 \
  --name formal_v1 --resume
```

该命令仍要求原数据、配置、权重及环境契约一致；formal_v1已经全部完成，通常无需运行它。归档配置的output仍指向原正式产物目录。新增实验使用当前源码和新名字，例如formal_clean_v1。

## 验证

- 全项目自有Python代码Ruff检查通过，src/tests/reports格式检查通过。
- 23项测试通过，包括归档/当前EMG前向和所有参数梯度逐位相同、标量参考特征逐位相同。
- 21项已有回归测试继续通过。
- 分析脚本输出与旧版JSON统计完全一致。
- 独立代码审查确认清理局限于上述范围，无网络/损失/训练行为变化。
- 旧代码和清理后代码指纹见cleanup_verification.json。没有启动新训练。
