# QMF 三模态分类：独立迁移目录

本目录包含运行所需的数据、模型代码、ImageNet 权重、正式实验最终模型和分析结果。训练与评估不依赖原来的 `/home/wp24/mix` 数据目录。主线为 `noise_aug_v1` 增强版 QMF。

> GitHub 提交版本为代码与结果归档：为保护患者级原始数据并控制仓库体积，`data/`、`weights/`、`*.pt`、`*.pth` 以及 `artifacts/prepared/` 不纳入版本库；实验指标、预测、配置、报告和文档会保留。

2026-09-24 完成的补充实验结果见 `artifacts/runs/supplement_*/`，汇总分析见 `reports/supplement_comparison/` 与 `reports/no_calf_followups/`。传统模型的 `*.joblib` 与逐患者临床测量表同样不纳入公开版本库。

## 换机器后如何运行

推荐 Linux 或 WSL2、Python 3.11。训练与完整扰动评估使用可用的 NVIDIA GPU；代码和数据可离线复制，首次安装 Python 依赖需要联网或准备离线安装包。

```bash
cd /你的路径/qmf_trimodal_portable
PYTHON_BIN=python3.11 bash scripts/setup.sh
bash scripts/check.sh --models --require-cuda
```

`setup.sh` 创建本地 `.venv`，按 `requirements-tested.txt` 安装当前验证过的版本。GPU 驱动必须与 PyTorch 配套；硬件驱动不包含在目录中。已有兼容环境也可跳过安装，所有脚本均支持 `PYTHON_BIN=/环境路径/bin/python`。选取顺序为 `PYTHON_BIN`、本目录 `.venv/bin/python`、系统 `python3`。

### 查看已有结果

详细方法与结果见 [增强版完整汇报](docs/QMF增强版三模态融合完整汇报.md)。主线模型和原始指标在 `artifacts/runs/noise_aug_v1/`，正式扰动结果在 `reports/qmf_robustness_noise_aug_v1/`。

### 重新训练主线方案

```bash
bash scripts/train_noise_aug_v1.sh
```

训练输出到 `artifacts/runs/portable_noise_aug_v1/`，不会覆盖已带入的 `noise_aug_v1`。两种融合方式、三个种子、五折共30个最终模型；中断后重跑同一命令续训。跨机器、跨依赖版本或代码发生变化时，严格续训检查可能拒绝恢复，此时使用新的运行名称重新训练。

```bash
bash scripts/train_noise_aug_v1.sh --name portable_noise_aug_v2
```

实验名称限英文字母、数字、下划线和连字符。

### 评估已带入的主线模型

已完成的报告可直接阅读。若要在新机器实际重跑推理，指定新的输出目录，避免仅复用已保存的预测：

```bash
bash scripts/evaluate_qmf_robustness.sh --run-name noise_aug_v1 \
  --output reports/noise_aug_recheck
```

评估本机新训练的模型：

```bash
bash scripts/evaluate_qmf_robustness.sh --run-name portable_noise_aug_v1
```

CPU 可用 `--device cpu` 做评估，但完整计算耗时较长。快速流程检查可加 `--smoke --seeds 42 --folds 0 --levels 0 0.5 --noise-repeats 1 --bootstrap 50`；初始迁移包中已清理旧 smoke 结果。

### 复现实验分支

- `bash scripts/train_base.sh`：重新训练原基础配置的等权、QMF、TMC，写入 `portable_base_v1`。
- `bash scripts/train_qmf_ablations.sh`：两组排序消融，写入 `portable_rank0_v1`、`portable_rank_original_v1`。
- `bash scripts/prepare.sh`：根据本目录原始输入重新构建缓存，校验患者划分。

## 目录内容

| 路径 | 内容 |
| --- | --- |
| `data/source/` | 实际使用的超声图像、血液表及肌电审计清单 |
| `data/raw/` | 实际使用的原始肌电信号及标记，保留所需目录结构 |
| `data/clinical.xlsx` | 原分组与临床指标表 |
| `weights/` | ResNet50 ImageNet 初始化权重 |
| `src/trimodal_joint/` | 模型、数据处理、训练、融合与增强代码 |
| `configs/`、`scripts/` | 相对路径配置和统一运行入口 |
| `artifacts/prepared/` | 两个历史缓存的原始窗口、表格与可迁移清单，不包含可重建的特征缓存 |
| `artifacts/runs/` | 105个正式最终模型、逐折预测、训练历史及参数 |
| `reports/` | 常规结果、相关性、尺度、置信区间、两版扰动分析 |
| `docs/` | 新完整汇报、历史汇报及运行说明 |
| `paper_draft/` | 论文正文、补充材料、图件及结果来源表 |
| `references/` | QMF 原论文及新增参考文献 |
| `三模态肌少症分类_内容补全版_v2.pptx` | 三模态分类汇报演示文稿 |
| `third_party/` | 原仓库参考代码及来源说明 |
| `archives/` | 原正式实验代码快照和完整性记录，供追溯 |
| `migration_models.json` | 模型迁移前后校验值；仅元数据路径调整，权重张量不变 |

## 迁移与校验说明

数据文件是实际副本，不是指向旧目录的软链接。105个最终模型保留原参数，配置路径改为本目录相对路径；历史缓存指纹保留原值以便查找，清单中的输入路径已重定位。新训练根据相对路径生成新的缓存和代码签名。

为了控制体积，未复制中间 `last.pt` 优化器检查点，也不复制旧环境和自动生成的特征缓存。因此带入模型可推理和复核，但不用于接着旧训练过程继续优化。新实验在本目录运行后会正常生成支持续跑的检查点。

可用 `bash scripts/check.sh --models` 检查663个输入资产、105个最终模型，以及患者、标签、五折分配和指标字段。历史报告和元数据中的来源信息仅作追溯，不参与训练。`archives` 中的旧代码快照保持原样，不作为当前入口。
