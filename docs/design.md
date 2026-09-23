# 三模态直接联合训练设计方案

日期：2026-09-15；状态更新2026-09-16：formal_v1全部45组正式实验已完成。本文件保留方案依据，当前版本与原始源码归档说明见README.md及cleanup.md。


## 设计依据

EMG按用户确认的“肌电分类程序”文件夹适配，以年龄二分类脚本的FTTransformerRaw为基准。六通道主体已在内存中实例化验证：572274个参数，原始波形[2,100,6]及手工特征[2,144]输出[2,2]。完整训练实现已完成；验证记录见verification.md。

## 1. 已确认的范围

- 三种方法：QMF 动态 logits 融合、等权晚期 logits 融合、TMC 证据融合。
- 三种方法分别训练独立模型，使用相同的三分支结构、患者、标签、划分、初始化规则和训练预算。
- 超声：ResNet50，加载本地 ImageNet1K V2 权重；全部层参与联合训练。
- EMG：“肌电分类程序”的FTTransformerRaw，适配六通道；4层、8头、128维、FFN256；随机初始化。
- 表格：v1.4原标签版本的95项候选变量，MLP随机初始化。
- 不做本任务的单分支预训练，不加载已训练的任务分支检查点。ImageNet初始化不属于本任务单分支预训练。
- 本文件夹现包含实现、测试和正式结果；原项目与原始数据未改写。

## 2. 数据与标签

源项目：`/home/wp24/mix/sarcopenia_original_us_trimodal_5fold - new`。

唯一标签来源：`/home/wp24/mix/肌少症患者数据统计修改版2 - 副本.xlsx` 中“分组”下的两列标记，按用户指定名称“肌少症”/“非肌少症”分别编码为1/0。当前磁盘文件B2/C2实际表头仍为“低肌肉量组”/“非低肌肉量组”，与用户提供表头名称存在差异；须记录原始表头和映射，不宣称源表已改名。使用这两列的原始分组标记，不新增标签计算规则。不按握力或DXA重算，不从文件夹名、旧清单或EMG缓存推导。

按 `scripts/v1_4/cohort_original.py` 实际读取确认：85名患者，62阳性、23阴性；95项候选特征。主任务命名为表格原始分组的肌少症/非肌少症分类；仅沿用表格分组，不另外推断或重建诊断标准。

患者关联：新版表格患者ID → 旧表格患者ID → 影像患者编号 → EMG patient_id。保存唯一对应关系；重复、冲突、缺模态报错，不按行序拼接。仅用匿名研究编号进入实验输出。

超声沿用 `data/ultrasound_roi15/images` 及患者清单，已确认每人2–14张图。患者是采样与评估单位，图像不能独立跨折。

EMG从`/home/wp24/mix/更新数据`下的原始采集记录重建六通道波形，沿用源项目v1.1已校正的患者关联、电极列映射及五段收缩协议。该目录原始CSV已经找到。已有985×6×8缓存只作为患者/来源核对，不能作为此网络输入，也不能用它反演波形。标签始终读取表格，不使用年龄脚本中的文件夹标签。

## 3. 预处理

### 超声

- 既有ROI转RGB，等比例缩放、黑边补齐到224×224。
- /255后按ImageNet mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]标准化。
- 每个训练epoch每位患者随机取1张图：水平翻转概率0.5，亮度系数[0.92,1.08]，对比度系数[0.90,1.10]。
- 验证/测试无增强，使用该患者全部ROI。
- 先对全部ROI的超声两类原始输出取均值，得到一个患者级超声输出，再与该患者EMG和表格融合一次。TMC在均值之后转换证据，避免把多张图重复当作独立证据。此规则三方法相同，是新实验明确规定的聚合规则；不同于旧v1.4逐图融合后均值的路径。

### EMG

网络及窗口特征处理依据：`/home/wp24/mix/肌电分类程序/2-FT-Transformer-v1.4-年龄二分类.py`。原程序16通道改6通道，1000Hz、100ms窗、30ms步长保留；原波形token与手工特征token相加结构保留。

采集数据适配：原始记录按既有电极映射提取六通道，使用源项目的三阶Butterworth 10–499Hz和50Hz/Q35双向滤波、五段收缩选取规则；通常为50–60、70–80、90–100、110–120、130–140秒。已有患者31的本患者滤波信号和第3–7次收缩标记特例保留，不二次滤波、不借用他人信号。这些滤波/收缩规则是本队列适配，原年龄脚本本身从准备好的CSV直接切窗。

每段10000采样点独立切窗：floor((10000-100)/30)+1=331窗，五段共1655窗。窗口不得跨收缩边界。五通道记录在固定映射中保留既有缺失槽位，以有效通道的逐采样中位数补值并记录质量标记；缺失槽位不能凭剩余列位置重新编号。无法解析或非有限原始输入显式报错，不自动造零患者。

按原程序ChannelZScore，仅用当前训练患者的窗口沿窗口和采样轴拟合六个通道的均值、标准差，应用同一状态到验证/测试。不使用此前特征缓存的患者Min-Max，不保留旧方案的8特征标准化。selection和refit各自拟合并保存状态。

在标准化后的波形上按原extract_features计算每通道24项：6时域（RMS/MAV/peak/ZC/WL/SSC）、9FFT（5频段能量+频谱质心/峰频/中值频率/平均功率频率）、5STFT频段能量、4小波能量。频段为0–50、50–100、100–200、200–400、400–500Hz；STFT窗64、步32、NFFT128；小波db4、level3。每窗总144维，按通道顺序排列；PyWavelets为必需依赖，不沿用缺库时静默补零的回退。

患者级适配：训练每位患者每次从每段随机无放回取8窗，共40窗；共享同一FTTransformer处理每窗，再平均40个两类logits，得到一个患者级EMG输出。每段等额采样，保留通道与采样点顺序，不新增可学习池化。验证/测试处理全部1655窗（可按256窗分块），先平均每段logits，再等权平均五段；分块只控制内存，不改变聚合。患者级输出与超声、表格融合一次，损失也按患者计算。多窗处理和聚合是从原窗口分类到本患者分类的必要适配。

不添加旧方案的噪声、窗口置零或通道置零增强；保留原网络Dropout(0.2)。手工特征在原始输入上计算，不参与梯度；其后的特征MLP和波形投影均参与训练。网络没有第二级时间窗Transformer，没有独立全程统计分支。

### 表格

40个临床编码变量+55个血液变量，白名单与列顺序沿用cohort_original.py。

临床40项：性别_男、年龄、发病天数、每日康复分钟、身高、体重、BMI、Morse、FOIS、NRS2002、ADL、糖尿病、高血压、心脏病、出血性卒中、吸烟、饮酒、卒中家族史、既往跌倒、疼痛、独立行走、胃管、独居、与配偶居住、与子女居住、集体居住、病灶_额叶、病灶_顶叶、病灶_颞叶、病灶_枕叶、病灶_基底节、病灶_丘脑、病灶_脑干、病灶_小脑、病灶_放射冠、病灶_内囊、病灶_脑室、左侧偏瘫、小腿围_患侧、小腿围_健侧。

血液55项（保留原列名）：CRP、白细胞计数、中性粒细胞、淋巴细胞绝对值、嗜酸性粒细胞、嗜碱性粒细胞、单核细胞绝对值、中性粒细胞百分数、淋巴细胞百分数、嗜酸性粒细胞百分数、嗜碱性粒细胞百分数、单核细胞百分数、红细胞计数、血红蛋白、红细胞比容、平均红细胞容积、平均红细胞血红蛋白量、平均红细胞血红蛋白浓度、红细胞体积分布宽度、血小板计数、平均血小板体积、血小板比积、血小板体积分布宽度、有核红细胞绝对数、有核红细胞百分数、肾小球滤过率估算值、间接胆红素、AST/AL比值、白/球蛋白比值、尿素氮肌酐比值、球蛋白、阴离子间隙、渗透压、钾、钠、氯、总钙、尿素、肌酐、总二氧化碳、葡萄糖、尿酸、总胆固醇、甘油三酯、高密度脂蛋白胆固醇、低密度脂蛋白胆固醇、总蛋白、白蛋白、总胆红素、直接胆红素、丙氨酸氨基转移酶、天门冬氨酸氨基转移酶、谷氨酸氨基转移酶、碱性磷酸酶、同型半胱氨酸。

白名单直接来源于cohort_original.py的candidate_names；实施时导出features.json并校验名称及顺序。

排除姓名、ID、标签、握力、DXA、随访评分、随访跌倒、费用、风湿、L/H和CK-MB/CK。

仅训练集删除缺失率>50%或有限取值不足2种的列，再用训练中位数填补、训练均值和标准差标准化。标准差<1e-6置1。不额外添加缺失指示变量，不做全队列筛选。每折实际维数D及列顺序随预处理状态保存。

## 4. 网络

### 超声

ResNet50卷积骨干 → 全局平均池化 → 2048维 → Dropout(0.35) → Linear(2048,2)。本地权重`/home/wp24/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth`存在，加载骨干、替换ImageNet分类层，新分类头随机初始化。全部卷积与BatchNorm仿射参数可训练；训练更新BN统计，推理使用保存统计。参数23,512,130，其中骨干23,508,032，头4,098。

### EMG

原实现参考`/home/wp24/mix/肌电分类程序/2-FT-Transformer-v1.4-年龄二分类.py::FTTransformerRaw`。

每窗原始波形[B,100,6]转[B,6,100]，共享Linear(100,128)得到六个波形token。每通道24个手工特征经过LayerNorm(24) → Linear(24,128) → GELU → Dropout(0.2) → Linear(128,128)，与对应波形token相加。六通道由固定位置embedding区分，参数共享不等于合并通道。

添加可学习CLS和[1,7,128]位置embedding，进入4层TransformerEncoder：d_model128、8头、FFN256、GELU、pre-norm、dropout0.2。取128维CLS，经LayerNorm(128) → Linear(128,64) → GELU → Dropout(0.2) → Linear(64,2)。CLS和位置参数trunc_normal(std0.02)，其余沿用原层默认初始化，无单分支任务权重。

所有患者窗口共享同一网络，先输出窗口两类logits，再按数据章节规则聚合为患者两类logits。不存在额外912维统计分支、192维拼接或第二级时序Transformer。六通道/100点窗口/144维特征/两类配置已实例化计数：572274个可训练参数。

### 表格

Linear(D,64) → LayerNorm(64) → GELU → Dropout(0.20) → Linear(64,32) → GELU → Linear(32,2)。参数64D+2,338；D=95时8,418。

三方法所有分支结构相同。D=95时合计24,092,822个可训练参数，约2409万；融合公式自身没有额外可学习参数。参数量不等于运行显存，每患者多个波形窗另有激活开销。

## 5. 三种融合与损失

令三分支输出z_m∈R²，标签y∈{0,1}。

统一类别权重w_c=N_train/(2n_c)，只按当前训练患者计算。CE用逐样本负对数概率乘w_y后除以批次sum(w_y)；TMC也使用相同样本加权归约。排名历史用不加类别权重的逐患者单分支CE，避免把少数类权重当作质量。

### 等权晚期融合

z_f=(z_us+z_emg+z_tab)/3，p=softmax(z_f)。L=CE_w(z_f,y)+Σ_m CE_w(z_m,y)。辅助损失各系数1.0，融合系数1.0；不是独立训练后再平均，也不是平均三个softmax概率。

### QMF

采用本地官方RGB-D实现的完整梯度路径：q_m=logsumexp(z_m)/10，z_f=Σ_m q_m z_m。无跨模态softmax、无归一化、无正值裁剪，质量分数不等于概率，可能为负。输出p=softmax(z_f)。固定缩放0.1并非对logits先除以10的温度缩放。

L=CE_w(z_f,y)+Σ_m CE_w(z_m,y)+0.1Σ_m L_rank,m。

每模态、每训练患者独立累计未加权历史CE，复用原RGB-D行为：当前batch先用旧累计历史排名，再更新当前batch的detached损失。无历史平均、无epoch冻结快照。训练患者每epoch无放回各访问一次，因此首epoch当前batch的患者都未访问，排名自然为0。历史min-max归一化的分母下限1e-12，batch1返回可微零。selection/refit重新建立训练集局部索引和历史；验证测试不更新。

对历史损失h_i<h_j要求q_i>q_j，s_ij=sign(h_j-h_i)，margin=|h_i-h_j|/(h_max-h_min)，每模态排名为mean(ReLU(margin-s_ij(q_i-q_j)))，三模态相加。此处最终实现明确采用原累计历史，替代初稿中的平均/epoch快照约定。

q不detach，所以融合损失经z及q两条路径回传：∂L_f/∂z_m=q_m g+(g·z_m)softmax(z_m)/10；再叠加单分支CE与排名梯度。原图文脚本的融合权重detach与RGB-D实现有区别，本方案固定后者，不把二者混用。

### TMC

e_m=softplus(z_m)，alpha_m=e_m+1；S_m=Σ_c alpha_mc，b_mc=e_mc/S_m，u_m=2/S_m。

两模态冲突K=Σ_(i≠j)b_ai b_bj；合并b_c=(b_ac b_bc+b_ac u_b+b_bc u_a)/(1-K)，u=u_a u_b/(1-K)，再令S=2/u、alpha=bS+1。固定顺序先US与EMG，再与表格。仅三个真实模态，无伪模态。

推理p_c=alpha_fc/Σ alpha_fc，不对alpha再softmax；不确定度u_f=2/Σ alpha_fc。

单项证据损失EDL(alpha,y)=Σ_c y_c[digamma(S)-digamma(alpha_c)]+lambda(t)KL(Dir(alpha_tilde)||Dir(1))，alpha_tilde=y+(1-y)alpha，lambda(t)=min(1,t/10)，t从1开始，每次refit重置。

L=EDL_w(alpha_f,y)+Σ_m EDL_w(alpha_m,y)。融合及三个分支系数均1.0；没有额外普通CE、QMF排名、focal或label smoothing。KL和DS计算保留梯度。核心运算float32，DS采用原公式消去分母后的代数等价形式e_ab=e_a+e_b+e_a*e_b/K，避免高冲突除法；非有限损失/梯度显式报错。测试证据恒等性、概率和为1以及融合排列近似等价。

## 6. 训练设置

以下为已实现的固定起始设置，不是已搜索出的最优参数。

| 项目 | 值 |
|---|---|
| 主划分 | 表格标签分层患者5折；复用v1.4原标签主划分，seed=20260911 |
| 训练种子 | 42、43、44；fold内seed=seed+fold，另独立保存数据随机状态 |
| 选择阶段 | 测试fold=k；验证fold=(k+1)%5；其余3折训练 |
| 最大epoch / patience | 100 / 12 |
| 最佳选择 | 患者级未加权验证log loss最低；严格下降，持平保留更早轮 |
| 重训 | 全新初始化，在4个非测试折训练选中的E轮，无测试早停 |
| batch | 6位患者，shuffle=True，无放回、drop_last=False |
| 显存不足 | 另立记录为microbatch=2、累积3；不得静默改变，BN和QMF批内排名不与batch6严格等价 |
| optimizer | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| ResNet骨干LR | 1e-5 |
| 超声新分类头LR | 2e-4 |
| EMG全部参数LR | 2e-4 |
| 表格全部参数LR | 2e-4 |
| weight_decay | 1e-3，全部可训练参数采用同一规则 |
| scheduler / warmup | 无；固定学习率 |
| 梯度裁剪 | 全部参数global L2 norm=5.0 |
| 精度 | float32，AMP关闭 |
| dataloader | workers=0，CUDA时pin_memory=True |
| 训练冻结 | 无；全部分支从第1轮更新 |
| QMF | scale=0.1，rank系数0.1；原累计历史，首轮自然为零 |
| TMC | KL最终系数1.0，10轮退火；代数等价DS计算 |
| 分类 | 阳性概率≥0.5预测1 |

每步：取同批患者三模态→三分支前向→融合→算总损失→zero_grad→一次backward→检查有限梯度→clip_grad_norm→一次optimizer.step。三个分支同时更新。QMF按原RGB-D顺序，在当前batch排名后更新累计历史。验证时eval/no_grad，无optimizer及历史更新。

固定随机种子、NumPy/Python/Torch/CUDA状态及确定性设置；记录Torch、torchvision、CUDA和设备版本，确定性不可满足时显式报错或记录明确变更，不能默默退化。

选择阶段约51训练/17验证/17测试；重训68训练/17测试。全流程3方法×3种子×5折=45个外层实验，每个selection+refit，共90次训练过程。每种子85条完整OOF预测，逐种子指标mean±sample SD和三种子概率平均指标分别报告。

## 7. 分类、报告与追踪

输出研究ID、fold、表格标签、阳性概率、预测类别、三分支患者级输出。QMF另存原始q，不伪称归一化概率；TMC存三模态及融合alpha、u。

评价ROC-AUC、PR-AUC、ACC、balanced ACC、sensitivity、specificity、F1、Brier、log loss、混淆矩阵。ROC-AUC为主区分指标，验证log loss只用于选择轮数。以相同患者的OOF预测对比三方法，不比较训练损失数值，不预设方法排名。bootstrap若做仅标记为固定OOF预测的患者配对不确定性，不代表完整重训不确定性。

每折保存：配置、输入文件SHA256及代码版本、患者划分、表格字段和预处理、EMG标准化状态、ImageNet权重SHA256、初始化状态指纹、最佳轮数、历史曲线、检查点、随机状态、optimizer、QMF历史或TMC轮数、OOF预测。恢复训练仅接受一致配置和数据指纹；跨fold、seed、method不能混用检查点。

## 8. 文件组织与代码要求

项目目录`/home/wp24/mix/qmf_trimodal_joint_v1_4`。

```text
README.md
pyproject.toml
requirements-tested.txt
configs/base.yaml
src/trimodal_joint/
  config.py
  io.py
  cohort.py
  raw.py
  prepare.py
  emg_features.py
  features.py
  tabular.py
  data.py
  emg_model.py
  models.py
  fusion.py
  losses.py
  evaluation.py
  training.py
  cli.py
tests/
docs/
third_party/
artifacts/
```

采用清晰独立适配层，QMF相关原代码与许可放third_party只读参照并记录来源。不要复制原仓库全部数据、模型与运行输出。不得使用临时sys.path注入、硬编码.cuda()、宽泛捕获后继续、通过assert承担生产数据验证；使用可安装包、Path、类型标注、显式异常、统一日志、严格配置校验、ruff和pytest。

训练器接收统一结构化输出；分支输出raw_scores，融合分别提供probabilities及方法专用状态，不能把TMC alpha伪装成普通logits。三方法共有数据和骨干配置；只有融合与损失专属配置不同。

## 9. 实施顺序与验收

1. 数据适配与审计：核对85人、62/23、95候选、所有文件存在；证明患者不跨折、标准化只使用训练患者；导出数据与特征清单。
2. 三分支：核对输入输出和参数数量；验证ImageNet只加载骨干；六通道位置参数、波形/特征对应及切窗边界单测；所有预期可训练模块都收到有限梯度。
3. 三融合及损失：手算案例验证等权、QMF、TMC；排名方向、历史范围零/批次1；QMF质量路径梯度；TMC无证据输入、冲突、数值稳定、KL和融合梯度。
4. 训练与恢复：验证selection/refit完全重新初始化，训练预处理重拟合；验证测试阶段不改模型和历史；同环境中断恢复可重现。
5. 每方法seed42/fold0短程冒烟，仅查可运行、有限损失和输出；标记smoke，不用于正式结果。
6. 配置锁定后运行45个外层正式实验，生成完整患者级OOF和比较报告。

## 10. 依据

- 本地v1.4：`scripts/v1_4/cohort_original.py`、`scripts/v1_4/train_original_labels.py`、`scripts/v1_3/train_direct_joint.py`。
- EMG：`/home/wp24/mix/肌电分类程序/2-FT-Transformer-v1.4-年龄二分类.py::FTTransformerRaw`、`ChannelZScore`和`extract_features`；采集适配参考源项目`scripts/v1_1/rebuild_emg_v1_1.py`。
- QMF本地原实现：`/home/wp24/mix/QMF/RGBD-scene-recognition/models/QMF.py`及`train_QMF_nyud2.py`。
- TMC本地原实现：`/home/wp24/mix/QMF/RGBD-scene-recognition/models/TMC.py`。
- QMF论文：https://proceedings.mlr.press/v202/zhang23ar.html
- TMC作者代码：https://github.com/Han-Zongbo/TMC

QMF/TMC是基于原方法的三模态任务适配，不能把小样本任务上的表现或稳定性当作已验证事实。训练代码已通过测试及真实GPU短测，formal_v1正式实验已完成。原始和整理后代码的数值一致性检查见cleanup.md；不能将适配算法称为原QMF完整复现。

## 实现差异补充（2026-09-16）

QMF动态融合公式及累计历史顺序沿用RGB-D思路，但本版排序约束的margin符号与原仓库不同。令r=sign(H_i-H_j)、d为历史差绝对值，当前为relu(r*(q_i-q_j)+d)，原函数为relu(r*(q_i-q_j)-d)。该差异保留以保持formal_v1模型定义，不能写成等价化简。当前EMG为24项特征/通道，不是后续已删除版本的MATLAB九特征。
