> **English**: Case 2 (Section 4.2) — protocol-matched PyTorch re-implementation of FedEEG (Priyanshu, 2021) with a loss-based membership-inference spot check; reported configuration: 27 rounds x 3 local epochs under the 60:20:20 validation-split protocol. English quick-start and expected results are in the repository root README.md; this document is the full Chinese documentation.

# 案例二：联邦平均协同学习（FedEEG）PyTorch 重实现

对应章节：4.2 案例二——训练环节的联邦平均协同学习。

## 这是什么

原论文（Priyanshu, 2021，见章节参考文献[14]）的官方实现基于
TensorFlow 1.x / Keras 旧版 API，在当前 Python 环境已无法直接运行。
本目录按其公开源码**逐行复刻协议并以 PyTorch 重实现**，并对原协议作了两处修正：

- 数据：原仓库自带 `dataset_hand_movement/user_a-d.csv`
  （4 名被试手部运动 EEG 特征，每人 2880 样本，112 维带功率特征，3 类平衡）；
- 预处理：补零至 121 维 → **各客户端仅用本地训练集计算均值/标准差做标准化**
  → reshape (11, 11)。
  （修正①：原协议用 user_a 全量数据的全局统计量，其测试数据参与了标准化
  参数计算，构成预处理层面的测试信息泄漏，且与"数据不出本地"的设定冲突。）
- 模型：LSTM(11→16) → LSTM(16→16) → BatchNorm1d → Linear(16,3)；
- 协议：客户端 80:20 划分，Adam(lr=1e-3)、批 12、每轮本地 3 epoch，
  5 轮 FedAvg 简单平均；**每轮聚合后以全局模型在各客户端测试集上评估**
  （修正②：原协议在聚合前逐一测试本地模型，所记录并非聚合后全局模型的性能）；
  对照为等累计训练量的集中式基线；
- 3 个随机种子，报告均值±标准差。

## 文件

| 文件 | 作用 |
|---|---|
| `fedegg_reimpl.py` | 联邦主循环 + 集中式基线（可按种子分次运行：`python fedegg_reimpl.py 42`） |
| `merge_results.py` | 合并各种子部分结果，生成 `federated_rounds.csv` 与 `summary.json` |
| `membership_inference.py` | 成员推断抽查：基于损失的白盒 MIA（Yeom AUC 口径），逐轮评估聚合全局模型，训练配置与主实验逐参数一致 |
| `mia_merge.py` | 合并各种子 MIA 结果，生成 `membership_inference.json` |
| `local_only_baseline.py` | 本地独立训练基线（各客户端不聚合，等累计训练量） |
| `sweep.py` | 调参筛查：仅改轮数/本地 epoch/学习率的联邦曲线扫描 |
| `final_run.py` | 调参后正式配置（24 轮 × 3 本地 epoch）四臂运行：fed / centralized / local / mia（+miacent） |
| `merge_tuned.py` | 合并调参后正式结果；原 5 轮配置结果归档为 `results/*_r5.*` |
| `valsplit_run.py` | 验证集协议（60:20:20 训练/验证/测试三划分）重跑：sweep / fed / centralized / local / mia / miacent 六臂；轮数只按验证集曲线选取，测试集仅用于最终报告 |
| `merge_valsplit.py` | 合并验证集协议正式结果（论文报告值）；早期 24 轮配置结果归档为 `results/*_tuned24.*` |
| `make_fig.py` | 由 `results/federated_rounds.csv` 重绘图 2：逐轮验证曲线 + R* 处各客户端测试终值（测试集仅参与最终报告；跨平台中文字体回退） |
| `results/` | 本机实测：逐轮 CSV、汇总 JSON、成员推断 JSON、图 |

## 运行

```bash
git clone https://github.com/AmanPriyanshu/FedEEG.git
export FEDEEG_DATA=/path/to/FedEEG/dataset_hand_movement   # Windows: set FEDEEG_DATA=...
python fedegg_reimpl.py          # 或分次：python fedegg_reimpl.py 42 / 7 / 2024
python merge_results.py
python make_fig.py
python membership_inference.py   # 成员推断抽查（或分次给种子）
python mia_merge.py
```

未设置 `FEDEEG_DATA` 时脚本会给出明确提示而非晦涩报错。
原 5 轮配置全流程 CPU 约 15 分钟；调参后配置（24 轮，3 种子）约 50 分钟；
验证集协议正式配置（27 轮 × 6 臂 × 3 种子，论文报告值）约 45 分钟。

## 预期结果（本机实测，3 种子均值；修正协议后）

**原协议 5 轮配置**：聚合全局模型 46.15% ± 0.4%，集中式基线 53.33% ± 1.1%
（差距约 7.2 个百分点）；逐轮曲线在五轮内持续上升，属欠训练状态。
原论文报告 57.67% vs 61.83%（其记录口径为聚合前本地模型）。

**正式配置（27 轮 × 3 本地 epoch，60:20:20 三划分；轮数只按验证集曲线选取，
测试集仅用于最终报告——论文报告值，协议见 valsplit_run.py）**：
聚合全局模型 **47.90% ± 1.0%**（末轮各客户端 39.24%–54.86%）；
等累计训练量（81 epoch）集中式基线 **54.92% ± 0.8%**、本地独立训练基线
**63.79% ± 0.5%**。联邦与集中式差距约 7.0 个百分点，排序不变
（本地 > 集中式 > 联邦）：该 4 客户端、强异构小规模设定下联邦协作不带来
精度收益，其价值在"数据不出本地"的架构属性。逐轮数据见
`results/federated_rounds.csv`。

早期调参结果（80:20 协议、24 轮，轮数选择参考了测试集曲线，属轻度测试
自适应，仅存档备查）：聚合全局模型 50.91% ± 0.2%，集中式 56.61% ± 1.0%，
本地 66.85% ± 0.8%；归档为 `results/summary_tuned24.json` 等 `_tuned24`
文件，原 5 轮配置结果归档为 `_r5` 文件。

## 隐私验收：成员推断抽查（对应正文 4.2 节）

架构特征只保证原始数据不出本地；聚合后的全局模型是否携带成员信息，
须以攻击实验验收。`membership_inference.py` 实现基于损失的白盒成员推断
（Yeom AUC 口径，阈值无关）：成员 = 各客户端训练样本，非成员 = 同客户端
留出测试样本，逐轮评估聚合全局模型，3 种子均值。

实测：原 5 轮欠训练配置下逐轮 AUC 贴近随机（末轮 0.506±0.001），
阴性源于欠拟合（训练—测试间隙≈0），不构成隐私证据。
正式配置下（验证集协议 27 轮，`results/membership_inference.json`），
末轮 AUC 升至 **0.547 ± 0.007**（FPR=10% 处 TPR≈0.125，训练—测试间隙
+0.056~+0.079），等训练量集中式模型 AUC 达 **0.651 ± 0.007**。
限定：两种模型拟合水平不同（测试准确率 47.90% 对 54.92%），AUC 差异
同时反映聚合过程与过拟合程度的差别，不能单独归因于联邦聚合；
抽查仅针对全局模型、仅用样本损失评分，不覆盖利用梯度或中间表示的
更强白盒攻击，也不覆盖对单客户端上传更新的推断。
早期 24 轮配置 MIA 归档于 `results/membership_inference_tuned24.json`，
原 5 轮配置归档于 `results/membership_inference_r5.json`。

## 结果解释范围

本实验是联邦协议在小客户端、强异构设定下的概念验证复刻：训练预算增加后
精度与成员泄漏同步上升（欠拟合缓解、MIA 由阴性转为可检测），7.0 个百分点
的联邦—集中式差距不应外推为"联邦学习在 EEG 上的通用代价"；客户端上传侧
噪声注入（差分隐私调度）与安全聚合是影响结论的两个主要扩展维度，
可作为后续实验方向。

## 许可与数据来源

- 本目录代码为本书配套材料（PyTorch 重实现部分）；
- 上游 FedEEG 仓库（含数据）以 **MIT 许可证**发布
  （Copyright (c) 2021 Aman Priyanshu），使用时请保留其许可声明并引用原论文；
- 本代码包不再分发该数据集，请按上方命令从原仓库获取。
