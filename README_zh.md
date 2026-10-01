# 脑电数据保护与隐私计算——三个经典案例的本机实测

本代码包配套《脑电数据的可信流转：三类隐私保护路径与可验证实践》。

## 环境

- 案例一、二：Python 3.10+（实测 3.12）、纯 CPU 即可；依赖 `torch`（实测 2.9.1+cpu）、`numpy`、`scipy`、`scikit-learn`、`pyedflib`、`matplotlib`、`pandas`
- 案例三：实测为 Python 3.11 + PyTorch 2.11.0+cu126 + PyWavelets 1.8，单机 GPU（NVIDIA GTX 1660 SUPER）加速；纯 CPU 亦可运行（训练明显变慢，约 1.5 小时量级，估算值）
- 安装示例：`pip install torch numpy scipy scikit-learn pyedflib matplotlib pandas PyWavelets`（锁定版本见 `requirements.txt`）
- 操作系统：Windows / Linux / macOS 均可（实测为 Windows + Git Bash）

## 目录结构

```
随书代码包/
├── case1_user_wise_perturbations/   案例一：发布前的用户身份扰动保护
│   ├── README.md                    案例一详细说明与结果解释范围
│   ├── download.py                  eegmmidb 左/右手 MI 子集下载器（幂等续跑）
│   ├── preprocess_mi2.py            预处理：带通 + 分段 + 逐run标准化
│   ├── eeg_models.py                独立重实现的 EEGNet/ShallowConvNet/分类头
│   ├── run_perturbations.py         四种扰动生成与任务/UID 评估（配置哈希缓存，可断点续跑）
│   ├── make_fig.py                  图 1 重绘（均值±标准差）
│   └── results/                     本机实测结果（CSV/图）
├── case2_fedeeg/                    案例二：联邦平均协同学习
│   ├── README.md                    案例二详细说明、协议与许可（上游 MIT）
│   ├── fedegg_reimpl.py             LSTM+FedAvg 的 PyTorch 重实现
│   ├── merge_results.py             合并各种子结果为 CSV/JSON
│   ├── membership_inference.py      成员推断抽查（基于损失的白盒 MIA）
│   ├── mia_merge.py                 合并各种子 MIA 结果
│   ├── local_only_baseline.py       本地独立训练基线（各客户端不聚合，等累计训练量）
│   ├── sweep.py                     调参筛查（轮数/本地 epoch/学习率）
│   ├── final_run.py                 调参后配置（24 轮×3 epoch）四臂运行
│   ├── merge_tuned.py               合并调参后结果（原 5 轮归档为 *_r5）
│   ├── valsplit_run.py              验证集协议（60:20:20）六臂重跑（论文报告值）
│   ├── merge_valsplit.py            合并验证集协议正式结果（24 轮归档为 *_tuned24）
│   ├── make_fig.py                  收敛曲线绘图
│   └── results/                     本机实测结果（CSV/JSON/图 + 种子级 partial_*.json、hist_*.npy）
└── case3_epilepsygan_chbmit/        案例三：癫痫发作合成数据发布
    ├── README.md                    案例三详细说明、失效模式与许可
    ├── download_segments.py         CHB-MIT 分段下载器（HTTP Range 按需下载）
    ├── preprocess.py                4 秒窗切分（ictal 3s 重叠 / interictal）
    ├── epilepsygan_chbmit.py        条件 GAN 留一法训练（PyTorch）
    ├── evaluate_chbmit.py           TSTR 效用 + 重识别隐私 + 最近邻审计 + 谱相似度评估
    ├── attacker_power_control.py    攻击器能力阳性对照（同状态、无重叠窗）
    ├── regen_timeiso.py             时间隔离再生成（条件窗限于训练侧前半时段）
    ├── run_timeiso_eval.py          时间隔离版评估驱动（UID / NN+谱相似度）
    ├── tstr_td_features.py          时域/非线性特征 TSTR 复核（指标耦合对照）
    └── results/                     本机实测结果（JSON/图 + 逐种子日志 uid_logs/）
```

> 关于上游源码：本代码包的三个案例均为**独立重实现**，不捆绑上游仓库源文件。
> 复现案例二需自行克隆 FedEEG 上游仓库（含数据，见下文案例二说明）；
> 三篇原论文与其官方源码的获取方式见各案例 README 与章节参考文献，
> 请按上游各自许可使用，不要将上游代码或论文 PDF 并入本发布包。

---

## 案例一：用户级扰动（User-wise Perturbations @ eegmmidb）

**自包含**：模型与扰动算法按原论文期刊版本（Chen et al., J. Neural Eng.
22(1):016040, 2025）的算法描述**独立重实现**（`eeg_models.py`），
不依赖、不复制上游仓库源文件。

**运行步骤**（在 `case1_user_wise_perturbations/` 下）：

```bash
# 1) 下载 eegmmidb 36 名被试 × R04/R08/R12 左/右手运动想象 run（约 270MB，幂等续跑）
python download.py                            # 自动循环直至全部完成

# 2) 预处理：[4,32]Hz 带通，提示后 [0,4]s 窗，逐 run 通道标准化
python preprocess_mi2.py                      # 输出 windows_lr/block{1,2,3}.npz

# 3) 扰动实验（5 种扰动 × 5 种子 + 跨架构攻击器 + 时移鲁棒性 + 范数报告，
#    全程约 2.5 小时，缓存绑定配置哈希，可断点续跑）
python run_perturbations.py                   # 被中断后重复执行直至 ALL DONE

# 4) 重绘正文图 1（校验种子齐全，均值±标准差）
python make_fig.py
```

**预期结果**（`results/perturbation_results.csv` 等，5 种子均值±标准差）：

| 扰动 | 任务 BCA | UID BCA (EEGNet) | UID BCA (ShallowConvNet) |
|---|---|---|---|
| 无扰动 | 56.40±1.50% | 42.31±2.91% | 31.61±1.93% |
| RAND | 54.86±1.07% | 3.46±0.80% | 5.07±0.90% |
| SN | 53.91±1.33% | 3.11±0.60% | 3.63±0.79% |
| EMIN | 54.94±1.90% | 2.72±0.08% | 3.61±0.69% |
| EMAX | 56.49±2.89% | 3.37±0.52% | 4.07±0.73% |

（UID 随机水平 = 1/36 ≈ 2.78%。）时移鲁棒性见 `results/robustness.csv`，
跨架构攻击见 `results/cross_arch.csv`，实际扰动范数见
`results/perturbation_norms.csv`。

**关键语义提醒**：eegmmidb 的 R04/R08/R12（T1/T2=左/右手想象）与
R06/R10/R14（T1/T2=双手/双脚想象）标注符号相同但任务不同，**不能合并**，
本案例只使用前者；这与原论文 MI2 的左/右手任务定义一致。
另注意：任务基线对预处理敏感——若不做逐 run 通道标准化，跨 block 任务
精度会跌至随机水平附近，这是本章写作过程中发现并已写入正文的实践要点。

**结果解释范围**：本协议验证"扰动发布数据 → 干净跨块数据"方向的
关联能力下降；不构成完整去身份化（扰动模板在发布数据内部仍可能被
匹配/聚类利用），详见 mi2 目录 README 与正文 4.1 节。

---

## 案例二：联邦平均（FedEEG，PyTorch 重实现）

**额外依赖**：原仓库自带数据（原实现为 TF1.x/Keras，已无法在现代环境运行，
本案例按其源码逐行复刻协议并以 PyTorch 重实现）

```bash
git clone https://github.com/AmanPriyanshu/FedEEG.git   # 解压后主目录名 FedEEG-main
# 数据在 FedEEG-main/dataset_hand_movement/user_a-d.csv
```

**运行步骤**（在 `case2_fedeeg/` 下）：

```bash
FEDEEG_DATA=/path/to/FedEEG-main/dataset_hand_movement python fedegg_reimpl.py
python merge_results.py   # 合并各种子结果（原 5 轮配置）
python make_fig.py        # 生成逐轮收敛图
python membership_inference.py && python mia_merge.py   # 成员推断抽查（隐私验收）
# 调参后配置（24 轮 × 3 本地 epoch）：
python final_run.py fed 42 && python final_run.py centralized 42 && ...  # 按种子分次
python merge_tuned.py
# 验证集协议正式配置（27 轮，60:20:20 三划分，论文报告值）：
python valsplit_run.py sweep 42          # 轮数筛查（仅用验证集）
python valsplit_run.py fed 42 27 && ...  # 六臂 × 3 种子，按种子分次
python merge_valsplit.py && python make_fig.py
```

**预期结果**（3 种子均值，修正协议后）：原 5 轮配置聚合全局模型 46.15%±0.4%、
集中式基线 53.33%±1.1%；正式配置（验证集协议：60:20:20 三划分、27 轮，
轮数只按验证集选取）聚合全局模型 47.90%±1.0%，等训练量集中式基线
54.92%±0.8%、本地独立训练基线 63.79%±0.5%（联邦—集中式差距约 7.0 个
百分点，排序不变：本地 > 集中式 > 联邦）。本实现对原协议有两处修正：
标准化参数改为各客户端仅用本地训练集计算；每轮聚合后以全局模型评估
（原协议记录的是聚合前本地模型，原论文报告 57.67% vs 61.83% 即该口径）。
逐轮准确率见 `results/federated_rounds.csv`。

**隐私验收（成员推断抽查）**：对逐轮聚合全局模型做基于损失的成员推断
（受 Yeom 阈值攻击启发的 AUC 口径，3 种子）。原 5 轮欠训练配置下 AUC 贴近
随机（末轮 0.506±0.001，阴性源于欠拟合，不构成隐私证据）；正式配置
（27 轮）下末轮 AUC 0.547±0.007，等训练量集中式模型 AUC 0.651±0.007——
两者拟合程度不同，差异不能单独归因于联邦聚合；详见 `case2_fedeeg/README.md`。

---

## 案例三：癫痫发作合成数据（EpilepsyGAN @ CHB-MIT）

完全自包含，数据从 PhysioNet 公开免费获取（无需账号），
利用 EDF 定长记录结构 + HTTP Range 请求，仅下载所需片段（约 60MB 而非整库 40GB+）。

**运行步骤**（在 `case3_epilepsygan_chbmit/` 下）：

```bash
python download_segments.py     # 按需下载 4 名被试的发作段与间期段（可续跑）
python preprocess.py            # 4 秒窗切分，输出 windows/*.npy
python epilepsygan_chbmit.py    # 4 折留一训练（默认 EPOCHS=40 + 身份对抗去偏；
                                #   GPU 约 5 分钟，CPU 约 1.5 小时（估算），断点续跑；
                                #   逐折清单 manifest_<sid>.json 绑定配置/数据/产物哈希）
python regen_timeiso.py         # 时间隔离再生成：条件窗限于间期池前半时段（与 TSTR 训练侧同侧）
# 时间隔离版完整评估（论文最终数字，生成 results/summary_timeiso.json）：
GAN_DIR=gan_results_timeiso MODE=all OUT_JSON=results/summary_timeiso.json python run_timeiso_eval.py
GAN_DIR=gan_results_timeiso python tstr_td_features.py   # 时域特征 TSTR 复核
python attacker_power_control.py # 攻击器能力阳性对照（同状态、无重叠窗）
python make_fig_timeiso.py       # 重绘图 3/4
```

**最终结果（时间区间隔离协议，v5 修正版实现，`results/summary_timeiso.json`，2026-09-30 重跑）**：

- 效用（TSTR，几何均值，Welch 带功率特征；间期窗非重叠且按时间序对半分、
  生成条件窗限于训练侧时段）：真实数据基线 83.94% vs 去偏后合成 69.12%；
  去偏前对照（λ_adv=0，同协议）82.33%，即身份去偏的效用代价约 13.2 个百分点；
  时域 20 维特征复核：合成 63.78% vs 基线 85.57%
- 隐私（患者重识别，平衡准确率 BCA 主指标，5 个攻击器初始化种子，随机水平 25%）：
  合成发作 49.90%±0.11%（2.00×随机）；去偏前对照 47.73%±1.21%（1.91×）；
  λ_adv=5（3 种子）47.68%±0.14%（1.91×）——梯度反转去偏在修正后的实现下
  不能降低可重识别性；真实发作 18.53%±3.38%（0.74×随机，成因未能确证），
  跨状态对照未通过，隐私结论仅以"合成 vs 真实"的相对比较为限；因每患者合成
  数据来自不同留一折，该信号报告为折相关重识别信号（见正文 4.3.3）
- 攻击器能力阳性对照（`results/attacker_power_control.json`，同状态、无重叠窗）：
  间期→间期 56.30%±4.55%、发作→发作 68.65%±1.39%（2.25×/2.75×随机），
  提示攻击器在单一状态内能够学习身份特征
- 最近邻记忆审计：synth/real 最近邻距离比 0.93–1.10 ≈ 1，近重复率（<1e-3）为 0，
  未观察到逐样本记忆迹象
- 谱余弦相似度：real-real 0.653–0.737 与 real-synth 0.606–0.737 总体区间重叠
  （患者间排序不一致，该指标仅支持群体层面频谱合理性）
- 非隔离版结果留存于 `results/summary.json` 备查

**审计结论（本代码包 2026-09 多轮修订的核心结果）**：v1 的重识别攻击器
能力过弱（无 DWT 输入、12 epoch、单次运行），隐私结论证据不足；攻击器对齐
原论文后，v3 曾观察到身份对抗去偏"有效"的表象，但 v4/v5 的逐项修正
（谱损失逐通道化、抗混叠降采样、时间隔离边界、逐折种子、逐通道滤波）表明
该表象与实现缺陷混淆。在修正后的实现下，梯度反转去偏在 λ_adv=0/2/5 下
均不能降低合成发作的可重识别性（1.91–2.00×随机），反而带来显著效用代价
（−13.2pp）。由于多项修正并行引入，不作单因素归因。完整的隐私—效用权衡表
与配置说明见案例三 README。

**复现纪律提示**：若自行修改网络或评估管道，注意本章报告的四个失效模式——
LSGAN 判别器误接 sigmoid 会导致生成振幅坍缩；L1 参照随机配对会导致逐点中位数回归
（故 λ 下调为 10、判别器保持线性输出）；评估时间期窗若训练/测试两侧复用，
特异性会虚高至 1.000 的假象（本包已修复为非重叠窗按时间序对半切分，
且 regen_timeiso.py 将生成条件窗限定于训练侧前半时段，实现原始时间区间隔离）；
条件 GAN 的条件通路会携带折相关的重识别信号，且梯度反转去偏未能消除
（须用对齐原论文的强化攻击器审计，详见案例三 README）。

---

## 数据集获取

| 数据集 | 用途 | 获取 |
|---|---|---|
| PhysioNet eegmmidb 1.0.0 | 案例一 | https://physionet.org/content/eegmmidb/1.0.0/ （脚本自动下载） |
| EEG data from hands movement | 案例二 | 随 FedEEG 仓库分发（GitHub） |
| CHB-MIT Scalp EEG 1.0.0 | 案例三 | https://physionet.org/content/chbmit/1.0.0/ （脚本按需下载） |

## 引用

使用本代码包请引用对应章节及三篇原论文（Chen et al., J. Neural Eng. 2025；Priyanshu, 2021；Pascual et al., 2021），
完整文献信息见章节参考文献[13][14][15]。
