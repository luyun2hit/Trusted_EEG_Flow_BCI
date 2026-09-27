> **English**: Case 1 (Section 4.1) — user-wise identity perturbation on eegmmidb (RAND / SN / EMIN / EMAX), independently re-implemented from Chen et al., J. Neural Eng. 22(1):016040, 2025. Quick-start commands and expected results in English are in the repository root README.md; this document is the full Chinese documentation.

# 案例一：用户级扰动身份保护（User-wise Perturbations）实测代码

对应章节：4.1 案例一——发布前的用户身份扰动保护。

## 这是什么

按 Chen et al. (J. Neural Eng. 22(1):016040, 2025, DOI:
10.1088/1741-2552/ad88a5) 的**算法描述独立重实现**的四类用户级扰动
（RAND / SN / EMIN / EMAX）简化实验，在 PhysioNet eegmmidb 的
36 名被试（S001–S036）**左/右手运动想象**子集上真实运行：

- 训练块：R04（T1=想象左手，T2=想象右手）；
- 测试块：R08 + R12（跨块泛化协议）；
- 只扰动训练（发布）侧，测试侧不改动。

> 注意：eegmmidb 的 R06/R10/R14 中 T1/T2 的含义是"双手/双脚"，
> 与 R04/R08/R12 的"左手/右手"**不是同一任务，绝不能合并**。
> 这是本案例修订过程中修正过的关键错误。

## 与原论文/官方仓库的关系与差异

- 模型（EEGNet / ShallowConvNet / 分类头）与数据变换为本书**独立编写**
  （`eeg_models.py`），不复制上游仓库源文件，规避其未明确许可证的
  再分发风险；算法逻辑按上述期刊论文描述实现。
- EMIN 严格按原算法：**随机初始化（不训练）**的 UID 模型上最小化交叉熵；
  EMAX：先训练 3 个替代 UID 模型再最大化交叉熵。
- 简化之处（因此正文称"简化协议下的独立验证"，不称严格复现）：
  36/109 名被试、30/100 训练 epoch、扰动优化 20/100 epoch、
  单一 EEGNet 主攻击器（另附 ShallowConvNet 跨架构攻击器）。

## 文件

| 文件 | 作用 |
|---|---|
| `download.py` | 从 PhysioNet 下载 36 被试 × R04/R08/R12（幂等续跑，校验 curl 退出码与 EDF 完整性） |
| `preprocess_mi2.py` | 4–32 Hz 带通、cue 后 [0,4)s 分段、逐 run 通道标准化 → `windows_lr/` |
| `eeg_models.py` | 独立重实现的 EEGNet/ShallowConvNet/分类头/时间分段重组 |
| `run_perturbations.py` | 主实验：5 扰动 × 5 种子，任务+UID、跨架构攻击、时移鲁棒性、扰动范数报告；断点续跑，缓存绑定配置哈希 |
| `make_fig.py` | 由 `results/perturbation_results.csv` 重绘图 1（均值±标准差，校验种子齐全） |

## 运行

```bash
pip install numpy scipy pyedflib torch matplotlib
python download.py        # 约 540 MB，断网续跑
python preprocess_mi2.py  # 约 1 分钟
python run_perturbations.py   # CPU 约 2.5 小时，可随时中断再续跑
python make_fig.py
```

预期输出（本机实测，5 种子均值±标准差）：

| 扰动 | 任务 BCA(%) | UID BCA(%) EEGNet | UID BCA(%) ShallowConvNet |
|---|---|---|---|
| 无扰动 | 56.40±1.50 | 42.31±2.91 | 31.61±1.93 |
| RAND | 54.86±1.07 | 3.46±0.80 | 5.07±0.90 |
| SN | 54.98±3.51 | 3.39±0.95 | 3.83±0.82 |
| EMIN | 54.94±1.90 | 2.72±0.08 | 3.61±0.69 |
| EMAX | 56.49±2.89 | 3.37±0.52 | 4.07±0.73 |

（UID 随机水平 = 1/36 ≈ 2.78%；完整逐种子结果见 `results/`。）

## 结果能说明什么、不能说明什么

本实验协议验证的是：在扰动发布数据上训练的 UID 模型，难以识别同一批
用户的**干净跨块数据**——即降低"发布数据→后续干净记录"的关联能力。
它**不**证明发布数据已完成去身份化：同一用户的全部样本共享同一扰动
模板，"扰动数据→扰动数据"条件下的模板匹配、聚类与链接攻击不在本协议
覆盖范围内（原论文协议同样如此），发布前应按正文 4.1 节"适用边界"
补充审计。

## 许可与数据来源

- 本目录代码为本书配套材料，可按书附许可使用与修改；
- eegmmidb 数据来自 PhysioNet（https://physionet.org/content/eegmmidb/），
  使用须遵守其数据使用条款（ODC-BY），请勿再分发原始数据；
- 上游算法出处：Chen et al., J. Neural Eng. 22(1):016040, 2025；
  官方仓库 github.com/xqchen914/Unlearnable-examples-for-EEG
  （本书代码未复制其源文件）。
