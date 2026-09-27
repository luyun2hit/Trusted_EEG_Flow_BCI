# -*- coding: utf-8 -*-
"""
eeg_models.py — 案例一独立重实现的模型与变换模块。

本模块为本书随书代码独立编写，不复制任何上游仓库源文件：
- EEGNet 架构按公开发表论文 Lawhern et al., "EEGNet: a compact convolutional
  neural network for EEG-based brain-computer interfaces", J. Neural Eng. 2018
  的结构描述实现（紧凑卷积：时间卷积 + 深度可分离空间卷积）。
- ShallowConvNet 架构按 Schirrmeister et al., Human Brain Mapping 2017 的
  浅层卷积结构描述实现，仅用作跨架构 UID 攻击器。
- Classifier / Discriminator 为常规线性 / 两层 MLP 分类头。
- time_chunk_shuffle 为按时间分段随机重排的常规数据增强。
扰动算法（RAND/SN/EMIN/EMAX）的算法逻辑源自 Chen et al.,
"User-wise Perturbations for User Identity Protection in EEG-based
Brain-Computer Interfaces"（J. Neural Eng. 2025），见 run_perturbations.py
中的引用与说明；本仓库代码为按论文算法描述的独立重实现。
"""
import numpy as np
import torch
import torch.nn as nn


def calc_out_size(model, chans, samples):
    """用一次前向传播推断特征维度。"""
    with torch.no_grad():
        out = model(torch.zeros(1, 1, chans, samples))
    return out.shape[-1]


def init_weights(m):
    """常规初始化：卷积/线性层 Xavier 均匀，BN 权重 1 偏置 0。"""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class EEGNet(nn.Module):
    """EEGNet（Lawhern 2018）：(B,1,C,T) -> (B, F2 * ceil(T/32)) 特征。"""

    def __init__(self, chans, samples, kern_length=64, F1=4, D=2, F2=8,
                 dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((kern_length // 2 - 1,
                          kern_length - kern_length // 2, 0, 0)),
            nn.Conv2d(1, F1, (1, kern_length), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x.reshape(x.size(0), -1)


class ShallowConvNet(nn.Module):
    """浅层卷积网络（Schirrmeister 2017 风格），跨架构攻击器用。"""

    def __init__(self, chans, samples, mid_dim=40, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, mid_dim, (1, 13)),
            nn.Conv2d(mid_dim, mid_dim, (chans, 1)),
            nn.BatchNorm2d(mid_dim),
            nn.ELU(),
            nn.AvgPool2d((1, 35), stride=(1, 7)),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block1(x).reshape(x.size(0), -1)


class Classifier(nn.Module):
    """线性分类头（任务分类用）。"""

    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)

    def forward(self, feat):
        return self.fc(feat)


class Discriminator(nn.Module):
    """两层 MLP 身份判别头（UID 用）。"""

    def __init__(self, input_dim, n_subjects, hidden=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, feat):
        return self.net(feat)


class TimeChunkShuffle:
    """把末维（时间）切成 num_chunks 段后随机重排，batch 级共享同一排列。"""

    def __init__(self, num_chunks):
        self.num_chunks = num_chunks

    def __call__(self, x):
        chunks = torch.chunk(x, self.num_chunks, dim=-1)
        order = torch.randperm(len(chunks))
        return torch.cat([chunks[i] for i in order], dim=-1)


def freeze(model):
    """冻结参数（只进 eval 还不够，需关闭 requires_grad 以节省扰动优化显存/内存）。"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
