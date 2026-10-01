# -*- coding: utf-8 -*-
"""dsp.py — 共享信号处理工具。

decimate()：逐通道抗混叠 2 倍降采样（256 Hz -> 128 Hz）。
输入为双通道拼接窗 [ch1: L 点 | ch2: L 点]；v5 修复：先按通道拆分再分别
滤波，旧版对拼接向量整体滤波会把通道 1 末端与通道 2 起点误当作连续信号，
在连接点产生跨通道振铃污染（实测最大 25 µV，集中于拼接处）。
训练、生成与全部评估必须使用同一实现。
"""
import numpy as np
from scipy.signal import resample_poly


def decimate(X, factor=2, n_channels=2):
    """逐通道抗混叠降采样。X: (..., n_channels*L) -> (..., n_channels*L/factor)。"""
    X = np.asarray(X)
    length = X.shape[-1]
    if length % n_channels != 0:
        raise ValueError("拼接长度不能被通道数整除")
    Xc = X.reshape(X.shape[:-1] + (n_channels, length // n_channels))
    Yc = resample_poly(Xc, up=1, down=factor, axis=-1)
    return Yc.reshape(X.shape[:-1] + (-1,)).astype(np.float32)
