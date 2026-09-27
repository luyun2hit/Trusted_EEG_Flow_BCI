# -*- coding: utf-8 -*-
"""
preprocess_mi2.py — eegmmidb (MI2) 预处理，语义一致的左/右手运动想象划分。

数据语义（PhysioNet eegmmidb 官方说明）：
  R04/R08/R12：想象左手（T1） vs 想象右手（T2）
  R06/R10/R14：想象双手（T1） vs 想象双脚（T2）
本脚本只使用 R04/R08/R12，保证标签语义统一为"左手 vs 右手运动想象"：
  block1 = R04（训练块），block2 = R08、block3 = R12（两个独立测试块）
与原论文（Chen et al., JNE 2025，109 名被试左/右手任务）任务定义一致；
本书案例受 CPU 预算限制使用 S001–S036 共 36 名被试子集。

处理流程：4–32 Hz 带通（4 阶 Butterworth，零相位）→ 取 cue 后 [0,4)s
（640 点 @160Hz）→ 每个 run 内逐通道标准化（消除跨块幅度漂移）。
输出：windows_lr/block{1,2,3}.npz，x:(N,64,640) float32, y:(N,), s:(N,)。
"""
import os
import sys
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "windows_lr")

SUBJECTS = [f"S{i:03d}" for i in range(1, 37)]
BLOCKS = {1: ["R04"], 2: ["R08"], 3: ["R12"]}  # 仅左/右手想象 run
FS = 160
N_CHANS = 64
WIN = 4 * FS  # 640 samples


def extract_block(runs):
    b, a = butter(4, [4, 32], btype="bandpass", fs=FS)
    X, Y, S = [], [], []
    missing, bad = [], []
    for si, sub in enumerate(SUBJECTS):
        for run in runs:
            path = os.path.join(RAW, f"{sub}{run}.edf")
            if not os.path.exists(path):
                missing.append(path)
                continue
            f = pyedflib.EdfReader(path)
            n = f.signals_in_file
            fs = f.getSampleFrequency(0)
            if n != N_CHANS or int(fs) != FS:
                bad.append((path, n, fs))
                f.close()
                continue
            sig = np.array([f.readSignal(i) for i in range(n)])
            onsets, durations, descs = f.readAnnotations()
            f.close()
            sig = filtfilt(b, a, sig, axis=-1)
            sig = (sig - sig.mean(axis=-1, keepdims=True)) / \
                  (sig.std(axis=-1, keepdims=True) + 1e-8)
            cnt = {"T1": 0, "T2": 0}
            for ons, dur, desc in zip(onsets, durations, descs):
                if desc not in ("T1", "T2"):
                    continue  # 跳过 T0 静息
                st = int(round(ons * FS))
                if dur < 4.0:
                    print(f"WARN {sub}{run}: {desc} 标注时长 {dur:.2f}s < 4s")
                seg = sig[:, st:st + WIN]
                if seg.shape[-1] < WIN:
                    continue  # run 末尾截断的试次
                X.append(seg.astype(np.float32))
                Y.append(0 if desc == "T1" else 1)  # 0=左手想象, 1=右手想象
                S.append(si)
                cnt[desc] += 1
            print(f"{sub} {run}: 左手(T1)={cnt['T1']} 右手(T2)={cnt['T2']}")
    return X, Y, S, missing, bad


def main():
    os.makedirs(OUT, exist_ok=True)
    all_missing, all_bad = [], []
    for blk, runs in BLOCKS.items():
        X, Y, S, missing, bad = extract_block(runs)
        all_missing += missing
        all_bad += bad
        if not X:
            print(f"block{blk}: 无数据，跳过"); continue
        X = np.stack(X)
        Y = np.array(Y, dtype=np.int64)
        S = np.array(S, dtype=np.int64)
        np.savez_compressed(os.path.join(OUT, f"block{blk}.npz"),
                            x=X, y=Y, s=S)
        print(f"block{blk}: x{X.shape} 类别计数={np.bincount(Y)} "
              f"被试数={len(np.unique(S))}")
    if all_missing:
        print(f"缺失文件 {len(all_missing)} 个（已跳过）：", all_missing[:5])
    if all_bad:
        print(f"格式异常文件 {len(all_bad)} 个（已跳过）：", all_bad[:5])
    n_done = len([f for f in os.listdir(OUT) if f.endswith(".npz")])
    if n_done < len(BLOCKS):
        print("预处理不完整，请先运行 download.py 补齐数据。")
        sys.exit(1)
    print("预处理完成。")


if __name__ == "__main__":
    main()
