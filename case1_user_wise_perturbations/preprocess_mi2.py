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

v3 变更（严格模式）：
  - 运行前清空输出目录中的旧 block*.npz，杜绝不完整结果残留；
  - 任何缺失或格式异常的 EDF 一律报错退出（不再静默跳过）；
  - 每个 block 输出前断言：540 个试次（36 被试 × 每 run 15 试次）、
    64 通道、640 采样点、36 名被试齐全、数值有限；
  - 新增 --no-norm：跳过标准化，输出到 windows_lr_nonorm/，
    供"标准化是否必要"消融实验使用。
"""
import argparse
import os
import sys
import numpy as np
import pyedflib
from scipy.signal import butter, filtfilt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

SUBJECTS = [f"S{i:03d}" for i in range(1, 37)]
BLOCKS = {1: ["R04"], 2: ["R08"], 3: ["R12"]}  # 仅左/右手想象 run
FS = 160
N_CHANS = 64
WIN = 4 * FS  # 640 samples
TRIALS_PER_RUN = 15          # eegmmidb 每个任务 run 15 个试次
EXPECTED_PER_BLOCK = len(SUBJECTS) * TRIALS_PER_RUN  # 540


def extract_block(runs, normalize):
    b, a = butter(4, [4, 32], btype="bandpass", fs=FS)
    X, Y, S = [], [], []
    problems = []
    for si, sub in enumerate(SUBJECTS):
        for run in runs:
            path = os.path.join(RAW, f"{sub}{run}.edf")
            if not os.path.exists(path):
                problems.append(f"缺失 {path}")
                continue
            try:
                f = pyedflib.EdfReader(path)
                n = f.signals_in_file
                fs = f.getSampleFrequency(0)
                if n != N_CHANS or int(fs) != FS:
                    problems.append(f"格式异常 {path}: {n} 通道 {fs}Hz")
                    f.close()
                    continue
                sig = np.array([f.readSignal(i) for i in range(n)])
                onsets, durations, descs = f.readAnnotations()
                f.close()
            except Exception as e:
                problems.append(f"读取失败 {path}: {e}")
                continue
            sig = filtfilt(b, a, sig, axis=-1)
            if normalize:
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
                    problems.append(f"试次截断 {path} @ {ons:.1f}s")
                    continue
                X.append(seg.astype(np.float32))
                Y.append(0 if desc == "T1" else 1)  # 0=左手想象, 1=右手想象
                S.append(si)
                cnt[desc] += 1
            print(f"{sub} {run}: 左手(T1)={cnt['T1']} 右手(T2)={cnt['T2']}")
    return X, Y, S, problems


def check_block(blk, X, Y, S):
    """严格完整性断言，不满足即退出。"""
    assert X.shape == (EXPECTED_PER_BLOCK, N_CHANS, WIN), \
        f"block{blk} 形状 {X.shape} != ({EXPECTED_PER_BLOCK},{N_CHANS},{WIN})"
    assert np.isfinite(X).all(), f"block{blk} 含 NaN/Inf"
    subs = np.unique(S)
    assert len(subs) == len(SUBJECTS), \
        f"block{blk} 被试数 {len(subs)} != {len(SUBJECTS)}"
    cnt = np.bincount(S, minlength=len(SUBJECTS))
    assert (cnt == TRIALS_PER_RUN).all(), \
        f"block{blk} 被试试次数不齐: {dict(zip(SUBJECTS, cnt))}"
    assert set(np.unique(Y)) == {0, 1}, f"block{blk} 类别不全: {np.unique(Y)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-norm", action="store_true",
                    help="跳过 run 内标准化，输出到 windows_lr_nonorm/")
    args = ap.parse_args()
    out = os.path.join(HERE, "windows_lr_nonorm" if args.no_norm
                       else "windows_lr")
    os.makedirs(out, exist_ok=True)
    # 清空旧输出，避免不完整结果残留
    for name in os.listdir(out):
        if name.startswith("block") and name.endswith(".npz"):
            os.remove(os.path.join(out, name))
            print(f"删除旧文件 {name}")

    all_problems = []
    for blk, runs in BLOCKS.items():
        X, Y, S, problems = extract_block(runs, normalize=not args.no_norm)
        all_problems += problems
        if problems:
            print(f"block{blk}: {len(problems)} 个文件问题，中止：")
            for p in problems[:10]:
                print("  " + p)
            sys.exit(1)
        X = np.stack(X)
        Y = np.array(Y, dtype=np.int64)
        S = np.array(S, dtype=np.int64)
        check_block(blk, X, Y, S)
        np.savez_compressed(os.path.join(out, f"block{blk}.npz"),
                            x=X, y=Y, s=S)
        print(f"block{blk}: x{X.shape} 类别计数={np.bincount(Y)} "
              f"被试数={len(np.unique(S))}")
    print("预处理完成（全部完整性断言通过）。")


if __name__ == "__main__":
    main()
