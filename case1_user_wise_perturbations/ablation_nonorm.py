# -*- coding: utf-8 -*-
"""ablation_nonorm.py — 无标准化消融：验证 run 内标准化对任务性能的必要性。

读取 preprocess_mi2.py --no-norm 生成的 windows_lr_nonorm/（仅带通滤波，
不做 run 内标准化），在无任何扰动下训练任务分类器（EEGNet），
5 个种子，输出 results/nonorm_ablation.csv。

用途：支撑正文"不做 run 内标准化时任务准确率降至随机水平附近"的论断，
使该论断有可直接复跑的实验依据。
"""
import csv
import os
import numpy as np

from run_perturbations import train_eval, SEEDS, HERE

NONORM = os.path.join(HERE, "windows_lr_nonorm")
OUT = os.path.join(HERE, "results", "nonorm_ablation.csv")


def load_nonorm():
    d1 = np.load(os.path.join(NONORM, "block1.npz"))
    d2 = np.load(os.path.join(NONORM, "block2.npz"))
    d3 = np.load(os.path.join(NONORM, "block3.npz"))
    xtr, ytr = d1["x"][:, None].astype(np.float32), d1["y"]
    xte = np.concatenate([d2["x"], d3["x"]])[:, None].astype(np.float32)
    yte = np.concatenate([d2["y"], d3["y"]])
    return xtr, ytr, xte, yte


def main():
    if not os.path.isdir(NONORM):
        raise SystemExit("缺少 windows_lr_nonorm/，请先运行 "
                         "preprocess_mi2.py --no-norm")
    xtr, ytr, xte, yte = load_nonorm()
    print(f"无标准化数据: train {xtr.shape} test {xte.shape} "
          f"std {xtr.std():.2f}")
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fp:
            done = {int(r["seed"]) for r in csv.DictReader(fp)}
    hdr = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["seed", "task_bca_nonorm"])
        if hdr:
            w.writeheader()
        for seed in SEEDS:
            if seed in done:
                continue
            acc = train_eval(xtr, ytr, xte, yte, 2, "eegnet", seed)
            w.writerow({"seed": seed, "task_bca_nonorm": round(acc, 4)})
            fp.flush()
            print(f"NONORM seed{seed}: task_bca={acc:.4f}", flush=True)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
