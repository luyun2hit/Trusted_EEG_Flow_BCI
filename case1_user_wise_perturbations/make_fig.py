# -*- coding: utf-8 -*-
"""make_fig.py — 由结果 CSV 重新生成图 1（任务/UID 平衡准确率，均值±标准差）。

校验每种方法的 5 个种子恰好为 {0,1,2,3,4}（不多不少、不重复）；
缺失或重复时拒绝绘图并提示。
误差棒为总体标准差（np.std 默认 ddof=0），与正文口径一致。
结果目录自动发现 results/ 下最新（按修改时间）且含
perturbation_results.csv 的子目录，也可用 --res 显式指定。
"""
import argparse
import csv
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["no", "rand", "sn", "optim_linf", "adv_linf"]
NAMES = {"no": "No perturbation", "rand": "RAND", "sn": "SN",
         "optim_linf": "EMIN", "adv_linf": "EMAX"}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def find_res_dir():
    root = os.path.join(HERE, "results")
    cands = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "perturbation_results.csv" in filenames:
            cands.append(dirpath)
    if not cands:
        sys.exit("results/ 下找不到 perturbation_results.csv")
    return max(cands, key=lambda d: os.path.getmtime(
        os.path.join(d, "perturbation_results.csv")))


def load(res, path):
    with open(os.path.join(res, path)) as fp:
        return list(csv.DictReader(fp))


def stats(rows, key):
    tm, ts, um, us = [], [], [], []
    for pm in ORDER:
        r = [x for x in rows if x["perturbation"] == pm]
        seeds = sorted(int(x["seed"]) for x in r)
        if set(seeds) != EXPECTED_SEEDS or len(seeds) != len(EXPECTED_SEEDS):
            sys.exit(f"{pm} 种子集合 {seeds} != {sorted(EXPECTED_SEEDS)}，"
                     f"拒绝绘图")
        t = [float(x["task_bca"]) for x in r]
        u = [float(x[key]) for x in r]
        tm.append(np.mean(t) * 100); ts.append(np.std(t) * 100)
        um.append(np.mean(u) * 100); us.append(np.std(u) * 100)
    return tm, ts, um, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=None, help="结果目录（默认自动发现最新）")
    args = ap.parse_args()
    res = args.res or find_res_dir()
    print(f"结果目录: {res}")
    rows = load(res, "perturbation_results.csv")
    tm, ts, um, us = stats(rows, "uid_bca")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ORDER)); w = 0.36
    ax.bar(x - w / 2, tm, w, yerr=ts, capsize=3,
           label="Motor imagery task BCA", color="#4C72B0")
    ax.bar(x + w / 2, um, w, yerr=us, capsize=3,
           label="User identity (UID) BCA", color="#DD8452")
    ax.axhline(2.78, ls="--", c="gray", lw=1)
    ax.text(len(ORDER) - 0.65, 5, "chance 2.78%", fontsize=9, color="gray")
    ax.set_xticks(x); ax.set_xticklabels([NAMES[p] for p in ORDER])
    ax.set_ylabel("Balanced accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("User-wise perturbations on eegmmidb (36 subjects, L/R-hand MI, mean±SD over 5 seeds)")
    ax.legend()
    for xi, (t, u) in enumerate(zip(tm, um)):
        ax.text(xi - w / 2, t + ts[xi] + 2, f"{t:.1f}", ha="center",
                fontsize=9)
        ax.text(xi + w / 2, u + us[xi] + 2, f"{u:.1f}", ha="center",
                fontsize=9)
    out = os.path.join(res, "fig_perturbations.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
