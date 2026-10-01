# -*- coding: utf-8 -*-
"""download.py — eegmmidb 左/右手运动想象子集下载（36 被试 × R04/R08/R12）。

幂等续跑：已存在且可解析的 EDF 自动跳过；循环多轮直至全部完成或达到
MAX_PASSES；curl 非零退出码视为失败并删除残留文件。
"""
import os
import subprocess
import sys

SUBJECTS = [f"S{i:03d}" for i in range(1, 37)]
RUNS = ["R04", "R08", "R12"]  # 仅左/右手运动想象 run（语义一致）
BASE = "https://physionet.org/files/eegmmidb/1.0.0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
MAX_PASSES = 20


def edf_ok(path):
    """文件足够大且 EDF 头可解析、含注释，才算下载完成。"""
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        return False
    try:
        import pyedflib
        f = pyedflib.EdfReader(path)
        f.readAnnotations()
        f.close()
        return True
    except Exception:
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    total = len(SUBJECTS) * len(RUNS)
    for p in range(MAX_PASSES):
        todo = [(s, r) for s in SUBJECTS for r in RUNS
                if not edf_ok(os.path.join(OUT, f"{s}{r}.edf"))]
        print(f"pass {p + 1}: 剩余 {len(todo)}/{total}")
        if not todo:
            print("下载完成。")
            return
        # 本轮遍历全部缺失文件，每 12 个一批并行（v3 修复：旧版每轮
        # 只处理前 6 个，靠多轮碰运气补齐）
        for i in range(0, len(todo), 12):
            procs = []
            for s, r in todo[i:i + 12]:
                fn = f"{s}{r}.edf"
                procs.append((fn, subprocess.Popen(
                    ["curl", "-sS", "--fail", "--location", "--retry", "3",
                     "-C", "-", "-o", os.path.join(OUT, fn),
                     f"{BASE}/{s}/{fn}"])))
            for fn, proc in procs:
                rc = proc.wait()
                if rc != 0:
                    print(f"curl 失败 rc={rc}: {fn}")
                    try:
                        os.remove(os.path.join(OUT, fn))
                    except OSError:
                        pass
    done = sum(1 for s in SUBJECTS for r in RUNS
               if edf_ok(os.path.join(OUT, f"{s}{r}.edf")))
    print(f"达到最大轮数仍未完成：{done}/{total}")
    sys.exit(1)


if __name__ == "__main__":
    main()
