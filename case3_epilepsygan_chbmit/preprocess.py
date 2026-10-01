# -*- coding: utf-8 -*-
"""
CHB-MIT 预处理：分段 -> 4 秒窗样本
- ictal: 发作区间内 4s 窗、3s 重叠（与 EpilepsyGAN 原论文一致）
- interictal: 200s 间期块内 4s 窗（训练池 stride=1s；另存 stride=4s 的非重叠集合供测试）
输出: windows/{sid}_ictal.npy, {sid}_inter.npy, {sid}_inter_noover.npy  (N, 2048) float32
"""
import json, os, re, glob, sys
import numpy as np

FS = 256
WIN = 4 * FS          # 1024 per channel
HERE = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(HERE, 'segments')
OUT = os.path.join(HERE, 'windows')


def seizure_span(ann, sid, fn, t0):
    """返回段内发作的相对起止秒（段起点为 t0=s-4）"""
    for f, seiz in ann[sid]['files'].items():
        if f == fn:
            # 一个文件一个段；多发作文件取覆盖 t0 的那次
            for s, e in seiz:
                if abs((s - 4) - t0) < 1:
                    return 4, 4 + (e - s)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, 'annotations.json')) as fp:
        ann = json.load(fp)
    if not os.path.isdir(SEG) or not glob.glob(os.path.join(SEG, "*.npy")):
        sys.exit("未找到 segments/*.npy，请先运行 download_segments.py")
    for sid in ann:
        ictal_wins, inter_wins, inter_noover = [], [], []
        for path in sorted(glob.glob(os.path.join(SEG, f"{sid}_*.npy"))):
            base = os.path.basename(path)[:-4]
            m = re.match(r'(chb\d+)_((?:chb\d+_\d+))_(\d+)_(\d+)_(ictal|interictal)', base)
            _, fn, t0, t1, kind = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
            sig = np.load(path)  # (2, nsec*256)
            if kind == 'ictal':
                s0, s1 = seizure_span(ann, sid, fn + '.edf', t0)
                a, b = s0 * FS, s1 * FS
                starts = range(a, b - WIN + 1, FS)  # stride 1s (3s overlap)
                for st in starts:
                    w = sig[:, st:st + WIN]
                    if w.shape[1] == WIN:
                        ictal_wins.append(np.concatenate([w[0], w[1]]))
            else:
                for st in range(0, sig.shape[1] - WIN + 1, FS):  # stride 1s 池
                    w = sig[:, st:st + WIN]
                    inter_wins.append(np.concatenate([w[0], w[1]]))
                for st in range(0, sig.shape[1] - WIN + 1, WIN):  # 非重叠
                    w = sig[:, st:st + WIN]
                    inter_noover.append(np.concatenate([w[0], w[1]]))
        np.save(os.path.join(OUT, f"{sid}_ictal.npy"), np.array(ictal_wins, np.float32))
        np.save(os.path.join(OUT, f"{sid}_inter.npy"), np.array(inter_wins, np.float32))
        np.save(os.path.join(OUT, f"{sid}_inter_noover.npy"), np.array(inter_noover, np.float32))
        print(sid, 'ictal:', len(ictal_wins), 'interictal pool:', len(inter_wins), 'no-overlap:', len(inter_noover))
    print('saved to', OUT)


if __name__ == '__main__':
    main()
