# -*- coding: utf-8 -*-
"""时域/非线性特征的 TSTR 复核（评审建议：打破"谱损失训练—谱特征评价"的指标耦合）。
协议与 evaluate_chbmit.tstr_experiment 完全一致（时间隔离、同一随机数序列、
同一 RF 配置），仅将 Welch 频带功率特征替换为纯时域/非线性特征：
每通道 均值、标准差、偏度、峰度、Hjorth 活动性/移动性/复杂度、
平均绝对一阶差分、过零率、排列熵（阶 3）——每通道 10 个特征，合计 20 维。
用法: GAN_DIR=gan_results_timeiso python tstr_td_features.py
输出: results/tstr_td.json
v3（2026-09-30）：降采样改用抗混叠 ev.ds()；Welch 参照值不再硬编码，
改为读取 results/summary_timeiso.json（不存在则省略）。
"""
import os, json, math
import numpy as np

# v5 修复：本脚本必须与时间隔离版合成数据配套（参照 summary_timeiso.json），
# 未显式指定 timeiso 目录时拒绝运行，避免误读非隔离版数据却对照隔离版摘要
_gan_dir = os.environ.get('GAN_DIR', '')
assert 'timeiso' in os.path.basename(_gan_dir.rstrip('/\\')), \
    '请先设置 GAN_DIR 指向时间隔离版目录，例如: ' \
    'GAN_DIR=gan_results_timeiso python tstr_td_features.py'

import evaluate_chbmit as ev  # 复用数据加载与常量；GAN_DIR 由环境变量控制

N_TRAIN, N_REP = ev.N_TRAIN, ev.N_REP
OUT = os.path.join(ev.RES_DIR, 'tstr_td.json')


def _perm_entropy(x, order=3):
    """排列熵（阶 order，时延 1），返回归一化值 [0,1]"""
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(x, order)
    pat = np.argsort(np.argsort(w, axis=1), axis=1)
    # 映射到排列编号
    codes = np.zeros(len(w), np.int64)
    for k in range(order):
        codes = codes * order + pat[:, k]
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    ent = -(p * np.log2(p)).sum()
    return ent / np.log2(math.factorial(order))


def td_features(X):
    """X: (N, L) 拼接窗（自动按长度拆两通道），每通道 10 维时域/非线性特征 => 20 维"""
    from scipy.stats import skew, kurtosis
    n = X.shape[0]
    half = X.shape[1] // 2
    Xc = X.reshape(n, 2, half).astype(np.float64)
    feats = np.zeros((n, 20), np.float32)
    for i in range(n):
        row = []
        for c in range(2):
            x = Xc[i, c]
            dx = np.diff(x)
            ddx = np.diff(dx)
            v0, v1, v2 = x.var(), dx.var(), ddx.var()
            mobility = np.sqrt(v1 / max(v0, 1e-12))
            complexity = np.sqrt(v2 / max(v1, 1e-12)) / max(mobility, 1e-12)
            row += [x.mean(), x.std(), float(skew(x)), float(kurtosis(x)),
                    v0, float(mobility), float(complexity),
                    float(np.abs(dx).mean()),               # 平均绝对一阶差分
                    float((np.diff(np.signbit(x))).mean()),  # 过零率
                    float(_perm_entropy(x))]
        feats[i] = row
    return feats


def run():
    from sklearn.ensemble import RandomForestClassifier
    rows = []
    for sid in ev.SUBJECTS:
        inter_noover_all = ev.ds(ev.load(sid, 'inter_noover'))
        half = len(inter_noover_all) // 2
        inter_tr, inter_te = inter_noover_all[:half], inter_noover_all[half:]
        ictal_real = ev.ds(ev.load(sid, 'ictal')[::4])
        synth = ev.load_synth(sid)
        assert len(synth) >= N_TRAIN, \
            f'{sid} 合成样本不足: {len(synth)} < {N_TRAIN}（疑似读错 GAN_DIR）'
        others = np.concatenate(
            [ev.load(s, 'ictal') for s in ev.ALL_SUBJECTS if s != sid])
        others = ev.ds(others)
        n_test_i = len(ictal_real)
        n_test_n = min(2 * n_test_i, len(inter_te))
        test_X = np.concatenate([ictal_real, inter_te[:n_test_n]])
        test_y = np.array([1] * n_test_i + [0] * n_test_n)
        ftest = td_features(test_X)
        g_t, g_b = [], []
        for rep in range(N_REP):
            r = np.random.default_rng(rep)
            si = r.integers(0, len(synth), min(N_TRAIN, len(synth)))
            ii = r.integers(0, len(inter_tr), N_TRAIN)
            yi = np.array([1] * len(si) + [0] * N_TRAIN)
            fTi = td_features(np.concatenate([synth[si], inter_tr[ii]]))
            rf = RandomForestClassifier(n_estimators=200, n_jobs=8, random_state=rep)
            rf.fit(fTi, yi)
            g_t.append(ev.gmean_sens_spec(test_y, rf.predict(ftest)))
            bi = r.integers(0, len(others), N_TRAIN)
            fBi = td_features(np.concatenate([others[bi], inter_tr[ii]]))
            rf2 = RandomForestClassifier(n_estimators=200, n_jobs=8, random_state=rep)
            rf2.fit(fBi, yi)
            g_b.append(ev.gmean_sens_spec(test_y, rf2.predict(ftest)))
        gt = np.mean([x[0] for x in g_t]) * 100
        gb = np.mean([x[0] for x in g_b]) * 100
        rows.append(dict(patient=sid, baseline=round(gb, 2), synthetic=round(gt, 2),
                         diff=round(gt - gb, 2),
                         sens=round(np.mean([x[1] for x in g_t]), 3),
                         spec=round(np.mean([x[2] for x in g_t]), 3)))
        print(f'[TSTR-TD] {sid}: baseline={gb:.2f}% synthetic={gt:.2f}% '
              f'diff={gt-gb:+.2f}%', flush=True)
    mean_t = float(np.mean([r['synthetic'] for r in rows]))
    mean_b = float(np.mean([r['baseline'] for r in rows]))
    out = dict(features='时域/非线性特征：每通道 10 个（均值/标准差/偏度/峰度/方差/'
                        'Hjorth×2/平均绝对一阶差分/过零率/排列熵），合计 20 维',
               protocol='与 summary_timeiso.json 相同的时间隔离 TSTR 协议',
               rows=rows, mean_synthetic=round(mean_t, 2), mean_baseline=round(mean_b, 2),
               mean_diff=round(mean_t - mean_b, 2))
    # v3：Welch 参照值从 summary_timeiso.json 读取，不再硬编码
    ref_path = os.path.join(ev.RES_DIR, 'summary_timeiso.json')
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path, encoding='utf-8'))
        out['reference_welch'] = {'mean_synthetic': ref.get('mean_synthetic'),
                                  'mean_baseline': ref.get('mean_baseline')}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('saved', OUT)
    print(f"时域特征 TSTR: 合成={mean_t:.2f}% 基线={mean_b:.2f}% "
          f"差值={mean_t-mean_b:+.2f}%")


if __name__ == '__main__':
    run()
