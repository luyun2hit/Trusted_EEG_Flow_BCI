# -*- coding: utf-8 -*-
"""图 2：验证集协议（60:20:20，R*=27）联邦曲线——测试集仅参与最终报告。
实线：各客户端逐轮验证准确率（3 种子均值）；粗虚线：验证集均值（轮数选择依据）；
菱形标记：R*=27 处各客户端测试准确率（测试集仅在所选轮次出现）；
竖虚线：R*=27；横线：集中式 / 本地单训等预算基线、随机水平。
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
for _f in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
CLIENTS = ['user_a', 'user_b', 'user_c', 'user_d']
SEEDS = [42, 7, 2024]

df = pd.read_csv(os.path.join(RES, 'federated_rounds.csv'))
summ = json.load(open(os.path.join(RES, 'summary.json'), encoding='utf-8'))
fed = {s: json.load(open(os.path.join(RES, f'valsplit_fed_{s}.json'), encoding='utf-8'))
       for s in SEEDS}
R = summ['selection']['R_star']
cent = summ['centralized']['mean']
loc = json.load(open(os.path.join(RES, 'local_only_baseline.json'), encoding='utf-8'))['mean']

per_client_val = {c: np.mean([[h['val_per_client'][c] for h in fed[s]['hist']]
                              for s in SEEDS], axis=0) for c in CLIENTS}
per_client_test_final = {c: float(np.mean([fed[s]['hist'][-1]['test_per_client'][c]
                                           for s in SEEDS])) for c in CLIENTS}

fig, ax = plt.subplots(figsize=(7, 4.2))
colors = {}
for c in CLIENTS:
    line, = ax.plot(df['round'], per_client_val[c], 'o-', lw=1.3, ms=3,
                    label=f'{c} (val)')
    colors[c] = line.get_color()
ax.plot(df['round'], df['val_mean'], 's--', color='tab:purple', lw=2.0, ms=4,
        label='validation mean (round selection)')
# 测试集仅在 R* 处出现
for c in CLIENTS:
    ax.plot([R], [per_client_test_final[c]], 'D', color=colors[c], ms=8,
            mec='k', mew=0.8, zorder=5)
ax.plot([], [], 'D', color='gray', ms=8, mec='k', mew=0.8,
        label='per-client test at R* (final report only)')
ax.axvline(R, ls=':', color='tab:purple', lw=1.2)
ax.annotate(f'R*={R}', xy=(R, 0.345), xytext=(R - 4.6, 0.352), fontsize=9,
            color='tab:purple')
ax.axhline(cent, ls='--', color='k', lw=1.2,
           label=f'centralized, matched budget ({cent:.3f})')
ax.axhline(loc, ls='-.', color='tab:red', lw=1.2,
           label=f'local-only, matched budget ({loc:.3f})')
ax.axhline(1/3, ls=':', color='gray', lw=1, label='chance (1/3)')
ax.set_xlabel('Federated round'); ax.set_ylabel('Accuracy')
ax.set_title('FedEEG (PyTorch re-implementation, 60:20:20 train/val/test):\n'
             'per-client validation accuracy per round, test reported only at R*'
             ' (mean over 3 seeds)')
ax.legend(fontsize=7.5, ncol=2); fig.tight_layout()
fig.savefig(os.path.join(RES, 'fig_fed_rounds.png'), dpi=300, bbox_inches='tight')
print('saved fig_fed_rounds.png')

out = dict(round=list(df['round']),
           val_mean=list(df['val_mean']),
           per_client_val={c: [round(float(v), 4) for v in per_client_val[c]]
                           for c in CLIENTS},
           per_client_test_at_R={c: round(v, 4)
                                 for c, v in per_client_test_final.items()},
           R_star=R, centralized_mean=cent, local_only_mean=loc,
           test_per_round_note='per-round test accuracies retained in '
                               'federated_rounds.csv for transparency; '
                               'not used for selection')
json.dump(out, open(os.path.join(RES, 'fig2_curve_data.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved fig2_curve_data.json')
