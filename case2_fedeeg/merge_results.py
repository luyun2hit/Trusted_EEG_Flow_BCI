# -*- coding: utf-8 -*-
"""合并各种子的部分结果，生成 federated_rounds.csv 与 summary.json"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
CLIENTS = ['user_a', 'user_b', 'user_c', 'user_d']
N_ROUNDS = 5

hists, feds, bases = [], [], []
for pf in sorted(glob.glob(os.path.join(OUT, 'partial_*.json'))):
    d = json.load(open(pf))
    s = d['seed']
    h = np.load(os.path.join(OUT, f'hist_{s}.npy'))
    hists.append(h); feds.append(d['fed']); bases.append(d['base'])
all_hist = np.array(hists)  # (seeds, rounds, clients, 2)

rows = []
for r in range(N_ROUNDS):
    row = {'round': r + 1}
    for ci, c in enumerate(CLIENTS):
        row[c + '_acc'] = round(float(all_hist[:, r, ci, 1].mean()), 4)
        row[c + '_loss'] = round(float(all_hist[:, r, ci, 0].mean()), 4)
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'federated_rounds.csv'), index=False)
summary = dict(federated_mean=round(float(np.mean(feds)), 4),
               federated_std=round(float(np.std(feds)), 4),
               centralized_mean=round(float(np.mean(bases)), 4),
               centralized_std=round(float(np.std(bases)), 4),
               rounds=rows)
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'rounds'}, ensure_ascii=False))
