# -*- coding: utf-8 -*-
"""合并各种子的成员推断结果，生成 results/membership_inference.json"""
import os, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')

parts = []
for pf in sorted(glob.glob(os.path.join(OUT, 'mia_partial_*.json'))):
    parts.append(json.load(open(pf, encoding='utf-8')))
assert parts, '未找到 mia_partial_*.json，请先运行 membership_inference.py'

n_rounds = len(parts[0]['rounds'])
rounds = []
for r in range(n_rounds):
    aucs = [p['rounds'][r]['auc_pooled'] for p in parts]
    tprs = [p['rounds'][r]['tpr_at_fpr10'] for p in parts]
    gaps = [p['rounds'][r]['acc_gap'] for p in parts]
    rounds.append(dict(round=r + 1,
                       auc_mean=round(float(np.mean(aucs)), 4), auc_std=round(float(np.std(aucs)), 4),
                       tpr10_mean=round(float(np.mean(tprs)), 4),
                       gap_mean=round(float(np.mean(gaps)), 4)))
base_aucs = [p['centralized']['auc'] for p in parts]
summary = dict(
    note='基于损失的白盒成员推断（Yeom AUC 口径）：成员=客户端训练样本，'
         '非成员=同客户端留出测试样本；AUC=0.5 表示无可用泄漏。'
         '训练配置与 fedegg_reimpl.py 完全一致。',
    n_seeds=len(parts), seeds=[p['seed'] for p in parts],
    federated_rounds=rounds,
    centralized_auc_mean=round(float(np.mean(base_aucs)), 4),
    centralized_auc_std=round(float(np.std(base_aucs)), 4),
    final_round_per_client={c: round(float(np.mean(
        [p['rounds'][-1]['auc_per_client'][c] for p in parts])), 4)
        for c in parts[0]['rounds'][-1]['auc_per_client']})
json.dump(summary, open(os.path.join(OUT, 'membership_inference.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
