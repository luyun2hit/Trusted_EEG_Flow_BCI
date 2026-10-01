# -*- coding: utf-8 -*-
"""合并验证集协议（60:20:20，R*=27）正式实验结果，重新生成标准结果文件。

输入：results/valsplit_{sweep,fed,centralized,local,mia,miacent}_*.json(l)
输出（覆盖现行标准文件，旧的 24 轮调参版先备份为 *_tuned24）：
  summary.json / federated_rounds.csv / membership_inference.json /
  local_only_baseline.json / valsplit_summary.json（含筛查曲线与选择规则）
"""
import os, json, glob, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
SEEDS = [42, 7, 2024]
R_STAR = 27
EPOCHS = R_STAR * 3


def load(arm, seed):
    p = os.path.join(OUT, f'valsplit_{arm}_{seed}.json')
    return json.load(open(p, encoding='utf-8'))


def mean_std(xs):
    return round(float(np.mean(xs)), 4), round(float(np.std(xs, ddof=1)), 4)


# ---- 备份 24 轮调参版 ----
for name in ['summary.json', 'federated_rounds.csv', 'membership_inference.json',
             'local_only_baseline.json']:
    src = os.path.join(OUT, name)
    dst = os.path.join(OUT, name.replace('.json', '_tuned24.json')
                                 .replace('.csv', '_tuned24.csv'))
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)
        print(f'backup {name} -> {os.path.basename(dst)}')

# ---- 筛查曲线与选择规则 ----
sweep = [json.loads(l) for l in
         open(os.path.join(OUT, 'valsplit_sweep_42.jsonl'), encoding='utf-8')]
best = max(sweep, key=lambda r: r['val_mean'])
assert best['round'] == R_STAR, f"argmax 轮数 {best['round']} 与 R*={R_STAR} 不一致"
print(f"sweep: R*={R_STAR} val_mean={best['val_mean']} (argmax of 30 rounds, seed 42)")

# ---- 联邦 ----
fed = {s: load('fed', s) for s in SEEDS}
fed_test = [fed[s]['final_test'] for s in SEEDS]
fed_val = [fed[s]['final_val'] for s in SEEDS]
m, sd = mean_std(fed_test)
print(f"fed test@{R_STAR}: {fed_test} -> {m:.4f} ± {sd:.4f}")

# 逐轮表：每轮各种子 val/test 均值
rounds_rows = []
for r in range(R_STAR):
    row = dict(round=r + 1)
    for s in SEEDS:
        h = fed[s]['hist'][r]
        row[f'val_seed{s}'] = h['val_mean']
        row[f'test_seed{s}'] = h['test_mean']
    row['val_mean'] = round(float(np.mean([row[f'val_seed{s}'] for s in SEEDS])), 4)
    row['test_mean'] = round(float(np.mean([row[f'test_seed{s}'] for s in SEEDS])), 4)
    rounds_rows.append(row)

import csv
with open(os.path.join(OUT, 'federated_rounds.csv'), 'w', newline='',
          encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rounds_rows[0].keys()))
    w.writeheader(); w.writerows(rounds_rows)

# ---- 集中式 / 本地 ----
cent = [load('centralized', s)['acc'] for s in SEEDS]
mc, sdc = mean_std(cent)
print(f"centralized @{EPOCHS}ep: {cent} -> {mc:.4f} ± {sdc:.4f}")

loc_recs = {s: load('local', s) for s in SEEDS}
loc = [loc_recs[s]['mean'] for s in SEEDS]
ml, sdl = mean_std(loc)
print(f"local @{EPOCHS}ep: {loc} -> {ml:.4f} ± {sdl:.4f}")

# ---- MIA ----
mia = {s: load('mia', s) for s in SEEDS}
mia_final = {s: mia[s]['records'][-1] for s in SEEDS}
aucs = [mia_final[s]['auc'] for s in SEEDS]
tprs = [mia_final[s]['tpr_at_fpr10'] for s in SEEDS]
gaps = [mia_final[s]['gap'] for s in SEEDS]
ma, sda = mean_std(aucs); mt, sdt = mean_std(tprs); mg, sdg = mean_std(gaps)
print(f"MIA fed AUC@{R_STAR}: {aucs} -> {ma:.4f} ± {sda:.4f}")
print(f"MIA fed TPR@10%FPR: {tprs} -> {mt:.4f} ± {sdt:.4f}")
print(f"MIA fed gap: {gaps} -> {mg:+.4f} ± {sdg:.4f}")

cent_mia = [load('miacent', s)['centralized']['auc'] for s in SEEDS]
cent_tpr = [load('miacent', s)['centralized']['tpr_at_fpr10'] for s in SEEDS]
mca, sdca = mean_std(cent_mia); mct, sdct = mean_std(cent_tpr)
print(f"MIA centralized AUC: {cent_mia} -> {mca:.4f} ± {sdca:.4f}")
print(f"MIA centralized TPR@10%FPR: {cent_tpr} -> {mct:.4f} ± {sdct:.4f}")

# 末轮各客户端测试准确率范围（图 2 正文用）
per_client_last = {}
for c in ['user_a', 'user_b', 'user_c', 'user_d']:
    vals = [fed[s]['hist'][-1]['test_per_client'][c] for s in SEEDS]
    per_client_last[c] = round(float(np.mean(vals)), 4)
lo = min(per_client_last.values()); hi = max(per_client_last.values())
print(f"末轮各客户端（3种子均值）: {per_client_last} 范围 {lo:.4f}–{hi:.4f}")

# ---- 写标准文件 ----
summary = dict(
    protocol='60:20:20 train/val/test per client (shuffle seed 42); '
             'test split identical to previous 80:20 protocol; '
             'standardization fit on train part only',
    selection=dict(rule='R* = argmax validation accuracy over 30 rounds, seed 42; '
                        'test set not used for selection',
                   sweep_file='valsplit_sweep_42.jsonl',
                   R_star=R_STAR, R_star_val=best['val_mean']),
    config=dict(rounds=R_STAR, local_epochs=3, lr=1e-3, batch=12),
    federated=dict(per_seed=dict(zip(map(str, SEEDS), fed_test)),
                   val_per_seed=dict(zip(map(str, SEEDS), fed_val)),
                   mean=m, std=sd,
                   final_round_per_client=per_client_last),
    centralized=dict(epochs=EPOCHS, per_seed=dict(zip(map(str, SEEDS), cent)),
                     mean=mc, std=sdc),
    note='numbers supersede tuned24 (80:20, 24 rounds) versions',
)
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

json.dump(dict(
    protocol=summary['protocol'], epochs=EPOCHS,
    per_seed={str(s): loc_recs[s]['per_client'] for s in SEEDS},
    mean_per_seed=dict(zip(map(str, SEEDS), loc)), mean=ml, std=sdl,
), open(os.path.join(OUT, 'local_only_baseline.json'), 'w', encoding='utf-8'),
    ensure_ascii=False, indent=1)

json.dump(dict(
    protocol='members=train split (1728/client), nonmembers=test split (576/client); '
             'test set not used for round selection; loss-based score (Yeom-style), '
             'threshold-free ROC-AUC + TPR@10%FPR',
    rounds=R_STAR,
    federated=dict(per_seed={str(s): mia[s]['records'] for s in SEEDS},
                   final_auc=dict(zip(map(str, SEEDS), aucs)),
                   final_tpr_at_fpr10=dict(zip(map(str, SEEDS), tprs)),
                   final_gap=dict(zip(map(str, SEEDS), gaps)),
                   auc_mean=ma, auc_std=sda, tpr_mean=mt, tpr_std=sdt,
                   gap_mean=mg, gap_std=sdg),
    centralized=dict(per_seed_auc=dict(zip(map(str, SEEDS), cent_mia)),
                     per_seed_tpr=dict(zip(map(str, SEEDS), cent_tpr)),
                     auc_mean=mca, auc_std=sdca, tpr_mean=mct, tpr_std=sdct),
), open(os.path.join(OUT, 'membership_inference.json'), 'w', encoding='utf-8'),
    ensure_ascii=False, indent=1)

json.dump(dict(summary=summary, local=dict(mean=ml, std=sdl),
               mia_fed=dict(auc=ma, tpr=mt, gap=mg),
               mia_centralized=dict(auc=mca, tpr=mct)),
          open(os.path.join(OUT, 'valsplit_summary.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('ALL FILES WRITTEN')
