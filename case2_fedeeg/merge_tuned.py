# -*- coding: utf-8 -*-
"""合并调参后（24 轮 × 3 本地 epoch）正式结果：
- 归档原 5 轮配置文件（*_r5 后缀，仅首次执行）
- 生成新版 summary.json / federated_rounds.csv / membership_inference.json /
  local_only_baseline.json（文件名与论文附录 A 引用一致）
"""
import os, json, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')
SEEDS = [42, 7, 2024]

# ---- 归档原 5 轮结果 ----
for name in ['summary.json', 'federated_rounds.csv',
             'membership_inference.json', 'local_only_baseline.json']:
    src = os.path.join(OUT, name)
    dst = os.path.join(OUT, name.replace('.', '_r5.', 1)
                       if name.endswith('.json') else name.replace('.csv', '_r5.csv'))
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)
        print('archived:', name, '->', os.path.basename(dst))


def load(arm, seed):
    return json.load(open(os.path.join(OUT, f'final_{arm}_{seed}.json'),
                          encoding='utf-8'))


def stats(vals):
    return dict(mean=round(float(np.mean(vals)), 4),
                std=round(float(np.std(vals, ddof=1)), 4),
                per_seed={str(s): round(v, 4) for s, v in zip(SEEDS, vals)})

# ---- 联邦 ----
fed = [load('fed', s) for s in SEEDS]
fed_final = [r['final_mean'] for r in fed]
fed_stat = stats(fed_final)

# 逐轮逐客户端 3 种子均值 -> federated_rounds.csv
rounds = list(range(1, 25))
clients = ['user_a', 'user_b', 'user_c', 'user_d']
rows = []
for i, rnd in enumerate(rounds):
    row = {'round': rnd}
    for c in clients:
        row[c + '_acc'] = round(float(np.mean(
            [r['hist'][i]['per_client'][c] for r in fed])), 4)
    row['mean_acc'] = round(float(np.mean(
        [r['hist'][i]['mean'] for r in fed])), 4)
    rows.append(row)
import csv
with open(os.path.join(OUT, 'federated_rounds.csv'), 'w', newline='',
          encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['round'] + [c + '_acc' for c in clients]
                       + ['mean_acc'])
    w.writeheader()
    w.writerows(rows)
print('written federated_rounds.csv (24 rounds, 3-seed mean)')

# ---- 集中式 / 本地 ----
cent = [load('centralized', s)['acc'] for s in SEEDS]
cent_stat = stats(cent)
loc = [load('local', s) for s in SEEDS]
loc_stat = stats([r['mean'] for r in loc])

# ---- MIA ----
mia = [load('mia', s) for s in SEEDS]
mia_last = [r['rounds'][-1] for r in mia]
mia_fed_stat = stats([r['auc'] for r in mia_last])
mia_fed_tpr = stats([r['tpr_at_fpr10'] for r in mia_last])
gap_last = stats([r['gap'] for r in mia_last])
miacent = [load('miacent', s)['centralized'] for s in SEEDS]
mia_cent_stat = stats([c['auc'] for c in miacent])
mia_cent_tpr = stats([c['tpr_at_fpr10'] for c in miacent])

mia_out = dict(
    note=('调参后配置（24 轮 × 3 本地 epoch，lr=1e-3）的成员推断抽查：'
          '基于损失的阈值无关 AUC 口径，成员=各客户端训练样本，'
          '非成员=同客户端留出测试样本；3 随机种子。'
          '集中式对照为等累计训练量（72 epoch）合并训练模型。'
          '原 5 轮配置结果归档于 membership_inference_r5.json。'),
    seeds=SEEDS,
    federated_last_round=dict(auc=mia_fed_stat, tpr_at_fpr10=mia_fed_tpr,
                              train_test_gap=gap_last),
    centralized_baseline=dict(auc=mia_cent_stat, tpr_at_fpr10=mia_cent_tpr),
    per_round_auc_mean=[round(float(np.mean(
        [r['rounds'][i]['auc'] for r in mia])), 4) for i in range(24)],
)
json.dump(mia_out, open(os.path.join(OUT, 'membership_inference.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
print('written membership_inference.json')

loc_out = dict(
    note=('各客户端仅用本地训练集独立训练 72 epoch（与调参后联邦累计训练量对齐），'
          '在本地测试集上评估；无聚合、无参数交换。'
          '原 15 epoch 版本归档于 local_only_baseline_r5.json。'),
    seeds=SEEDS, epochs=72,
    per_seed_client_acc={str(s): r['per_client'] for s, r in zip(SEEDS, loc)},
    mean_per_seed=[r['mean'] for r in loc],
    overall_mean=loc_stat['mean'], overall_std=loc_stat['std'],
    reference={'federated_final_round': fed_stat['mean'],
               'centralized': cent_stat['mean']},
)
json.dump(loc_out, open(os.path.join(OUT, 'local_only_baseline.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=2)
print('written local_only_baseline.json')

summary = dict(
    config=('调参后正式配置：24 轮 × 3 本地 epoch，lr=1e-3，批 12，'
            'LSTM(11→16→16)+BN+Linear；仅聚合轮数由原协议 5 轮提高到 24 轮，'
            '其余与 fedegg_reimpl.py 一致。原 5 轮配置结果归档于 summary_r5.json。'),
    seeds=SEEDS,
    federated_final_round=fed_stat,
    centralized_72ep=cent_stat,
    local_only_72ep=loc_stat,
    centralized_mean=cent_stat['mean'],  # 供 make_fig.py 使用
    local_only_mean=loc_stat['mean'],
    gap_fed_vs_centralized=round(cent_stat['mean'] - fed_stat['mean'], 4),
    gap_fed_vs_local=round(loc_stat['mean'] - fed_stat['mean'], 4),
    mia_fed_auc=mia_fed_stat, mia_centralized_auc=mia_cent_stat,
    tuning_log=('sweep（seed 42，末轮均值）：5 轮 0.4579 → 15 轮 0.4870 → '
                '24 轮 0.5113 → 30 轮 0.5104（平台期）；'
                '30×1ep 0.4883、15×3ep lr3e-3 0.4991、24×3ep lr3e-3 0.5065。'
                '选取 24 轮 × 3 epoch：仅改轮数、曲线单调上升且未过平台。'),
)
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w',
                      encoding='utf-8'), ensure_ascii=False, indent=2)
print('written summary.json')
print(json.dumps({k: summary[k] for k in
      ['federated_final_round', 'centralized_72ep', 'local_only_72ep',
       'gap_fed_vs_centralized', 'gap_fed_vs_local']},
      ensure_ascii=False, indent=1))
