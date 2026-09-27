# -*- coding: utf-8 -*-
"""Local-only 基线：各客户端仅用本地训练集独立训练（不做任何聚合），
在本地测试集上评估。用于回答"联邦协作是否优于各自为政的本地建模"。
训练量与联邦/集中式对齐：N_ROUNDS × LOCAL_EPOCHS = 15 epoch，同模型同超参。
输出：results/local_only_baseline.json
"""
import os, json
import numpy as np
import torch

from fedegg_reimpl import (CLIENTS, N_ROUNDS, LOCAL_EPOCHS, SEEDS, OUT,
                           client_data, standardize, LSTMClassifier,
                           train_local, evaluate)


def run_seed(seed):
    data = {c: client_data(c, 42) for c in CLIENTS}
    data = {c: (standardize(v[0], v[0].mean(0), v[0].std(0)), v[1],
                standardize(v[2], v[0].mean(0), v[0].std(0)), v[3])
            for c, v in data.items()}
    accs = {}
    for c in CLIENTS:
        xtr, ytr, xte, yte = data[c]
        torch.manual_seed(seed)
        model = LSTMClassifier()
        train_local(model, xtr, ytr, epochs=N_ROUNDS * LOCAL_EPOCHS)
        _, acc = evaluate(model, xte, yte)
        accs[c] = float(acc)
    mean_acc = float(np.mean(list(accs.values())))
    print(f'  seed {seed}: ' +
          ' '.join(f'{c}={a:.4f}' for c, a in accs.items()) +
          f'  mean={mean_acc:.4f}', flush=True)
    return accs, mean_acc


if __name__ == '__main__':
    import sys
    seeds = [int(a) for a in sys.argv[1:]] or SEEDS
    per_seed, means = {}, []
    for s in seeds:
        accs, m = run_seed(s)
        per_seed[str(s)] = accs
        means.append(m)
    out = {
        'note': '各客户端仅用本地训练集独立训练 15 epoch（与联邦累计训练量对齐），'
                '在本地测试集上评估；无聚合、无参数交换。',
        'seeds': seeds,
        'per_seed_client_acc': per_seed,
        'mean_per_seed': means,
        'overall_mean': float(np.mean(means)),
        'overall_std': float(np.std(means, ddof=1)),
        'reference': {'federated_final_round': 0.4615, 'centralized': 0.5333},
    }
    with open(os.path.join(OUT, 'local_only_baseline.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"local-only 总体: {out['overall_mean']:.4f} ± {out['overall_std']:.4f}")
