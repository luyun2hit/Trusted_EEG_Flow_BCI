# -*- coding: utf-8 -*-
"""案例二训练配置 sweep：在相同模型与数据协议下，仅改变联邦轮数 /
本地 epoch / 学习率，筛选欠训练问题的修复配置。
用法：python sweep.py <rounds> <epochs> <lr> <seed> <tag>
输出：results/sweep.jsonl 逐行追加。
"""
import os, sys, json, time
import numpy as np
import torch

import fedegg_reimpl as M


def load_data():
    data = {c: M.client_data(c, 42) for c in M.CLIENTS}
    return {c: (M.standardize(v[0], v[0].mean(0), v[0].std(0)), v[1],
                M.standardize(v[2], v[0].mean(0), v[0].std(0)), v[3])
            for c, v in data.items()}


def run_fed(rounds, epochs, lr, seed):
    M.LR = lr  # train_local 读取模块级 LR
    data = load_data()
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    hist = []
    for rnd in range(rounds):
        states = []
        for c in M.CLIENTS:
            local = M.LSTMClassifier()
            local.load_state_dict(model.state_dict())
            M.train_local(local, data[c][0], data[c][1], epochs=epochs)
            states.append(local.state_dict())
        model.load_state_dict(M.fedavg(states))
        perf = [M.evaluate(model, data[c][2], data[c][3]) for c in M.CLIENTS]
        hist.append(float(np.mean([a for _, a in perf])))
        print(f'  round {rnd+1}: mean_acc={hist[-1]:.4f}', flush=True)
    return hist


if __name__ == '__main__':
    rounds, epochs, lr, seed, tag = (int(sys.argv[1]), int(sys.argv[2]),
                                     float(sys.argv[3]), int(sys.argv[4]),
                                     sys.argv[5])
    t0 = time.time()
    hist = run_fed(rounds, epochs, lr, seed)
    rec = dict(tag=tag, rounds=rounds, epochs=epochs, lr=lr, seed=seed,
               hist=[round(h, 4) for h in hist], final=round(hist[-1], 4),
               best=round(max(hist), 4), elapsed=round(time.time() - t0, 1))
    out = os.path.join(M.OUT, 'sweep.jsonl')
    with open(out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('SWEEP-REC ' + json.dumps(rec, ensure_ascii=False), flush=True)
