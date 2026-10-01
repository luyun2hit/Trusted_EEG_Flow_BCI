# -*- coding: utf-8 -*-
"""案例二调参后正式运行：配置 B（24 轮 × 3 本地 epoch，lr=1e-3，其余与原协议一致）。
用法：python final_run.py <arm> <seed>
  arm = fed         联邦 24 轮，逐轮聚合后全局模型在各客户端测试集评估
  arm = centralized 集中式基线（72 epoch，与联邦累计训练量对齐）
  arm = local       本地单训基线（各客户端独立 72 epoch，无聚合）
  arm = mia         成员推断抽查（24 轮逐轮 AUC + 过拟合间隙 + 集中式对照）
输出：results/final_<arm>_<seed>.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

import fedegg_reimpl as M

ROUNDS, EPOCHS, LR = 24, 3, 1e-3


def load_data():
    data = {c: M.client_data(c, 42) for c in M.CLIENTS}
    return {c: (M.standardize(v[0], v[0].mean(0), v[0].std(0)), v[1],
                M.standardize(v[2], v[0].mean(0), v[0].std(0)), v[3])
            for c, v in data.items()}


def merged(data):
    Xtr = np.concatenate([data[c][0] for c in M.CLIENTS])
    ytr = np.concatenate([data[c][1] for c in M.CLIENTS])
    Xte = np.concatenate([data[c][2] for c in M.CLIENTS])
    yte = np.concatenate([data[c][3] for c in M.CLIENTS])
    return Xtr, ytr, Xte, yte


def arm_fed(seed):
    data = load_data()
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    hist = []
    for rnd in range(ROUNDS):
        states = []
        for c in M.CLIENTS:
            local = M.LSTMClassifier()
            local.load_state_dict(model.state_dict())
            M.train_local(local, data[c][0], data[c][1], epochs=EPOCHS)
            states.append(local.state_dict())
        model.load_state_dict(M.fedavg(states))
        perf = [M.evaluate(model, data[c][2], data[c][3]) for c in M.CLIENTS]
        hist.append(dict(round=rnd + 1,
                         per_client={c: round(a, 4) for c, (_, a) in zip(M.CLIENTS, perf)},
                         mean=round(float(np.mean([a for _, a in perf])), 4)))
        print(f'  round {rnd+1}: mean={hist[-1]["mean"]:.4f}', flush=True)
    return dict(arm='fed', seed=seed, rounds=ROUNDS, epochs=EPOCHS, lr=LR,
                hist=hist, final_mean=hist[-1]['mean'])


def arm_centralized(seed):
    data = load_data()
    Xtr, ytr, Xte, yte = merged(data)
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    M.train_local(model, Xtr, ytr, epochs=ROUNDS * EPOCHS)
    loss, acc = M.evaluate(model, Xte, yte)
    print(f'  centralized acc={acc:.4f}', flush=True)
    return dict(arm='centralized', seed=seed, epochs=ROUNDS * EPOCHS, lr=LR,
                acc=round(float(acc), 4), loss=round(float(loss), 4))


def arm_local(seed):
    data = load_data()
    accs = {}
    for c in M.CLIENTS:
        torch.manual_seed(seed)
        model = M.LSTMClassifier()
        M.train_local(model, data[c][0], data[c][1], epochs=ROUNDS * EPOCHS)
        _, acc = M.evaluate(model, data[c][2], data[c][3])
        accs[c] = round(float(acc), 4)
        print(f'  {c}: acc={acc:.4f}', flush=True)
    return dict(arm='local', seed=seed, epochs=ROUNDS * EPOCHS, lr=LR,
                per_client=accs, mean=round(float(np.mean(list(accs.values()))), 4))


def per_sample_loss(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return nn.CrossEntropyLoss(reduction='none')(
            logits, torch.from_numpy(Y).long()).numpy()


def auc(ms, ns):
    s = np.concatenate([ms, ns])
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    rm = ranks[:len(ms)]
    return float((rm.sum() - len(ms) * (len(ms) + 1) / 2) / (len(ms) * len(ns)))


def tpr_at_fpr(ms, ns, fpr=0.1):
    thr = np.quantile(ns, 1 - fpr)
    return float((ms >= thr).mean())


def arm_mia(seed):
    data = load_data()
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    rounds = []
    for rnd in range(ROUNDS):
        states = []
        for c in M.CLIENTS:
            local = M.LSTMClassifier()
            local.load_state_dict(model.state_dict())
            M.train_local(local, data[c][0], data[c][1], epochs=EPOCHS)
            states.append(local.state_dict())
        model.load_state_dict(M.fedavg(states))
        ms, ns, tr_acc, te_acc = [], [], [], []
        for c in M.CLIENTS:
            xtr, ytr, xte, yte = data[c]
            lm = per_sample_loss(model, xtr, ytr)
            ln = per_sample_loss(model, xte, yte)
            ms.append(-lm); ns.append(-ln)
            tr_acc.append(float((model(torch.from_numpy(xtr)).argmax(1).numpy() == ytr).mean()))
            te_acc.append(float((model(torch.from_numpy(xte)).argmax(1).numpy() == yte).mean()))
        m_, n_ = np.concatenate(ms), np.concatenate(ns)
        rounds.append(dict(round=rnd + 1, auc=round(auc(m_, n_), 4),
                           tpr_at_fpr10=round(tpr_at_fpr(m_, n_), 4),
                           train_acc=round(float(np.mean(tr_acc)), 4),
                           test_acc=round(float(np.mean(te_acc)), 4),
                           gap=round(float(np.mean(tr_acc) - np.mean(te_acc)), 4)))
        print(f'  round {rnd+1}: AUC={rounds[-1]["auc"]:.4f} gap={rounds[-1]["gap"]:+.4f}',
              flush=True)
    return dict(arm='mia', seed=seed, rounds=rounds)


def arm_miacent(seed):
    """集中式对照模型的成员推断（等累计训练量 72 epoch）。"""
    data = load_data()
    Xtr, ytr, Xte, yte = merged(data)
    torch.manual_seed(seed)
    base = M.LSTMClassifier()
    M.train_local(base, Xtr, ytr, epochs=ROUNDS * EPOCHS)
    lb_m = per_sample_loss(base, Xtr, ytr)
    lb_n = per_sample_loss(base, Xte, yte)
    cent = dict(auc=round(auc(-lb_m, -lb_n), 4),
                tpr_at_fpr10=round(tpr_at_fpr(-lb_m, -lb_n), 4))
    print(f'  centralized MIA AUC={cent["auc"]:.4f}', flush=True)
    return dict(arm='miacent', seed=seed, centralized=cent)


ARMS = dict(fed=arm_fed, centralized=arm_centralized, local=arm_local,
            mia=arm_mia, miacent=arm_miacent)

if __name__ == '__main__':
    M.LR = LR
    arm, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    rec = ARMS[arm](seed)
    rec['elapsed'] = round(time.time() - t0, 1)
    out = os.path.join(M.OUT, f'final_{arm}_{seed}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f'DONE {arm} seed={seed} elapsed={rec["elapsed"]}s -> {out}', flush=True)
