# -*- coding: utf-8 -*-
"""案例二验证集协议重跑（修正“测试集参与轮数选择”的方法学缺陷）。

划分协议（每客户端 2880 样本，洗牌种子 42，与现行版本同一洗牌）：
  训练  X[:1728]   60%
  验证  X[1728:2304] 20%   —— 仅用于轮数/超参选择
  测试  X[2304:]    20%   —— 与现行 80:20 协议的测试集逐样本相同，只用于最终报告
标准化参数仅由训练部分（1728 样本）估计。

用法：python valsplit_run.py <arm> <seed> [budget]
  arm = sweep       联邦训练 30 轮，逐轮在验证集评估（测试集不参与），
                    增量写 results/valsplit_sweep_<seed>.jsonl（防超时丢失）
  arm = fed         正式联邦运行，budget=轮数 R*，逐轮记录验证/测试准确率
  arm = centralized 集中式基线，budget=epoch 数（R*×3，等累计训练量）
  arm = local       本地单训基线，budget=epoch 数（R*×3）
  arm = mia         成员推断抽查：成员=训练集，非成员=测试集（未参与选择）
  arm = miacent     集中式对照模型的成员推断
输出：results/valsplit_<arm>_<seed>.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

import fedegg_reimpl as M

TRAIN_END = 1728   # 60%
VAL_END = 2304     # 80%（与原协议训练集大小一致，测试集不变）
EPOCHS = 3
LR = 1e-3
SWEEP_ROUNDS = 30


def load_data3():
    """三划分数据：返回 {client: (xtr,ytr, xva,yva, xte,yte)}，标准化参数只用训练部分。"""
    data = {}
    for c in M.CLIENTS:
        Xf, Yf, Xte, Yte = M.client_data(c, 42)   # Xf = 前 2304（原“训练”部分）
        Xtr, Ytr = Xf[:TRAIN_END], Yf[:TRAIN_END]
        Xva, Yva = Xf[TRAIN_END:VAL_END], Yf[TRAIN_END:VAL_END]
        mean, std = Xtr.mean(0), Xtr.std(0)
        data[c] = (M.standardize(Xtr, mean, std), Ytr,
                   M.standardize(Xva, mean, std), Yva,
                   M.standardize(Xte, mean, std), Yte)
    return data


def fed_round(model, data):
    states = []
    for c in M.CLIENTS:
        local = M.LSTMClassifier()
        local.load_state_dict(model.state_dict())
        M.train_local(local, data[c][0], data[c][1], epochs=EPOCHS)
        states.append(local.state_dict())
    model.load_state_dict(M.fedavg(states))


def eval_mean(model, data, split):
    """split: 1=验证, 2=测试（data[c] 内偏移 2/4）"""
    off = 2 if split == 'val' else 4
    accs = [M.evaluate(model, data[c][off], data[c][off + 1])[1] for c in M.CLIENTS]
    return {c: round(float(a), 4) for c, a in zip(M.CLIENTS, accs)}, \
        round(float(np.mean(accs)), 4)


def arm_sweep(seed):
    data = load_data3()
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    out = os.path.join(M.OUT, f'valsplit_sweep_{seed}.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for rnd in range(SWEEP_ROUNDS):
            fed_round(model, data)
            _, va = eval_mean(model, data, 'val')
            rec = dict(seed=seed, round=rnd + 1, val_mean=va)
            f.write(json.dumps(rec) + '\n'); f.flush()
            print(f'  round {rnd+1}: val_mean={va:.4f}', flush=True)
    return None


def arm_fed(seed, rounds):
    data = load_data3()
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    hist = []
    for rnd in range(rounds):
        fed_round(model, data)
        pv, va = eval_mean(model, data, 'val')
        pt, ta = eval_mean(model, data, 'test')
        hist.append(dict(round=rnd + 1, val_mean=va, test_mean=ta,
                         val_per_client=pv, test_per_client=pt))
        print(f'  round {rnd+1}: val={va:.4f} test={ta:.4f}', flush=True)
    return dict(arm='fed', seed=seed, rounds=rounds, epochs=EPOCHS, lr=LR,
                split='60:20:20', hist=hist,
                final_val=hist[-1]['val_mean'], final_test=hist[-1]['test_mean'])


def merged(data, off):
    X = np.concatenate([data[c][off] for c in M.CLIENTS])
    Y = np.concatenate([data[c][off + 1] for c in M.CLIENTS])
    return X, Y


def arm_centralized(seed, epochs):
    data = load_data3()
    Xtr, ytr = merged(data, 0)
    Xte, yte = merged(data, 4)
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    M.train_local(model, Xtr, ytr, epochs=epochs)
    loss, acc = M.evaluate(model, Xte, yte)
    print(f'  centralized acc={acc:.4f}', flush=True)
    return dict(arm='centralized', seed=seed, epochs=epochs, lr=LR,
                split='60:20:20', acc=round(float(acc), 4),
                loss=round(float(loss), 4))


def arm_local(seed, epochs):
    data = load_data3()
    accs = {}
    for c in M.CLIENTS:
        torch.manual_seed(seed)
        model = M.LSTMClassifier()
        M.train_local(model, data[c][0], data[c][1], epochs=epochs)
        _, acc = M.evaluate(model, data[c][4], data[c][5])
        accs[c] = round(float(acc), 4)
        print(f'  {c}: acc={acc:.4f}', flush=True)
    return dict(arm='local', seed=seed, epochs=epochs, lr=LR, split='60:20:20',
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


def arm_mia(seed, rounds):
    data = load_data3()
    torch.manual_seed(seed)
    model = M.LSTMClassifier()
    recs = []
    for rnd in range(rounds):
        fed_round(model, data)
        ms, ns, tr_a, te_a = [], [], [], []
        for c in M.CLIENTS:
            xtr, ytr, xte, yte = data[c][0], data[c][1], data[c][4], data[c][5]
            ms.append(-per_sample_loss(model, xtr, ytr))
            ns.append(-per_sample_loss(model, xte, yte))
            tr_a.append(float((model(torch.from_numpy(xtr)).argmax(1).numpy() == ytr).mean()))
            te_a.append(float((model(torch.from_numpy(xte)).argmax(1).numpy() == yte).mean()))
        m_, n_ = np.concatenate(ms), np.concatenate(ns)
        recs.append(dict(round=rnd + 1, auc=round(auc(m_, n_), 4),
                         tpr_at_fpr10=round(tpr_at_fpr(m_, n_), 4),
                         train_acc=round(float(np.mean(tr_a)), 4),
                         test_acc=round(float(np.mean(te_a)), 4),
                         gap=round(float(np.mean(tr_a) - np.mean(te_a)), 4)))
        print(f'  round {rnd+1}: AUC={recs[-1]["auc"]:.4f} gap={recs[-1]["gap"]:+.4f}',
              flush=True)
    return dict(arm='mia', seed=seed, rounds=rounds, split='60:20:20',
                members='train', nonmembers='test', records=recs)


def arm_miacent(seed, epochs):
    data = load_data3()
    Xtr, ytr = merged(data, 0)
    Xte, yte = merged(data, 4)
    torch.manual_seed(seed)
    base = M.LSTMClassifier()
    M.train_local(base, Xtr, ytr, epochs=epochs)
    lb_m = per_sample_loss(base, Xtr, ytr)
    lb_n = per_sample_loss(base, Xte, yte)
    cent = dict(auc=round(auc(-lb_m, -lb_n), 4),
                tpr_at_fpr10=round(tpr_at_fpr(-lb_m, -lb_n), 4))
    print(f'  centralized MIA AUC={cent["auc"]:.4f}', flush=True)
    return dict(arm='miacent', seed=seed, epochs=epochs, split='60:20:20',
                centralized=cent)


ARMS = dict(sweep=arm_sweep, fed=arm_fed, centralized=arm_centralized,
            local=arm_local, mia=arm_mia, miacent=arm_miacent)

if __name__ == '__main__':
    M.LR = LR
    arm, seed = sys.argv[1], int(sys.argv[2])
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else None
    t0 = time.time()
    if arm == 'sweep':
        arm_sweep(seed)
        print(f'DONE sweep seed={seed} elapsed={round(time.time()-t0,1)}s', flush=True)
    else:
        rec = ARMS[arm](seed, budget)
        rec['elapsed'] = round(time.time() - t0, 1)
        out = os.path.join(M.OUT, f'valsplit_{arm}_{seed}.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        print(f'DONE {arm} seed={seed} elapsed={rec["elapsed"]}s -> {out}', flush=True)
