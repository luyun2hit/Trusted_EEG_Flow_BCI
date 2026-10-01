# -*- coding: utf-8 -*-
"""FedEEG 成员推断抽查（对应正文 4.2 节隐私验收实验）。

威胁模型：好奇但诚实的中央聚合方（或任何获得全局模型的人），对聚合后的
全局模型拥有白盒访问权，试图推断某条记录是否属于某客户端的训练集
（成员推断攻击，MIA）。

方法：基于损失的阈值无关成员推断（Yeom et al., CSF 2018 的 AUC 口径）：
以"样本损失低 => 更可能是训练成员"为评分，对每轮聚合后的全局模型，
把四客户端的训练样本（成员）与测试样本（非成员，80:20 划分的留出部分）
混合计算 AUC 与低假阳率下的攻击优势（TPR@FPR=10%）。同时报告过拟合间隙
（训练/测试准确率差）与集中式基线模型的同一指标作对照。

AUC = 0.5 表示无可用泄漏；显著高于 0.5 表示全局模型携带成员信息。
训练配置与 fedegg_reimpl.py 完全一致（同种子下模型逐参数一致）。

用法：python membership_inference.py [seed ...]（默认 42 7 2024）
输出：results/mia_partial_<seed>.json；python mia_merge.py 合并为
results/membership_inference.json
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

from fedegg_reimpl import (LSTMClassifier, client_data, standardize,
                           train_local, fedavg, CLIENTS, N_ROUNDS, OUT)


def per_sample_loss(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return nn.CrossEntropyLoss(reduction='none')(
            logits, torch.from_numpy(Y).long()).numpy()


def auc(member_scores, nonmember_scores):
    """rank-based AUC：score 大 => 成员"""
    s = np.concatenate([member_scores, nonmember_scores])
    order = s.argsort()
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    rm = ranks[:len(member_scores)]
    n, m = len(member_scores), len(nonmember_scores)
    return float((rm.sum() - n * (n + 1) / 2) / (n * m))


def tpr_at_fpr(member_scores, nonmember_scores, fpr_target=0.1):
    """低假阳率下的攻击优势：阈值取非成员分数的 (1-fpr) 分位"""
    thr = np.quantile(nonmember_scores, 1 - fpr_target)
    return float((member_scores >= thr).mean())


def run_seed(seed):
    # 与 fedegg_reimpl.run_seed 完全相同的训练流程，逐轮增加 MIA 评估
    data = {c: client_data(c, 42) for c in CLIENTS}
    data = {c: (standardize(v[0], v[0].mean(0), v[0].std(0)), v[1],
                standardize(v[2], v[0].mean(0), v[0].std(0)), v[3])
            for c, v in data.items()}

    torch.manual_seed(seed)
    model = LSTMClassifier()
    rounds = []
    for rnd in range(N_ROUNDS):
        states = []
        for c in CLIENTS:
            local = LSTMClassifier()
            local.load_state_dict(model.state_dict())
            train_local(local, data[c][0], data[c][1])
            states.append(local.state_dict())
        model.load_state_dict(fedavg(states))
        # MIA：成员 = 各客户端训练样本，非成员 = 同一客户端的留出测试样本
        ms, ns = [], []
        tr_acc, te_acc = [], []
        per_client = {}
        for c in CLIENTS:
            xtr, ytr, xte, yte = data[c]
            lm = per_sample_loss(model, xtr, ytr)
            ln = per_sample_loss(model, xte, yte)
            ms.append(-lm); ns.append(-ln)
            per_client[c] = round(auc(-lm, -ln), 4)
            model.eval()
            with torch.no_grad():
                tr_acc.append(float((model(torch.from_numpy(xtr)).argmax(1).numpy() == ytr).mean()))
                te_acc.append(float((model(torch.from_numpy(xte)).argmax(1).numpy() == yte).mean()))
        m_, n_ = np.concatenate(ms), np.concatenate(ns)
        rounds.append(dict(
            round=rnd + 1, auc_pooled=round(auc(m_, n_), 4),
            tpr_at_fpr10=round(tpr_at_fpr(m_, n_), 4),
            train_acc=round(float(np.mean(tr_acc)), 4),
            test_acc=round(float(np.mean(te_acc)), 4),
            acc_gap=round(float(np.mean(tr_acc) - np.mean(te_acc)), 4),
            auc_per_client=per_client))
        print(f'  seed {seed} round {rnd+1}: MIA AUC={rounds[-1]["auc_pooled"]:.4f} '
              f'gap={rounds[-1]["acc_gap"]:+.4f}', flush=True)

    # 集中式基线对照（等累计训练量）
    torch.manual_seed(seed)
    base = LSTMClassifier()
    Xtr = np.concatenate([data[c][0] for c in CLIENTS]); ytr = np.concatenate([data[c][1] for c in CLIENTS])
    Xte = np.concatenate([data[c][2] for c in CLIENTS]); yte = np.concatenate([data[c][3] for c in CLIENTS])
    train_local(base, Xtr, ytr, epochs=N_ROUNDS * 3)
    lb_m = per_sample_loss(base, Xtr, ytr); lb_n = per_sample_loss(base, Xte, yte)
    base_auc = round(auc(-lb_m, -lb_n), 4)
    base_tpr = round(tpr_at_fpr(-lb_m, -lb_n), 4)
    print(f'  seed {seed}: centralized MIA AUC={base_auc:.4f}', flush=True)
    return dict(seed=seed, rounds=rounds,
                centralized=dict(auc=base_auc, tpr_at_fpr10=base_tpr))


if __name__ == '__main__':
    seeds = [int(a) for a in sys.argv[1:]] or [42, 7, 2024]
    for s in seeds:
        r = run_seed(s)
        json.dump(r, open(os.path.join(OUT, f'mia_partial_{s}.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'seed {s} done.', flush=True)
