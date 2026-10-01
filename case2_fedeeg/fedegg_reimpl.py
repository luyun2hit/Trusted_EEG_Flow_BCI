# -*- coding: utf-8 -*-
"""
FedEEG 的 PyTorch 忠实重实现（依据 FedEEG-main 源码逐条对应）
- 数据：仓库自带 dataset_hand_movement/user_a-d.csv（2880 样本/用户，112 维带功率特征，3 类平衡）
- 预处理：特征补零至 121 维 -> 各客户端用本地训练集均值/标准差标准化 -> reshape (11, 11)
- 模型：LSTM(11->16, return_seq) -> LSTM(16->16) -> BatchNorm1d -> Linear(16,3) softmax
- 客户端：80:20 划分，Adam(lr=1e-3)，批大小 12，每轮本地 3 epoch，交叉熵
- 聚合：5 轮，各客户端权重简单平均（FedAvg）；每轮聚合后以全局模型在各客户端测试集上评估
- 对照：集中式基线（四用户合并训练/测试，同模型同超参）
输出：结果 CSV、收敛图 PNG、汇总打印
"""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FEDEEG_DATA") or os.path.join(
    HERE, '..', 'FedEEG-main', 'dataset_hand_movement')
OUT = os.path.join(HERE, 'results')
os.makedirs(OUT, exist_ok=True)

CLIENTS = ['user_a', 'user_b', 'user_c', 'user_d']
N_ROUNDS = 5
LOCAL_EPOCHS = 3
BATCH = 12
LR = 1e-3
SEEDS = [42, 7, 2024]

if not os.path.isdir(DATA) or not all(
        os.path.exists(os.path.join(DATA, c + '.csv')) for c in CLIENTS):
    raise SystemExit(
        f"未找到 FedEEG 数据目录：{DATA}\n"
        "请克隆 github.com/AmanPriyanshu/FedEEG 并设置环境变量 "
        "FEDEEG_DATA 指向其 dataset_hand_movement/ 目录后重试。"
        "（数据随原仓库分发，本代码包不再分发，详见 README。）")

class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=11, hidden_size=16, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=16, batch_first=True)
        self.bn = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16, 3)
    def forward(self, x):
        h, _ = self.lstm1(x)
        h, _ = self.lstm2(h)
        h = h[:, -1, :]
        return self.fc(self.bn(h))

def preprocessing(path, seed=42):
    """与 dataloader.py 一致：补零到 121 维、one-hot、按种子打乱"""
    df = pd.read_csv(path, delimiter=',', index_col=False)
    X = df.iloc[:, 1:].copy()
    Y = df.iloc[:, 0].values.astype(int)
    for i in range(121 - X.shape[1]):
        X[f'complement{i}'] = 0
    X = X.values.astype(np.float64)
    idx = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(idx)
    return X[idx], Y[idx]

def client_data(client, seed):
    X, Y = preprocessing(os.path.join(DATA, client + '.csv'), seed)
    n = int(0.8 * len(X))
    return X[:n], Y[:n], X[n:], Y[n:]

def standardize(X, mean, std):
    return ((X - mean) / (std + 1e-5)).reshape(-1, 11, 11).astype(np.float32)

def train_local(model, xtr, ytr, epochs=LOCAL_EPOCHS):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    X = torch.from_numpy(xtr); Y = torch.from_numpy(ytr).long()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            opt.zero_grad()
            loss = ce(model(X[perm[i:i+BATCH]]), Y[perm[i:i+BATCH]])
            loss.backward(); opt.step()

@torch.no_grad()
def evaluate(model, xte, yte):
    model.eval()
    logits = model(torch.from_numpy(xte))
    loss = nn.CrossEntropyLoss()(logits, torch.from_numpy(yte).long()).item()
    acc = (logits.argmax(1).numpy() == yte).mean()
    return loss, acc

def fedavg(states):
    avg = {}
    for k in states[0]:
        avg[k] = torch.stack([s[k].float() for s in states]).mean(0)
    return avg

def run_seed(seed):
    # 各客户端仅用本地训练集计算标准化参数（修正：不再使用 user_a 全量数据的
    # 全局统计量——原协议中 user_a 的测试数据参与了标准化参数计算，构成预处理
    # 层面的测试信息泄漏，且与“原始数据不出本地”的隐私设定冲突）
    data = {c: client_data(c, 42) for c in CLIENTS}
    data = {c: (standardize(v[0], v[0].mean(0), v[0].std(0)), v[1],
                standardize(v[2], v[0].mean(0), v[0].std(0)), v[3])
            for c, v in data.items()}

    torch.manual_seed(seed)
    model = LSTMClassifier()
    history = []  # 每轮 [(loss, acc) x 4 clients]，为聚合后全局模型在各客户端测试集上的表现
    for rnd in range(N_ROUNDS):
        states = []
        for c in CLIENTS:
            xtr, ytr, xte, yte = data[c]
            local = LSTMClassifier()
            local.load_state_dict(model.state_dict())
            train_local(local, xtr, ytr)
            states.append(local.state_dict())
        model.load_state_dict(fedavg(states))
        # 修正：先聚合、后评估——用聚合后的全局模型在各客户端测试集上测试，
        # 而非（原协议的）聚合前逐一测试本地模型
        perf = [evaluate(model, data[c][2], data[c][3]) for c in CLIENTS]
        history.append(perf)
        print(f'  seed {seed} round {rnd+1}: ' +
              ' '.join(f'{c}={a:.4f}' for c, (_, a) in zip(CLIENTS, perf)), flush=True)
    fed_final = float(np.mean([a for _, a in history[-1]]))

    # 集中式基线：合并四用户数据训练同一模型
    torch.manual_seed(seed)
    base = LSTMClassifier()
    Xtr = np.concatenate([data[c][0] for c in CLIENTS]); ytr = np.concatenate([data[c][1] for c in CLIENTS])
    Xte = np.concatenate([data[c][2] for c in CLIENTS]); yte = np.concatenate([data[c][3] for c in CLIENTS])
    train_local(base, Xtr, ytr, epochs=N_ROUNDS * LOCAL_EPOCHS)  # 等累计训练量
    bl_loss, bl_acc = evaluate(base, Xte, yte)
    print(f'  seed {seed}: federated(final-round mean)={fed_final:.4f} centralized={bl_acc:.4f}', flush=True)
    return history, fed_final, bl_acc

if __name__ == '__main__':
    import sys
    seeds = [int(a) for a in sys.argv[1:]] or SEEDS
    for s in seeds:
        h, f, b = run_seed(s)
        np.save(os.path.join(OUT, f'hist_{s}.npy'), np.array(h))
        json.dump({'seed': s, 'fed': f, 'base': b},
                  open(os.path.join(OUT, f'partial_{s}.json'), 'w'))
        print(f'seed {s} done: fed={f:.4f} base={b:.4f}', flush=True)
