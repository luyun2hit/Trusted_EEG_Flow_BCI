# -*- coding: utf-8 -*-
"""攻击器能力阳性对照（对应正文 4.3.3 节"攻击能力本身充足"的佐证实验）。

主协议（evaluate_chbmit.py）是跨状态的：间期训练 -> 发作测试，域偏移可能
压过身份指纹（Exp_orig 低于随机）。本脚本用同一攻击器（IDResNet，波形 +
db4 三级 DWT 输入，lr 5e-4 + 余弦退火，40 epoch，批 64，5 种子，cudnn
确定性）做两种**同状态**对照，验证攻击器在域偏移不存在时能力充足：

1) inter->inter：每被试非重叠间期窗（inter_noover，150 个）前 100 个训练、
   后 50 个测试（窗间不重叠，无训练/测试泄漏）；
2) ictal->ictal：每被试发作窗按 stride 4 去重叠后，前一半训练、后一半测试。

随机水平 = 1/4 = 25%。输出 results/attacker_power_control.json。
环境变量：CTRL_SEEDS(默认"0,1,2,3,4") CTRL_EPOCHS(默认40) DEVICE(默认auto)
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as Fn
import pywt

from evaluate_chbmit import SUBJECTS, FS, DS, load  # 复用主评估的数据接口

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, 'results')

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') \
    if os.environ.get('DEVICE', 'auto') != 'cpu' else 'cpu'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEEDS = [int(s) for s in os.environ.get('CTRL_SEEDS', '0,1,2,3,4').split(',')]
EPOCHS = int(os.environ.get('CTRL_EPOCHS', '40'))
N_USR = len(SUBJECTS)


def _dwt_concat(x1d):
    return np.concatenate(pywt.wavedec(x1d, 'db4', level=3))


def build_attack_input(X):
    """与 evaluate_chbmit.py 一致：原始波形 + DWT 系数拼接"""
    half = X.shape[1] // 2
    rows = []
    for i in range(X.shape[0]):
        c0, c1 = X[i, :half], X[i, half:]
        rows.append(np.concatenate([c0, c1, _dwt_concat(c0), _dwt_concat(c1)]))
    return np.stack(rows).astype(np.float32)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(cin)
        self.conv1 = nn.Conv1d(cin, cout, 5, stride, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 5, 1, 2, bias=False)
        self.skip = nn.Conv1d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        h = self.conv1(Fn.relu(self.bn1(x)))
        h = self.conv2(Fn.relu(self.bn2(h)))
        return h + self.skip(x)


class IDResNet(nn.Module):
    def __init__(self, n_users):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, 5, 2, 2, bias=False)
        self.blocks = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64, 2),
                                    ResBlock(64, 128, 2), ResBlock(128, 128, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, n_users)

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        h = self.pool(h).flatten(1)
        return self.fc2(Fn.relu(self.fc1(h)))


def train_attacker(X, y, seed):
    """与 evaluate_chbmit.py v2.2 相同配方"""
    torch.manual_seed(seed); np.random.seed(seed)
    m = IDResNet(N_USR).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X / 100.0).unsqueeze(1)
    yt = torch.from_numpy(y)
    g = torch.Generator(); g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                     batch_size=64, shuffle=True, generator=g)
    for ep in range(EPOCHS):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); ce(m(xb), yb).backward(); opt.step()
        sched.step()
    return m


def acc(m, X, y):
    m.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.from_numpy(X[i:i + 512] / 100.0).unsqueeze(1).to(DEVICE)
            preds.append(m(xb).argmax(1).cpu().numpy())
    return float((np.concatenate(preds) == y).mean()) * 100


def make_split(kind):
    """返回 (Xtr, ytr, Xte, yte)。kind: 'inter' 或 'ictal'。
    INTER_POOL_TRAIN=1 时，间期对照改用主协议的 stride-1 间期池（591/被试）
    作训练集——注意该池与 noover 测试窗来自相同间期段，存在近重复窗，
    仅用于复现早期宽松对照口径，不作为严格结论。"""
    Xtr, ytr, Xte, yte = [], [], [], []
    for k, sid in enumerate(SUBJECTS):
        if kind == 'inter':
            if os.environ.get('INTER_POOL_TRAIN', '0') == '1':
                tr = load(sid, 'inter')[:, ::DS]
                te = load(sid, 'inter_noover')[100:, ::DS]
            else:
                W = load(sid, 'inter_noover')[:, ::DS]      # 150 个非重叠窗
                tr, te = W[:100], W[100:]
        else:
            W = load(sid, 'ictal')[::4, ::DS]           # stride4 去重叠
            h = len(W) // 2
            tr, te = W[:h], W[h:]
        Xtr.append(tr); ytr += [k] * len(tr)
        Xte.append(te); yte += [k] * len(te)
    return np.concatenate(Xtr), np.array(ytr), np.concatenate(Xte), np.array(yte)


def augment(X, y, seed=123):
    """循环时移增强（与主评估一致：每窗复制一份，两通道分别随机平移）"""
    rng = np.random.default_rng(seed)
    half = X.shape[1] // 2
    aug = X.copy()
    for i in range(len(aug)):
        aug[i, :half] = np.roll(aug[i, :half], int(rng.integers(0, half)))
        aug[i, half:] = np.roll(aug[i, half:], int(rng.integers(0, half)))
    return np.concatenate([X, aug]), np.concatenate([y, y])


def run(kind):
    # 注意：train/test 按窗索引前后切分，两侧无重叠窗；
    # 训练侧可加循环时移增强（AUG=1），增强副本同样只来自训练侧
    Xtr, ytr, Xte, yte = make_split(kind)
    if os.environ.get('AUG', '0') == '1':
        Xtr, ytr = augment(Xtr, ytr)
    Xtr, Xte = build_attack_input(Xtr), build_attack_input(Xte)
    print(f'[CTRL {kind}] train {len(Xtr)} / test {len(Xte)}', flush=True)
    per_seed = []
    for s in SEEDS:
        m = train_attacker(Xtr, ytr, s)
        a = acc(m, Xte, yte)
        per_seed.append(dict(seed=s, acc=round(a, 2)))
        print(f'[CTRL {kind} seed{s}] acc={a:.2f}%', flush=True)
    v = np.array([p['acc'] for p in per_seed])
    return dict(mean=round(float(v.mean()), 2), std=round(float(v.std()), 2),
                ratio_to_chance=round(float(v.mean()) / (100.0 / N_USR), 2),
                n_seeds=len(SEEDS), epochs=EPOCHS, per_seed=per_seed)


if __name__ == '__main__':
    t0 = time.time()
    out = dict(
        note='同一攻击器（与 evaluate_chbmit.py v2.2 同配方）在同状态设置下的身份'
             '识别能力对照：证明主协议 Exp_orig 偏低源于跨状态域偏移而非攻击器不足',
        chance=round(100.0 / N_USR, 2),
        inter_to_inter=run('inter'),
        ictal_to_ictal=run('ictal'))
    json.dump(out, open(os.path.join(RES_DIR, 'attacker_power_control.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print('CTRL DONE in', round(time.time() - t0), 's')
    print(json.dumps(out, ensure_ascii=False, indent=1))
