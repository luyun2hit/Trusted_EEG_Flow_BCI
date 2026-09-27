# -*- coding: utf-8 -*-
"""
EpilepsyGAN 的 CHB-MIT 教学化重实现（PyTorch，CPU/GPU 自动检测）
忠实保留原论文核心设计：
- 条件 GAN：以发作间期(inter-ictal)窗为条件，生成发作期(ictal)窗
- U-Net 生成器（加权跳跃连接）+ LSGAN 损失 + L1 参照正则（λ=10，原论文 100，下调原因见章节"陷阱二"）
- Adam(beta1=0, beta2=0.9), lr_G=2e-4, lr_D=4e-4
- 留一被试(leave-one-out)训练，为未见被试生成合成发作数据
教学化简化（与原论文差异在章节中声明）：
- 4 名被试(chb01/03/05/08)、双通道 F7-T7/F8-T8、4s 窗(2048 点拼接)
- 网络宽度缩减(32/64/64/128, 4 块)、batch=32、epochs 可调
- 信号按 1/100 缩放(µV->约±3)以适配 tanh 输出头
v2 性能改进（2026-09，相对 v1 的三处训练侧升级）：
- MAX_PAIRS=0 默认使用训练折全部发作窗（v1 固定上限 800，数据利用率不足）；
- 每个 epoch 重新随机配对条件窗/参照窗（v1 全程固定配对，L1 中位数回归压力
  集中在固定参照上；逐轮重配使其在epoch间平均，并扩大条件组合覆盖）；
- 新增平移不变的谱损失项 LAMBDA_SPEC * mean|log|rFFT(fake)| - log|rFFT(real)||
  （时域 L1 是逐点约束、对相位随机信号退化为中位数回归；谱幅度损失直接约束
  频带功率分布，与 TSTR 评估的 Welch 带功率特征对齐，且不引入逐点坍缩）。
v3 隐私修正（2026-09）：强化攻击器审计发现条件通道身份泄漏——合成发作窗携带
条件间期窗的患者指纹（间期训练攻击器：合成 45.3% vs 真实发作 15.6%，N=4）。
新增身份对抗去偏：UID 头经梯度反转层读取生成窗的条件患者标签（训练折标签
已知），生成器被迫抹去该身份特征（LAMBDA_ADV=2.0，设 0 可关闭回退 v2）。
环境变量：EPOCHS(默认40) MAX_PAIRS(默认0=全部) LAMBDA_L1(默认10)
          LAMBDA_SPEC(默认1.0) LAMBDA_ADV(默认2.0) DEVICE(默认auto，可设 cpu 强制)
          ONLY(单折调试)
"""
import os, time, json
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') \
    if os.environ.get('DEVICE', 'auto') != 'cpu' else 'cpu'
if DEVICE == 'cpu':
    torch.set_num_threads(16)

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_DIR = os.path.join(HERE, 'windows')
OUT_DIR = os.path.join(HERE, 'gan_results')
os.makedirs(OUT_DIR, exist_ok=True)

SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb08']
SCALE = 100.0
DS = 2                      # 降采样因子（256Hz->128Hz，窗仍为4秒）
LENGTH = 2048 // DS         # 1024
EPOCHS = int(os.environ.get('EPOCHS', '40'))
BATCH = 32
MAX_PAIRS = int(os.environ.get('MAX_PAIRS', '0'))  # 0 = 使用全部发作窗
LAMBDA_L1 = float(os.environ.get('LAMBDA_L1', '10'))
LAMBDA_SPEC = float(os.environ.get('LAMBDA_SPEC', '1.0'))
LAMBDA_ADV = float(os.environ.get('LAMBDA_ADV', '2.0'))  # 身份对抗去偏权重（v3）
N_GEN = 500  # 每个留一被试生成的合成发作样本数

class GradReverse(torch.autograd.Function):
    """梯度反转层（GRL）：前向恒等，反向把梯度乘以 -λ。
    用于身份对抗去偏：UID 头正常最小化患者交叉熵，生成器收到反向梯度，
    被迫从输出中抹去条件窗携带的患者身份特征。"""
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None

class UIDHead(nn.Module):
    """患者身份分类头：读取生成窗中的条件患者身份（供 GRL 对抗去除）。
    标签为训练折被试编号（留一法下训练折标签已知）。"""
    def __init__(self, n_train):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 15, 4, 7), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 15, 4, 7), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 15, 4, 7), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, n_train))
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, chans=(32, 64, 64, 128)):
        super().__init__()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.skips = nn.ParameterList()  # 加权跳跃连接权重
        cin = 1
        for c in chans:
            self.convs.append(nn.Conv1d(cin, c, kernel_size=31, stride=1, padding=15, bias=False))
            self.pools.append(nn.MaxPool1d(2, 2))
            self.skips.append(nn.Parameter(torch.ones(1)))
            cin = c
    def forward(self, x):
        feats = []
        h = x
        for conv, pool, w in zip(self.convs, self.pools, self.skips):
            h = pool(nn.functional.leaky_relu(conv(h), 0.2))
            feats.append(h * w)
        return h, feats

class Generator(nn.Module):
    """U-Net：编码器(32,64,64,128) 2048->128；噪声拼接到隐码；
    解码器逐级反卷积上采样，并在对应分辨率拼接加权跳跃特征。"""
    def __init__(self, chans=(32, 64, 64, 128), noise_ch=128):
        super().__init__()
        self.enc = Encoder(chans)
        c = chans
        # 隐码: c[-1]+noise @ len 128
        # 逐级: ->64@256 (拼接 c[2]=64) ->64@512 (拼接 c[1]=64) ->32@1024 (拼接 c[0]=32) ->32@2048
        self.de1 = nn.ConvTranspose1d(c[3] + noise_ch, c[2], 4, 2, 1, bias=False)
        self.de2 = nn.ConvTranspose1d(c[2] + c[2], c[1], 4, 2, 1, bias=False)
        self.de3 = nn.ConvTranspose1d(c[1] + c[1], c[0], 4, 2, 1, bias=False)
        self.de4 = nn.ConvTranspose1d(c[0] + c[0], c[0], 4, 2, 1, bias=False)
        self.out = nn.Conv1d(c[0], 1, kernel_size=15, stride=1, padding=7, bias=False)
        self.noise_ch = noise_ch
    def forward(self, x):
        h, feats = self.enc(x)  # feats: [32@1024, 64@512, 64@256, 128@128](已加权)
        z = torch.randn(x.size(0), self.noise_ch, h.size(2), device=x.device)
        h = torch.cat([h, z], dim=1)
        lrelu = nn.functional.leaky_relu
        h = lrelu(self.de1(h), 0.2)                      # 64 @ 256
        h = torch.cat([h, feats[2]], dim=1)              # +64 -> 128 @ 256
        h = lrelu(self.de2(h), 0.2)                      # 64 @ 512
        h = torch.cat([h, feats[1]], dim=1)              # +64 -> 128 @ 512
        h = lrelu(self.de3(h), 0.2)                      # 32 @ 1024
        h = torch.cat([h, feats[0]], dim=1)              # +32 -> 64 @ 1024
        h = lrelu(self.de4(h), 0.2)                      # 32 @ 2048
        return torch.tanh(self.out(h))

class Discriminator(nn.Module):
    def __init__(self, chans=(32, 64, 64, 128)):
        super().__init__()
        self.enc = Encoder(chans)
        flat = chans[-1] * (LENGTH // (2 ** len(chans)))
        self.fc = nn.Linear(flat, 1)
    def forward(self, x):
        h, _ = self.enc(x)
        return self.fc(h.flatten(1))  # LSGAN：线性输出，不接 sigmoid

def load_windows(sid):
    ictal = np.load(os.path.join(WIN_DIR, f'{sid}_ictal.npy'))[:, ::DS] / SCALE
    inter = np.load(os.path.join(WIN_DIR, f'{sid}_inter.npy'))[:, ::DS] / SCALE
    return ictal.astype(np.float32), inter.astype(np.float32)

def zscore(X):
    """逐窗标准化：GAN 在形态空间学习，振幅在生成后按训练折发作窗 std 中位数还原"""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-6
    return ((X - m) / s).astype(np.float32)

def spec_loss(fake, real):
    """平移不变的谱幅度损失：log1p|rFFT| 的逐频点 L1。
    时域 L1 对相位随机的振荡信号退化为逐点中位数回归（振幅坍缩），
    谱幅度与相位无关，直接约束频带功率分布而不引入逐点坍缩。"""
    Ff = torch.log1p(torch.abs(torch.fft.rfft(fake, dim=-1)))
    Fr = torch.log1p(torch.abs(torch.fft.rfft(real, dim=-1)))
    return (Ff - Fr).abs().mean()

def train_fold(hold_out):
    ckpt = os.path.join(OUT_DIR, f'ckpt_{hold_out}.pt')
    final_synth = os.path.join(OUT_DIR, f'synth_ictal_{hold_out}.npy')
    if os.path.exists(final_synth):
        print(f'[{hold_out}] already done, skip')
        return
    train_subs = [s for s in SUBJECTS if s != hold_out]
    ictal_all, inter_all = [], []
    for s in train_subs:
        i, n = load_windows(s)
        ictal_all.append(i); inter_all.append(n)
    ictal = np.concatenate(ictal_all)
    inter = np.concatenate(inter_all)
    # 条件窗的患者标签（训练折编号），供身份对抗去偏的 UID 头使用
    inter_lbl = np.concatenate([[k] * len(n) for k, n in enumerate(inter_all)]).astype(np.int64)
    # 记录训练折真实发作窗的振幅分布（群体统计量，供生成后振幅还原）
    ictal_std_median = float(np.median(ictal.std(axis=1)))
    ictal = zscore(ictal)
    inter = zscore(inter)
    # v2：默认使用训练折全部发作窗（MAX_PAIRS=0）；条件/参照配对在每个 epoch
    # 重新随机抽取，避免 v1 固定配对导致的 L1 中位数回归压力集中
    rng = np.random.default_rng(42)
    if MAX_PAIRS > 0 and len(ictal) > MAX_PAIRS:
        ictal = ictal[rng.choice(len(ictal), MAX_PAIRS, replace=False)]
    n_pairs = len(ictal)
    Yr_all = torch.from_numpy(ictal).unsqueeze(1)   # 参照: ictal
    INTER = torch.from_numpy(inter)                  # 条件池: interictal
    INTER_LBL = torch.from_numpy(inter_lbl)          # 条件池患者标签（训练折编号）
    n_steps = n_pairs // BATCH
    print(f'  [{hold_out}] pairs={n_pairs} steps/epoch={n_steps} device={DEVICE} '
          f'lambda_adv={LAMBDA_ADV}', flush=True)

    G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
    U = UIDHead(len(train_subs)).to(DEVICE)
    # G 与 UID 头共用优化器：GRL 保证头最小化身份交叉熵、G 最大化之（DANN 式）
    optG = torch.optim.Adam(list(G.parameters()) + list(U.parameters()), lr=2e-4, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.0, 0.9))
    mse, l1, ce = nn.MSELoss(), nn.L1Loss(), nn.CrossEntropyLoss()

    log = []
    start_ep = 0
    if os.path.exists(ckpt):  # 断点续训
        st = torch.load(ckpt, weights_only=False, map_location=DEVICE)
        G.load_state_dict(st['G']); D.load_state_dict(st['D'])
        if 'U' in st:
            U.load_state_dict(st['U'])
        optG.load_state_dict(st['optG']); optD.load_state_dict(st['optD'])
        log = st['log']; start_ep = st['epoch']
        print(f'  [{hold_out}] resume from epoch {start_ep}', flush=True)
    t0 = time.time()
    for ep in range(start_ep, EPOCHS):
        dg, gg = 0.0, 0.0
        # 逐 epoch 重新随机配对条件窗，并打乱参照窗顺序
        pairs_idx = rng.integers(0, len(INTER), n_pairs)
        order = rng.permutation(n_pairs)
        for i in range(n_steps):
            b = order[i * BATCH:(i + 1) * BATCH]
            xc = INTER[pairs_idx[b]].unsqueeze(1).to(DEVICE)
            yr = Yr_all[b].to(DEVICE)
            ul = INTER_LBL[pairs_idx[b]].to(DEVICE)
            # --- D ---
            optD.zero_grad()
            fake = G(xc)
            ld = mse(D(yr), torch.ones_like(D(yr))) + mse(D(fake.detach()), torch.zeros_like(D(fake.detach())))
            ld.backward(); optD.step()
            # --- G + UID头（梯度反转身份对抗） ---
            optG.zero_grad()
            fake = G(xc)
            lg = (mse(D(fake), torch.ones_like(D(fake)))
                  + LAMBDA_L1 * l1(fake, yr)
                  + LAMBDA_SPEC * spec_loss(fake, yr)
                  + ce(U(GradReverse.apply(fake, LAMBDA_ADV)), ul))
            lg.backward(); optG.step()
            dg += ld.item(); gg += lg.item()
        if (ep + 1) % 5 == 0:
            print(f'  [{hold_out}] epoch {ep+1}/{EPOCHS} D={dg/n_steps:.3f} G={gg/n_steps:.1f} ({time.time()-t0:.0f}s)', flush=True)
        log.append({'epoch': ep + 1, 'd_loss': dg / n_steps, 'g_loss': gg / n_steps})
        torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'U': U.state_dict(),
                    'optG': optG.state_dict(), 'optD': optD.state_dict(),
                    'epoch': ep + 1, 'log': log}, ckpt)
    # 生成：用留一被试的 interictal 窗（标准化后）生成合成 ictal，
    # 再按训练折发作窗振幅中位数还原振幅（群体级统计量，不含个体信息）
    _, inter_h = load_windows(hold_out)
    inter_h = zscore(inter_h)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(inter_h), N_GEN)
    G.eval()
    synth = []
    with torch.no_grad():
        for i in range(0, N_GEN, 64):
            xc = torch.from_numpy(inter_h[idx[i:i+64]]).unsqueeze(1).to(DEVICE)
            synth.append(G(xc).squeeze(1).cpu().numpy())
    synth = np.concatenate(synth)
    synth = zscore(synth) * ictal_std_median * SCALE  # 单位化后按训练折振幅中位数还原
    print(f'  [{hold_out}] amplitude rescale x{ictal_std_median:.2f} (train-fold median ictal std)', flush=True)
    np.save(os.path.join(OUT_DIR, f'synth_ictal_{hold_out}.npy'), synth.astype(np.float32))
    torch.save(G.state_dict(), os.path.join(OUT_DIR, f'generator_loo_{hold_out}.pt'))
    json.dump(log, open(os.path.join(OUT_DIR, f'train_log_{hold_out}.json'), 'w'))
    print(f'[{hold_out}] done in {time.time()-t0:.0f}s, synth shape {synth.shape}', flush=True)

if __name__ == '__main__':
    only = os.environ.get('ONLY', '')
    folds = [only] if only else SUBJECTS
    for s in folds:
        print(f'=== leave-one-out: {s} ===', flush=True)
        train_fold(s)
    print('ALL DONE')
