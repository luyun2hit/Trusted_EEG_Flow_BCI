# -*- coding: utf-8 -*-
"""
CHB-MIT EpilepsyGAN 评估（对应原论文第 V/VI 节），v3 修正版（2026-09-30）
1) 效用：Train-on-Synthetic-Test-on-Real (TSTR)，随机森林，几何均值(灵敏度,特异度)
   - Ti: 合成 ictal + 真实 interictal（目标患者）
   - Bi: 其他患者真实 ictal + 真实 interictal（基线）
2) 隐私：患者重识别攻击器（间期训练 -> 真实/合成发作测试）
   v2 修正：攻击器对齐原论文——输入为 原始波形 + db4 三级 DWT 系数拼接
   （原论文明确 DWT 输入显著提升识别能力），网络为预激活残差块结构，
   多种子均值±标准差报告；并设阳性对照门槛：若真实数据可识别性
   （Exp_orig）未达到随机水平 2 倍，明确提示攻击器能力不足、隐私结论不成立。
   v1 的 4 层普通 CNN（无 DWT、12 epoch、单次运行）在 N=4 下 Exp_orig 仅
   39.7%（原文同规模约 70%，2.8×随机），阳性对照不达标，证据效力不足。
   v3 修正（2026-09-30，对应外部评审）：
   - 主指标改为平衡准确率（BCA = 各类召回率均值）：真实发作测试集类别
     不平衡（53/48/68/113，多数类基线 40.07%），合成测试集完全平衡
     （每类 500），原始 accuracy 不可直接比较；原始 accuracy 保留为补充，
     并报告逐类召回率与平均混淆矩阵；
   - 降采样统一改用 resample_poly 抗混叠（dsp.decimate），与训练侧一致；
   - 阳性对照不达标时的提示改为：跨状态攻击未通过阳性对照，所得准确率
     本身不能作为身份保护证据（不再断言"攻击器能力不足"）。
3) 最近邻记忆审计：合成窗到训练折真实发作窗的最近邻距离分布
   对比留一被试真实发作窗到同一参照集的距离分布，检验生成器逐样本记忆。
   v3 增补最小值、1% 分位数与近重复率。
4) 谱余弦相似度：real-real vs real-synthetic（v3：real-real 禁止自身配对）
输出: results/ 下的表格 CSV、图 PNG、汇总 JSON
环境变量：UID_SEEDS(默认"0,1,2,3,4") UID_EPOCHS(默认40) DEVICE(默认auto) GAN_DIR(默认gan_results)
"""
import os, json, time
import numpy as np

from dsp import decimate

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_DIR = os.path.join(HERE, 'windows')
GAN_DIR = os.environ.get('GAN_DIR', os.path.join(HERE, 'gan_results'))
RES_DIR = os.path.join(HERE, 'results')
os.makedirs(RES_DIR, exist_ok=True)

SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb08']
ALL_SUBJECTS = list(SUBJECTS)
FS = 256
DS = 2
N_TRAIN = 500   # Ti/Bi 中各类训练样本数（原论文 2000，按数据量缩减）
N_REP = 5       # 重复次数（原论文 15）

def load(sid, kind):
    return np.load(os.path.join(WIN_DIR, f'{sid}_{kind}.npy'))

def load_synth(sid):
    return np.load(os.path.join(GAN_DIR, f'synth_ictal_{sid}.npy'))

def ds(X):
    """抗混叠降采样到 128Hz（v3：替代旧版 ::2 抽取）。"""
    return decimate(X, DS)

# ---------- 特征（谱域，对应原论文 RF 管线，教学化简化） ----------
def extract_features(X):
    """X: (N, L) 拼接窗（自动按长度拆分两通道）。
    每通道 Welch 5 带对数功率 + 对数总谱功率 => 12 维
    （v3 更正命名：末项为 log10(完整 Welch PSD 积分)，即对数总谱功率，
    旧注释误写为"对数方差"）。"""
    from scipy.signal import welch
    n = X.shape[0]
    half = X.shape[1] // 2
    fs_eff = FS if half == 1024 else FS // DS
    feats = np.zeros((n, 12), np.float32)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    Xc = X.reshape(n, 2, half)
    for i in range(n):
        f0, P0 = welch(Xc[i, 0], fs=fs_eff, nperseg=min(256, half))
        f1, P1 = welch(Xc[i, 1], fs=fs_eff, nperseg=min(256, half))
        row = []
        for P, f in ((P0, f0), (P1, f1)):
            for lo, hi in bands:
                m = (f >= lo) & (f < hi)
                row.append(np.log10(np.trapezoid(P[m], f[m]) + 1e-12))
            row.append(np.log10(np.trapezoid(P, f) + 1e-12))
        feats[i] = row
    return feats

def gmean_sens_spec(y, p):
    tp = ((y == 1) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    tn = ((y == 0) & (p == 0)).sum(); fp = ((y == 0) & (p == 1)).sum()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return float(np.sqrt(sens * spec)), float(sens), float(spec)

def tstr_experiment():
    from sklearn.ensemble import RandomForestClassifier
    rows = []
    rng = np.random.default_rng(0)
    feat_cache = {}
    def feats_of(arr, key):
        if key not in feat_cache:
            feat_cache[key] = extract_features(arr)
        return feat_cache[key]
    for sid in SUBJECTS:
        inter_noover_all = ds(load(sid, 'inter_noover'))
        # 训练/测试间期窗按索引对半切分，杜绝同一窗同时进入训练与测试（泄漏修复）
        half = len(inter_noover_all) // 2
        inter_tr, inter_te = inter_noover_all[:half], inter_noover_all[half:]
        ictal_real = ds(load(sid, 'ictal')[::4])     # 去重叠（stride 4s）作测试
        synth = load_synth(sid)
        assert len(synth) >= N_TRAIN, \
            f'{sid} 合成样本 {len(synth)} < {N_TRAIN}，无法按协议抽样'
        others = np.concatenate([load(s, 'ictal') for s in ALL_SUBJECTS if s != sid])
        others = ds(others)
        n_test_i = len(ictal_real)
        n_test_n = min(2 * n_test_i, len(inter_te))
        test_X = np.concatenate([ictal_real, inter_te[:n_test_n]])
        test_y = np.array([1] * n_test_i + [0] * n_test_n)
        ftest = feats_of(test_X, f'test_{sid}')
        g_t, g_b = [], []
        for rep in range(N_REP):
            r = np.random.default_rng(rep)
            # Ti: 合成 ictal + 真实 interictal
            si = r.integers(0, len(synth), min(N_TRAIN, len(synth)))
            ii = r.integers(0, len(inter_tr), N_TRAIN)
            Ti = np.concatenate([synth[si], inter_tr[ii]])
            yi = np.array([1] * len(si) + [0] * N_TRAIN)
            fTi = extract_features(Ti)
            rf = RandomForestClassifier(n_estimators=200, n_jobs=8, random_state=rep)
            rf.fit(fTi, yi)
            g_t.append(gmean_sens_spec(test_y, rf.predict(ftest)))
            # Bi: 其他患者真实 ictal + 真实 interictal
            bi = r.integers(0, len(others), N_TRAIN)
            Bi = np.concatenate([others[bi], inter_tr[ii]])
            fBi = extract_features(Bi)
            rf2 = RandomForestClassifier(n_estimators=200, n_jobs=8, random_state=rep)
            rf2.fit(fBi, yi)
            g_b.append(gmean_sens_spec(test_y, rf2.predict(ftest)))
        gt = np.mean([x[0] for x in g_t]) * 100
        gb = np.mean([x[0] for x in g_b]) * 100
        st = np.std([x[0] for x in g_t]) * 100
        rows.append(dict(patient=sid, baseline=round(gb, 2), synthetic=round(gt, 2),
                         diff=round(gt - gb, 2), std=round(st, 2),
                         sens=round(np.mean([x[1] for x in g_t]), 3),
                         spec=round(np.mean([x[2] for x in g_t]), 3)))
        print(f'[TSTR] {sid}: baseline={gb:.2f}% synthetic={gt:.2f}% diff={gt-gb:+.2f}%', flush=True)
    return rows

# ---------- 患者重识别（v2：DWT+残差攻击器，多种子；v3：BCA 主指标） ----------
def _dwt_concat(x1d):
    """db4 三级 DWT 系数拼接（原论文攻击器输入的第二部分）"""
    import pywt
    return np.concatenate(pywt.wavedec(x1d, 'db4', level=3))

def build_attack_input(X):
    """X: (N, 1024) 双通道拼接窗 @128Hz -> (N, L) [ch0_raw, ch1_raw, ch0_dwt, ch1_dwt]
    对齐原论文攻击器输入：原始波形 + DWT（原论文 256Hz 下为 4136 维）"""
    half = X.shape[1] // 2
    rows = []
    for i in range(X.shape[0]):
        c0, c1 = X[i, :half], X[i, half:]
        rows.append(np.concatenate([c0, c1, _dwt_concat(c0), _dwt_concat(c1)]))
    return np.stack(rows).astype(np.float32)

def cnn_reidentification():
    import torch, torch.nn as nn
    import torch.nn.functional as Fn
    try:
        import pywt  # noqa: F401
    except ImportError:
        raise SystemExit('缺少 PyWavelets：pip install PyWavelets（攻击器 DWT 输入需要）')
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if os.environ.get('DEVICE', 'auto') != 'cpu' else 'cpu'
    # 确定性化（v2.3）：固定 cudnn 算法，保证同机同种子可复现；
    # 跨硬件/库版本仍可能有末位差异，报告以多种子均值±标准差为准
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if DEVICE == 'cpu':
        torch.set_num_threads(16)
    UID_SEEDS = [int(s) for s in os.environ.get('UID_SEEDS', '0,1,2,3,4').split(',')]
    UID_EPOCHS = int(os.environ.get('UID_EPOCHS', '40'))
    N_USR = len(SUBJECTS)

    class ResBlock(nn.Module):
        """预激活残差块（对齐原论文 Fig.3 的 pre-activation 设计）"""
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
        """原论文攻击器的轻量化对齐版：stem conv32/s2 + 4 残差块
        (64,64,128,128，首卷积 stride 2) + AvgPool + FC(128) + FC(n_users)"""
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
        # 攻击器训练稳定化（v2.2）：lr 5e-4 + 余弦退火 + 种子绑定的 shuffle，
        # 降低单种子方差（此前 Exp_orig 种子间波动 13–33%，测量噪声过大）
        torch.manual_seed(seed); np.random.seed(seed)
        m = IDResNet(N_USR).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UID_EPOCHS)
        ce = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(X / 100.0).unsqueeze(1)
        yt = torch.from_numpy(y)
        g = torch.Generator(); g.manual_seed(seed)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, yt),
                                         batch_size=64, shuffle=True, generator=g)
        for ep in range(UID_EPOCHS):
            m.train()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); ce(m(xb), yb).backward(); opt.step()
            sched.step()
        return m

    def metrics(m, X, y):
        """v3：返回 (原始accuracy%, 平衡准确率BCA%, 逐类召回率, 混淆矩阵)"""
        m.eval()
        import torch as _t
        preds = []
        with _t.no_grad():
            for i in range(0, len(X), 512):
                xb = _t.from_numpy(X[i:i+512] / 100.0).unsqueeze(1).to(DEVICE)
                preds.append(m(xb).argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        acc = float((p == y).mean()) * 100
        cm = np.zeros((N_USR, N_USR), np.int64)
        for t_, p_ in zip(y, p):
            cm[t_, p_] += 1
        recalls = np.array([cm[k, k] / max(cm[k].sum(), 1) for k in range(N_USR)])
        bca = float(recalls.mean()) * 100
        return acc, bca, recalls, cm

    # 间期窗训练身份分类器；输入 = 波形 + DWT。
    # v2.1：训练数据改用 stride-1s 间期训练池（591 窗/被试，v2.0 误用仅 150 窗的
    # 非重叠测试集——间期池与发作测试窗时间间隔 ≥300s，不存在泄漏，而攻击器
    # 训练样本量从 600 提升到 2364），并加循环时移增强提升攻击器泛化能力
    print('[UID] building attack inputs (waveform + db4 DWT) ...', flush=True)
    Xi, yi = [], []
    for k, sid in enumerate(SUBJECTS):
        a = ds(load(sid, 'inter'))  # 间期训练池降采样到 1024 长 @128Hz
        Xi.append(a); yi += [k] * len(a)
    Xw = np.concatenate(Xi)
    yi = np.array(yi)
    # 循环时移增强：每窗复制一份并对两通道分别随机平移（在 DWT 之前施加，
    # 保持波形与 DWT 系数一致）
    rng_aug = np.random.default_rng(123)
    half = Xw.shape[1] // 2
    aug = Xw.copy()
    for i in range(len(aug)):
        aug[i, :half] = np.roll(aug[i, :half], int(rng_aug.integers(0, half)))
        aug[i, half:] = np.roll(aug[i, half:], int(rng_aug.integers(0, half)))
    Xw = np.concatenate([Xw, aug])
    yi = np.concatenate([yi, yi])
    Xi = build_attack_input(Xw)
    print(f'[UID] attacker train samples: {len(Xi)} (pool x2 augmented)', flush=True)
    # Exp_orig 测试集：真实 ictal（每 8 窗抽 1，各类 53/48/68/113，不平衡）；
    # Exp_synt 测试集：合成 ictal（每类 500，平衡）——v3 起以 BCA 为主指标
    Xo = np.concatenate([ds(load(s, 'ictal')[::8]) for s in SUBJECTS])
    yo = np.concatenate([[k] * len(load(s, 'ictal')[::8]) for k, s in enumerate(SUBJECTS)])
    Xo = build_attack_input(Xo)
    Xs = build_attack_input(np.concatenate([load_synth(s) for s in SUBJECTS]))
    ys = np.concatenate([[k] * len(load_synth(s)) for k, s in enumerate(SUBJECTS)])

    per_seed = []
    cm_o_sum = np.zeros((N_USR, N_USR), np.float64)
    cm_s_sum = np.zeros((N_USR, N_USR), np.float64)
    for seed in UID_SEEDS:
        m = train_attacker(Xi, yi, seed)
        ao, bo, ro, cmo = metrics(m, Xo, yo)
        as_, bs, rs, cms = metrics(m, Xs, ys)
        cm_o_sum += cmo / np.maximum(cmo.sum(1, keepdims=True), 1)
        cm_s_sum += cms / np.maximum(cms.sum(1, keepdims=True), 1)
        # v5 修复：per_seed 存未舍入原始值，均值/标准差基于原始值计算，
        # 舍入仅发生在最终输出（避免先舍入再聚合引入的偏差）
        per_seed.append(dict(seed=seed,
                             exp_orig=float(ao), exp_orig_bca=float(bo),
                             exp_synt=float(as_), exp_synt_bca=float(bs),
                             recall_orig=[float(x) for x in ro],
                             recall_synt=[float(x) for x in rs]))
        print(f'[UID seed{seed}] Exp_orig acc={ao:.2f}% bca={bo:.2f}% | '
              f'Exp_synt acc={as_:.2f}% bca={bs:.2f}%', flush=True)
    acc_rand = 100.0 / N_USR
    eo = np.array([p['exp_orig'] for p in per_seed])
    es = np.array([p['exp_synt'] for p in per_seed])
    eob = np.array([p['exp_orig_bca'] for p in per_seed])
    esb = np.array([p['exp_synt_bca'] for p in per_seed])
    mo, ms = float(eo.mean()), float(es.mean())
    mob, msb = float(eob.mean()), float(esb.mean())
    # 阳性对照（信息性）：跨状态攻击在真实发作上的可识别性低于 2x 随机时，
    # 所得准确率本身不能作为身份保护证据（v3 措辞，不再断言攻击器能力不足）
    power_ok = bool(mob >= 2.0 * acc_rand)
    if not power_ok:
        print(f'[UID] NOTE: 跨状态攻击未通过阳性对照（Exp_orig BCA={mob:.1f}% < '
              f'2x随机={2*acc_rand:.1f}%）：所得准确率本身不能作为身份保护证据，'
              '隐私结论仅以合成 vs 真实的相对比较为限', flush=True)
    print(f'[UID] Exp_orig acc={mo:.2f}±{eo.std():.2f}% bca={mob:.2f}±{eob.std():.2f}% | '
          f'Exp_synt acc={ms:.2f}±{es.std():.2f}% bca={msb:.2f}±{esb.std():.2f}% | '
          f'Exp_rand={acc_rand:.1f}% (n_seeds={len(UID_SEEDS)})', flush=True)
    return dict(primary_metric='balanced accuracy (macro recall); raw accuracy '
                             'reported as supplementary (test-set class imbalance)',
                exp_orig=round(mo, 2), exp_orig_std=round(float(eo.std()), 2),
                exp_orig_bca=round(mob, 2), exp_orig_bca_std=round(float(eob.std()), 2),
                exp_synt=round(ms, 2), exp_synt_std=round(float(es.std()), 2),
                exp_synt_bca=round(msb, 2), exp_synt_bca_std=round(float(esb.std()), 2),
                exp_rand=round(acc_rand, 2),
                ratio_orig_bca=round(mob / acc_rand, 2),
                ratio_synt_bca=round(msb / acc_rand, 2),
                attacker='IDResNet (waveform + db4 DWT input, adapted from original)',
                n_seeds=len(UID_SEEDS), epochs=UID_EPOCHS,
                std_note='std 为攻击器初始化种子间的变异，非独立实验样本',
                attacker_power_ok=power_ok,
                confusion_orig_mean=np.round(cm_o_sum / len(UID_SEEDS), 3).tolist(),
                confusion_synt_mean=np.round(cm_s_sum / len(UID_SEEDS), 3).tolist(),
                per_seed=per_seed)

# ---------- 最近邻记忆审计（v2 新增；v3 增补 min/p01/近重复率） ----------
def _zw(X):
    """逐窗 z-score（消除振幅差异，聚焦波形形态的记忆）"""
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-6
    return ((X - m) / s).astype(np.float32)

def _nn_min_dist(Q, R, chunk=256):
    """Q 中每个样本到 R 的最小欧氏距离（分块矩阵实现）"""
    R2 = (R ** 2).sum(1)
    out = np.empty(len(Q), np.float64)
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        d2 = (q ** 2).sum(1)[:, None] + R2[None, :] - 2.0 * (q @ R.T)
        out[i:i + len(q)] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out

def nn_audit(n_ref=2000, n_q=500, dup_eps=1e-3):
    """逐样本记忆审计：比较"合成窗→训练折真实发作窗"与"留一被试真实发作窗→
    同一训练折"的最近邻距离分布。若合成窗距离系统性更小，说明生成器在逐样本
    复制训练数据（记忆泄漏）；与真实窗同分布则说明合成窗的新颖性与真实发作相当。
    距离在逐窗 z-score 空间计算（振幅已由群体统计量还原，形态才是记忆载体）。
    v3：除中位数与 5% 分位数外，增补最小值、1% 分位数与近重复率
    （距离 < dup_eps 的比例）。"""
    rng = np.random.default_rng(11)
    out = {}
    for sid in SUBJECTS:
        ref = np.concatenate([ds(load(s, 'ictal')) for s in ALL_SUBJECTS if s != sid])
        ref = _zw(ref[rng.choice(len(ref), min(n_ref, len(ref)), replace=False)])
        sq = _zw(load_synth(sid)[rng.choice(len(load_synth(sid)),
                                            min(n_q, len(load_synth(sid))), replace=False)])
        real = ds(load(sid, 'ictal'))
        rq = _zw(real[rng.choice(len(real), min(n_q, len(real)), replace=False)])
        dsyn = _nn_min_dist(sq, ref)
        dr = _nn_min_dist(rq, ref)
        out[sid] = dict(
            synth_nn_median=round(float(np.median(dsyn)), 3),
            real_nn_median=round(float(np.median(dr)), 3),
            synth_over_real=round(float(np.median(dsyn) / max(np.median(dr), 1e-9)), 3),
            synth_nn_p05=round(float(np.percentile(dsyn, 5)), 3),
            real_nn_p05=round(float(np.percentile(dr, 5)), 3),
            synth_nn_p01=round(float(np.percentile(dsyn, 1)), 3),
            real_nn_p01=round(float(np.percentile(dr, 1)), 3),
            synth_nn_min=round(float(dsyn.min()), 3),
            real_nn_min=round(float(dr.min()), 3),
            synth_dup_rate=round(float((dsyn < dup_eps).mean()), 5),
            real_dup_rate=round(float((dr < dup_eps).mean()), 5))
        print(f'[NN] {sid}: synth->train median={np.median(dsyn):.2f} '
              f'real->train median={np.median(dr):.2f} '
              f'ratio={out[sid]["synth_over_real"]:.2f} '
              f'dup={out[sid]["synth_dup_rate"]:.4f}', flush=True)
    ratios = [v['synth_over_real'] for v in out.values()]
    out['_summary'] = dict(median_ratio=round(float(np.median(ratios)), 3),
                           note='ratio≈1 表示合成窗与训练集的距离分布和真实发作窗相当；'
                                'ratio 显著<1 提示逐样本记忆风险')
    return out

# ---------- 谱余弦相似度 ----------
def spectral_similarity(n_pairs=1500):
    rng = np.random.default_rng(1)
    out = {}
    for sid in SUBJECTS:
        real = load(sid, 'ictal')[::2]   # 行子抽样（隔窗取），非降采样
        synth = load_synth(sid)
        def cossim_pairs(A, B, same_set=False):
            idx_a = rng.integers(0, len(A), n_pairs)
            idx_b = rng.integers(0, len(B), n_pairs)
            if same_set:  # v3：real-real 禁止样本与自身配对
                mask = idx_a == idx_b
                idx_b[mask] = (idx_b[mask] + 1) % len(B)
            sims = []
            for a, b in zip(A[idx_a], B[idx_b]):
                fa = np.abs(np.fft.rfft(a.reshape(2, -1), axis=1))
                fb = np.abs(np.fft.rfft(b.reshape(2, -1), axis=1))
                nmin = min(fa.shape[1], fb.shape[1])
                fa, fb = fa[:, :nmin], fb[:, :nmin]
                s = [(fa[c] @ fb[c]) / (np.linalg.norm(fa[c]) * np.linalg.norm(fb[c]) + 1e-12) for c in range(2)]
                sims.append(np.mean(s))
            return float(np.mean(sims))
        rr = cossim_pairs(real, real, same_set=True)
        rs = cossim_pairs(real, synth)
        out[sid] = dict(real_real=round(rr, 4), real_synth=round(rs, 4))
        print(f'[SIM] {sid}: real-real={rr:.3f} real-synth={rs:.3f}', flush=True)
    return out

# ---------- 图 ----------
def make_figures(tstr, sim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # 跨平台中文字体回退（Windows/macOS/Linux 常见 CJK 字体依次尝试）
    for _f in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC"]:
        try:
            matplotlib.font_manager.findfont(_f, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    # 图1：真实 vs 合成波形
    sid = 'chb01'
    real = load(sid, 'ictal')[10]
    synth = load_synth(sid)[10]
    t = np.arange(1024) / FS
    t2 = np.arange(512) / (FS // DS)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    for c, ch in enumerate(['F7-T3', 'F8-T4']):
        axes[0, c].plot(t, real[c * 1024:(c + 1) * 1024], lw=0.6, color='steelblue')
        axes[0, c].set_title(f'Real ictal ({ch})'); axes[0, c].set_xlabel('s'); axes[0, c].set_ylabel('µV')
        axes[1, c].plot(t2, synth[c * 512:(c + 1) * 512], lw=0.6, color='indianred')
        axes[1, c].set_title(f'Synthetic ictal ({ch})'); axes[1, c].set_xlabel('s'); axes[1, c].set_ylabel('µV')
    fig.suptitle('CHB-MIT: real vs EpilepsyGAN-synthetic ictal windows')
    fig.tight_layout(); fig.savefig(os.path.join(RES_DIR, 'fig_real_vs_synth.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

    # 图2：TSTR 基线 vs 合成
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tstr)); w = 0.38
    ax.bar(x - w / 2, [r['baseline'] for r in tstr], w, label='Baseline (real ictal, other patients)', color='gray')
    ax.bar(x + w / 2, [r['synthetic'] for r in tstr], w, label='Synthetic ictal (EpilepsyGAN)', color='indianred')
    ax.set_xticks(x); ax.set_xticklabels([r['patient'] for r in tstr])
    ax.set_ylabel('Geometric mean of sens./spec. (%)'); ax.legend(); ax.set_title('TSTR on CHB-MIT')
    fig.tight_layout(); fig.savefig(os.path.join(RES_DIR, 'fig_tstr.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

    # 图3：谱相似度
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(sim.keys()), [v['real_real'] for v in sim.values()], 'o-', label='real-real')
    ax.plot(list(sim.keys()), [v['real_synth'] for v in sim.values()], 's-', label='real-synthetic')
    ax.set_ylabel('Spectral cosine similarity'); ax.legend(); ax.set_title('Spectral similarity per patient')
    fig.tight_layout(); fig.savefig(os.path.join(RES_DIR, 'fig_similarity.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    import os as _os
    only = _os.environ.get('ONLY_TSTR', '')
    only_uid = _os.environ.get('ONLY_UID', '')
    t0 = time.time()
    if only:
        SUBJECTS = [only]
        tstr = tstr_experiment()
        print(json.dumps(tstr, ensure_ascii=False))
    elif only_uid:
        uid = cnn_reidentification()
        print(json.dumps(uid, ensure_ascii=False, indent=1))
    else:
        tstr = tstr_experiment()
        json.dump(tstr, open(os.path.join(RES_DIR, 'tstr.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        uid = cnn_reidentification()
        nnres = nn_audit()
        sim = spectral_similarity()
        make_figures(tstr, sim)
        summary = dict(tstr=tstr, uid=uid, nn_audit=nnres, similarity=sim,
                       mean_baseline=round(float(np.mean([r['baseline'] for r in tstr])), 2),
                       mean_synthetic=round(float(np.mean([r['synthetic'] for r in tstr])), 2))
        json.dump(summary, open(os.path.join(RES_DIR, 'summary.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print('ALL EVAL DONE in', round(time.time() - t0), 's')
        print(json.dumps(summary, ensure_ascii=False, indent=1)[:800])
