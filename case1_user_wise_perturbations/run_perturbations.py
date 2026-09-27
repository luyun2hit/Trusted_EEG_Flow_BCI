# -*- coding: utf-8 -*-
"""
run_perturbations.py — 案例一（用户级扰动）真实运行实验，v2 修正版。

数据：eegmmidb 36 被试（S001–S036），仅左/右手运动想象 run：
  训练块 block1=R04，测试块 block2+3=R08+R12（跨块泛化协议）。
扰动方法：no / rand / sn / optim_linf(EMIN) / adv_linf(EMAX)。
协议按原论文（Chen et al., J. Neural Eng. 2025）算法描述实现：
  - 扰动只加在训练（发布）侧，每个用户一个模板，Linfty 界 = std * maskamp
    （rand/sn 0.5，EMIN/EMAX 0.3，逐用户 std，与原仓库一致）；
  - EMIN：随机初始化的 EEGNet+Discriminator（不训练，冻结），
    TimeChunkShuffle(5) 增强下最小化 UID 交叉熵；
  - EMAX：先训练 3 个替代 UID 模型（冻结），最大化 UID 交叉熵；
  - 任务/UID 评估模型：EEGNet + Classifier/Discriminator，30 epochs，
    Adam lr 0.01（半程 x0.1），batch 128，5 个随机种子；
  - 跨架构攻击：ShallowConvNet + Discriminator 作为第二 UID 攻击器；
  - 鲁棒性抽查：测试窗随机时移（±80 点）下的 UID 识别率。
与原论文的尺度差异（36/109 被试、30/100 epochs、5 种子而非 5 次重复）
在 README 与正文中明确说明，结果为简化协议下的独立验证。

缓存绑定配置哈希：修改任何实验配置后旧缓存自动失效。
断点续跑：结果逐行追加 results/，扰动与替代模型缓存在 cache/<hash>/。
"""
import os, sys, time, json, argparse, hashlib, csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from eeg_models import (EEGNet, ShallowConvNet, Classifier, Discriminator,
                        calc_out_size, init_weights, TimeChunkShuffle, freeze)

torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.path.join(HERE, "windows_lr")
RES = os.path.join(HERE, "results")
CACHE_ROOT = os.path.join(HERE, "cache")
os.makedirs(RES, exist_ok=True)

CHANS, SAMPLES, N_SUBJ = 64, 640, 36
CONFIG = {
    "version": 2,
    "data": "eegmmidb S001-S036 R04(train)/R08+R12(test) left-vs-right MI",
    "win_dir": "windows_lr", "fs": 160, "chans": CHANS, "samples": SAMPLES,
    "epochs": 30, "lr": 0.01, "batch": 128,
    "pert_epochs": 20, "sub_epochs": 25,
    "maskamp": {"rand": 0.5, "sn": 0.5, "optim_linf": 0.3, "adv_linf": 0.3},
    "seeds": [0, 1, 2, 3, 4],
    "emin_init": "random-untrained-frozen", "emax_submodels": 3,
}
CFG = hashlib.sha256(json.dumps(CONFIG, sort_keys=True)
                     .encode()).hexdigest()[:12]
CACHE = os.path.join(CACHE_ROOT, CFG)
os.makedirs(CACHE, exist_ok=True)
with open(os.path.join(CACHE, "config.json"), "w") as fp:
    json.dump(CONFIG, fp, indent=2, ensure_ascii=False)

EPOCHS, LR, BATCH = CONFIG["epochs"], CONFIG["lr"], CONFIG["batch"]
PERT_EPOCHS, SUB_EPOCHS = CONFIG["pert_epochs"], CONFIG["sub_epochs"]
MASKAMP = CONFIG["maskamp"]
PERTS = ["no", "rand", "sn", "optim_linf", "adv_linf"]
SEEDS = CONFIG["seeds"]
CSV = os.path.join(RES, "perturbation_results.csv")


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def load_data():
    d1 = np.load(os.path.join(WIN, "block1.npz"))
    d2 = np.load(os.path.join(WIN, "block2.npz"))
    d3 = np.load(os.path.join(WIN, "block3.npz"))
    xtr, ytr, str_ = d1["x"], d1["y"], d1["s"]
    xte = np.concatenate([d2["x"], d3["x"]])
    yte = np.concatenate([d2["y"], d3["y"]])
    ste = np.concatenate([d2["s"], d3["s"]])
    return (xtr[:, None].astype(np.float32), ytr, str_,
            xte[:, None].astype(np.float32), yte, ste)


def batches(x, y, batch=BATCH, shuffle=True):
    n = len(x)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch):
        j = idx[i:i + batch]
        yield torch.from_numpy(x[j]), torch.from_numpy(y[j])


def train_net(x, y, epochs, lr, f, c, tag=""):
    opt = optim.Adam(list(f.parameters()) + list(c.parameters()),
                     lr=lr, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        l = lr * (0.1 if ep >= epochs // 2 else 1.0)
        for g in opt.param_groups:
            g["lr"] = l
        f.train(); c.train()
        for bx, by in batches(x, y):
            opt.zero_grad()
            loss = crit(c(f(bx)), by)
            loss.backward()
            opt.step()
    return f, c


def make_eegnet_uid():
    f = EEGNet(CHANS, SAMPLES)
    c = Discriminator(calc_out_size(f, CHANS, SAMPLES), N_SUBJ)
    return f, c


def emin_models(seed):
    """EMIN：随机初始化（不训练）并冻结的 UID 模型，与原论文算法一致。"""
    set_seed(10_000 + seed)
    f, c = make_eegnet_uid()
    f.apply(init_weights); c.apply(init_weights)
    return [freeze(f)], [freeze(c)]


def emax_models(xtr, str_, seed):
    """EMAX：训练 3 个替代 UID 模型后冻结，带缓存。"""
    fs, cs = [], []
    for i in range(CONFIG["emax_submodels"]):
        mk = os.path.join(CACHE, f"submodel_emax_seed{seed}_{i}.pt")
        f, c = make_eegnet_uid()
        if os.path.exists(mk):
            sd = torch.load(mk, weights_only=True)
            f.load_state_dict(sd["f"]); c.load_state_dict(sd["c"])
        else:
            set_seed(seed * 100 + i)
            f.apply(init_weights); c.apply(init_weights)
            train_net(xtr, str_, SUB_EPOCHS, 1e-2, f, c)
            torch.save({"f": f.state_dict(), "c": c.state_dict()}, mk)
        fs.append(freeze(f)); cs.append(freeze(c))
    return fs, cs


def gen_perturbation(pm, xtr, str_, seed):
    """返回加扰后的训练集（numpy）。缓存绑定配置哈希。"""
    ck = os.path.join(CACHE, f"pert_{pm}_seed{seed}.npy")
    if os.path.exists(ck):
        return np.load(ck)
    set_seed(seed)
    t0 = time.time()
    if pm == "rand":
        x = xtr.copy()
        for k in range(N_SUBJ):
            templ = (np.random.rand(1, CHANS, SAMPLES) * 2 - 1)
            templ = templ * np.std(xtr[str_ == k]) * MASKAMP["rand"]
            x[str_ == k] += templ.astype(np.float32)
    elif pm == "sn":
        x = xtr.copy()
        cho = np.random.choice(range(1, 1000), N_SUBJ, replace=False)
        for k in range(N_SUBJ):
            bits = list("{:0>10}".format(str(bin(cho[k]))[2:]))
            bits = np.array([list(map(int, b * 10)) for b in bits]).reshape(-1)
            temp = np.tile(bits, (CHANS, 500))[:, :SAMPLES] * 2 - 1
            coef = np.random.uniform(0.5, 1.5, CHANS)[:, None, None]
            templ = (np.std(xtr[str_ == k]) * coef * temp[None]
                     * MASKAMP["sn"])
            x[str_ == k] += templ.astype(np.float32)[0]
    elif pm in ("optim_linf", "adv_linf"):
        linf = float(np.std(xtr)) * MASKAMP[pm]
        if pm == "optim_linf":
            fs, cs = emin_models(seed)          # 随机初始化，不训练
        else:
            fs, cs = emax_models(xtr, str_, seed)  # 训练后的替代模型
        crit = nn.CrossEntropyLoss()
        pk = os.path.join(CACHE, f"pertck_{pm}_seed{seed}.pt")
        perturbation = torch.zeros(N_SUBJ, 1, CHANS, SAMPLES)
        nn.init.normal_(perturbation, mean=0, std=1e-3)
        opt = optim.Adam([perturbation.requires_grad_()], lr=1e-1)
        start_ep = 0
        if os.path.exists(pk):
            sd = torch.load(pk, weights_only=True)
            perturbation.data = sd["perturbation"]
            if "opt" in sd:
                opt.load_state_dict(sd["opt"])
            start_ep = sd["epoch"]
        tcf = TimeChunkShuffle(5)
        ep = start_ep
        while ep < PERT_EPOCHS:
            for bx, bs in batches(xtr, str_):
                perb = bx + torch.tanh(perturbation[bs]) * linf
                if pm == "optim_linf":
                    perb = tcf(perb)
                loss = sum((-crit(c(f(perb)), bs) if pm == "adv_linf"
                            else crit(c(f(perb)), bs))
                           for f, c in zip(fs, cs))
                opt.zero_grad()
                loss.backward()
                opt.step()
            ep += 1
            torch.save({"perturbation": perturbation.detach(),
                        "opt": opt.state_dict(), "epoch": ep}, pk)
            print(f"  [{pm} seed{seed}] pert epoch {ep}/{PERT_EPOCHS}",
                  flush=True)
        with torch.no_grad():
            x = (torch.from_numpy(xtr)
                 + torch.tanh(perturbation[torch.from_numpy(str_)])
                 * linf).numpy()
    else:
        raise ValueError(pm)
    np.save(ck, x)
    print(f"  [gen {pm} seed{seed}] {time.time() - t0:.0f}s", flush=True)
    return x


def eval_bca(f, c, xte, yte):
    f.eval(); c.eval()
    preds = []
    with torch.no_grad():
        for bx, _ in batches(xte, np.zeros(len(xte), np.int64),
                             shuffle=False):
            preds.append(c(f(bx)).argmax(dim=1).numpy())
    preds = np.concatenate(preds)
    accs = [np.mean(preds[yte == k] == k) for k in np.unique(yte)]
    return float(np.mean(accs))


def train_eval(xtr, ytr, xte, yte, n_classes, arch, seed):
    """训练 EEGNet/ShallowConvNet + 分类头，返回测试平衡准确率。"""
    set_seed(seed)
    if arch == "eegnet":
        f = EEGNet(CHANS, SAMPLES)
    elif arch == "shallow":
        f = ShallowConvNet(CHANS, SAMPLES)
    else:
        raise ValueError(arch)
    dim = calc_out_size(f, CHANS, SAMPLES)
    c = Discriminator(dim, n_classes) if n_classes > 2 \
        else Classifier(dim, n_classes)
    f.apply(init_weights); c.apply(init_weights)
    train_net(xtr, ytr, EPOCHS, LR, f, c)
    return eval_bca(f, c, xte, yte)


def append_csv(path, row, fields):
    hdr = not os.path.exists(path)
    with open(path, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        if hdr:
            w.writeheader()
        w.writerow(row)


def done_set(path, key=("perturbation", "seed")):
    out = set()
    if os.path.exists(path):
        with open(path) as fp:
            for row in csv.DictReader(fp):
                out.add(tuple(row[k] if k == "perturbation" else int(row[k])
                              for k in key))
    return out


def shifted(x, seed, max_shift=80):
    rng = np.random.RandomState(seed)
    out = x.copy()
    for i in range(len(out)):
        out[i] = np.roll(out[i], rng.randint(-max_shift, max_shift + 1),
                         axis=-1)
    return out


def report_norms(xtr, str_):
    """报告每种方法用户模板的实际 Linfty / 平均 L2 / 与数据 std 之比。"""
    path = os.path.join(RES, "perturbation_norms.csv")
    done = done_set(path)
    gstd = float(np.std(xtr))
    for pm in PERTS[1:]:
        for seed in SEEDS:
            if (pm, seed) in done:
                continue
            ck = os.path.join(CACHE, f"pert_{pm}_seed{seed}.npy")
            if not os.path.exists(ck):
                continue
            xp = np.load(ck)
            diff = xp - xtr
            linf = float(np.max(np.abs(diff)))
            l2 = float(np.mean(np.linalg.norm(
                diff.reshape(len(diff), -1), axis=1)))
            append_csv(path, {"perturbation": pm, "seed": seed,
                              "linf": round(linf, 4),
                              "linf_over_std": round(linf / gstd, 4),
                              "mean_l2": round(l2, 4)},
                       ["perturbation", "seed", "linf", "linf_over_std",
                        "mean_l2"])
            print(f"NORM {pm} seed{seed}: Linf={linf:.3f} "
                  f"({linf / gstd:.2f}xstd) L2={l2:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    xtr, ytr, str_, xte, yte, ste = load_data()
    print(f"config {CFG} | train {xtr.shape} test {xte.shape} "
          f"std {xtr.std():.2f}", flush=True)
    if args.bench:
        t0 = time.time()
        train_eval(xtr, ytr, xte[:64], yte[:64], 2, "eegnet", 0)
        print(f"bench {time.time() - t0:.0f}s")
        return

    # 1) 主实验：5 扰动 x 5 种子，任务 + UID（EEGNet 攻击器）
    done = done_set(CSV)
    for seed in SEEDS:
        for pm in PERTS:
            if (pm, seed) in done:
                continue
            t0 = time.time()
            xtr_p = xtr if pm == "no" else gen_perturbation(pm, xtr, str_,
                                                            seed)
            task_bca = train_eval(xtr_p, ytr, xte, yte, 2, "eegnet", seed)
            uid_bca = train_eval(xtr_p, str_, xte, ste, N_SUBJ, "eegnet",
                                 seed)
            append_csv(CSV, {"perturbation": pm, "seed": seed,
                             "task_bca": round(task_bca, 4),
                             "uid_bca": round(uid_bca, 4),
                             "seconds": round(time.time() - t0, 1)},
                       ["perturbation", "seed", "task_bca", "uid_bca",
                        "seconds"])
            print(f"DONE {pm} seed{seed}: task={task_bca:.4f} "
                  f"uid={uid_bca:.4f}", flush=True)

    # 2) 跨架构攻击器：ShallowConvNet UID
    cx = os.path.join(RES, "cross_arch.csv")
    done = done_set(cx)
    for seed in SEEDS:
        for pm in PERTS:
            if (pm, seed) in done:
                continue
            xtr_p = xtr if pm == "no" else gen_perturbation(pm, xtr, str_,
                                                            seed)
            acc = train_eval(xtr_p, str_, xte, ste, N_SUBJ, "shallow", seed)
            append_csv(cx, {"perturbation": pm, "seed": seed,
                            "uid_bca_shallow": round(acc, 4)},
                       ["perturbation", "seed", "uid_bca_shallow"])
            print(f"CROSS {pm} seed{seed}: uid_shallow={acc:.4f}",
                  flush=True)

    # 3) 鲁棒性抽查：随机时移下的 UID
    rk = os.path.join(RES, "robustness.csv")
    done = done_set(rk)
    for pm in ("rand", "adv_linf"):
        for seed in SEEDS:
            if (pm, seed) in done:
                continue
            xtr_p = gen_perturbation(pm, xtr, str_, seed)
            acc = train_eval(xtr_p, str_, shifted(xte, seed), ste,
                             N_SUBJ, "eegnet", seed)
            append_csv(rk, {"perturbation": pm, "seed": seed,
                            "uid_bca_shifted": round(acc, 4)},
                       ["perturbation", "seed", "uid_bca_shifted"])
            print(f"ROBUST {pm} seed{seed}: {acc:.4f}", flush=True)

    # 4) 实际扰动范数报告
    report_norms(xtr, str_)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
