# -*- coding: utf-8 -*-
"""时间区间隔离版再生成：仅用各留一被试间期池的前半时段窗（与 TSTR 训练负样本同侧）
作为条件，重新生成合成发作信号，供时间隔离版 TSTR 复核。
训练好的留一生成器不变（训练数据来自其他患者，与目标患者时段无关）。
输出: gan_results_timeiso/synth_ictal_{sid}.npy + manifest.json

v4 修复（2026-09-30）：
1) 时间隔离边界——旧版取间期池前 295 个窗（591//2），最后一个条件窗覆盖
   第二个间期段的 [97,101)s，而 TSTR 测试侧第一个非重叠窗覆盖 [100,104)s，
   两者共享 1 秒原始 EEG。现改为前 294 个窗：最后一个条件窗覆盖 [96,100)s，
   与测试侧起点 100s 严格衔接，无任何共享样本（conditioning_end <= test_start）。
   间期池布局：3 段 × 197 窗（stride 1s，4s 窗）；非重叠测试集 3 段 × 50 窗，
   后半 75 窗自第二段第 25 窗（[100,104)s）起。
2) 逐折生成种子——生成前重置 torch 随机状态（7000+折号），生成结果与
   折的运行/跳过顺序无关。

v5 修复（2026-09-30，第二轮外部评审）：
3) 溯源清单——旧版 skip 只看输出文件是否存在，会把旧生成器/旧配置/旧条件池
   生成的合成数据静默当作当前结果。现写 manifest.json，记录四个源生成器的
   SHA-256、源训练配置哈希、条件池大小与时间边界、逐折生成种子、各合成数组
   的 SHA-256 及本脚本与 dsp.py 的代码哈希；skip 前逐项校验，任一不匹配即
   重新生成。
4) torch.load 加 weights_only=True（生成器权重为纯张量，无需反序列化对象）。
"""
import os, json, hashlib, sys
import numpy as np
import torch

from epilepsygan_chbmit import (Generator, load_windows, zscore, SUBJECTS,
                                N_GEN, SCALE, DEVICE, CONFIG, _sha256,
                                _code_hash, CFG)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.environ.get('SRC_GAN_DIR', os.path.join(HERE, 'gan_results'))
OUT_DIR = os.environ.get('OUT_GAN_DIR', os.path.join(HERE, 'gan_results_timeiso'))
os.makedirs(OUT_DIR, exist_ok=True)

MANIFEST = os.path.join(OUT_DIR, 'manifest.json')
N_COND = 294  # 间期池前半时段丢弃边界窗后的条件窗数（v4）


def current_manifest():
    """按当前源生成器/配置/数据计算应有的清单内容。"""
    man = {"src_dir": os.path.abspath(SRC_DIR),
           "src_config_hash": CFG,
           "n_cond": N_COND,
           "time_isolation": "conditioning windows end <= test windows start "
                             "(no shared raw samples)",
           "gen_seed_base": CONFIG["gen_seed_base"],
           "code_hash": _code_hash(),
           "script_hash": _sha256(os.path.abspath(__file__)),
           "python": sys.version.split()[0],
           "torch": torch.__version__,
           "generators": {}, "outputs": {}}
    for sid in SUBJECTS:
        gp = os.path.join(SRC_DIR, f'generator_loo_{sid}.pt')
        if os.path.exists(gp):
            man["generators"][f'generator_loo_{sid}.pt'] = _sha256(gp)
        op = os.path.join(OUT_DIR, f'synth_ictal_{sid}.npy')
        if os.path.exists(op):
            man["outputs"][f'synth_ictal_{sid}.npy'] = _sha256(op)
    return man


def outputs_valid():
    """skip 的充要条件：清单存在且与当前源生成器/配置/产物哈希全部一致。"""
    if not os.path.exists(MANIFEST):
        return False
    try:
        old = json.load(open(MANIFEST))
        cur = current_manifest()
        keys = ("src_dir", "src_config_hash", "n_cond", "gen_seed_base",
                "code_hash", "script_hash", "generators", "outputs")
        return all(old.get(k) == cur.get(k) for k in keys)
    except Exception:
        return False


if outputs_valid():
    print(f'time-isolated outputs match manifest (config {CFG}), skip all')
    raise SystemExit(0)

# 任一校验失败：清理旧输出后全量重生成
for sid in SUBJECTS:
    op = os.path.join(OUT_DIR, f'synth_ictal_{sid}.npy')
    if os.path.exists(op):
        print(f'[{sid}] 清单校验未通过，删除旧合成数据并重新生成', flush=True)
        os.remove(op)

for hold_out in SUBJECTS:
    out_path = os.path.join(OUT_DIR, f'synth_ictal_{hold_out}.npy')
    # 训练折真实发作窗振幅中位数（与训练时一致：加载已按 /SCALE 缩放）
    train_subs = [s for s in SUBJECTS if s != hold_out]
    ictal_all = np.concatenate([load_windows(s)[0] for s in train_subs])
    ictal_std_median = float(np.median(ictal_all.std(axis=1)))
    # 条件窗：间期池前半时段，但丢弃边界窗（v4）：前 294 窗最晚覆盖 [96,100)s，
    # 测试侧自 [100,104)s 起，两侧不共享任何原始样本
    _, inter_h = load_windows(hold_out)
    n_cond = len(inter_h) // 2 - 1   # 295 - 1 = 294
    assert n_cond == N_COND, f'条件池大小异常: {n_cond} != {N_COND}'
    inter_h = inter_h[:n_cond]
    inter_h = zscore(inter_h)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(inter_h), N_GEN)
    # v4：逐折固定生成种子，与折运行顺序无关
    torch.manual_seed(CONFIG["gen_seed_base"] + SUBJECTS.index(hold_out))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["gen_seed_base"] + SUBJECTS.index(hold_out))
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(os.path.join(SRC_DIR, f'generator_loo_{hold_out}.pt'),
                                 map_location=DEVICE, weights_only=True))
    G.eval()
    synth = []
    with torch.no_grad():
        for i in range(0, N_GEN, 64):
            xc = torch.from_numpy(inter_h[idx[i:i + 64]]).unsqueeze(1).to(DEVICE)
            synth.append(G(xc).squeeze(1).cpu().numpy())
    synth = np.concatenate(synth)
    synth = zscore(synth) * ictal_std_median * SCALE
    np.save(out_path, synth.astype(np.float32))
    print(f'[{hold_out}] time-isolated regen done, cond pool {n_cond} windows '
          f'(boundary window dropped), synth {synth.shape}', flush=True)

json.dump(current_manifest(), open(MANIFEST, 'w'), indent=2, ensure_ascii=False)
print('ALL REGEN DONE')
