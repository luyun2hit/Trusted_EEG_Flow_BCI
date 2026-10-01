# -*- coding: utf-8 -*-
"""用时间隔离版合成数据重绘论文图件：波形对照（论文图 3）与谱相似度（论文图 4）。
输出覆盖 results/fig_real_vs_synth.png、results/fig_similarity.png，
并更新 results/fig_waveforms_source.npz 的源数据。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
WIN_DIR = os.path.join(HERE, 'windows')
GAN_DIR = os.path.join(HERE, 'gan_results_timeiso')
RES_DIR = os.path.join(HERE, 'results')
FS, DS = 256, 2

# ---- 图 3：真实 vs 合成波形（chb01，时间隔离版合成数据）----
real = np.load(os.path.join(WIN_DIR, 'chb01_ictal.npy'))[10]
synth = np.load(os.path.join(GAN_DIR, 'synth_ictal_chb01.npy'))[10]
np.savez(os.path.join(RES_DIR, 'fig_waveforms_source.npz'),
         real_chb01_ictal_win10=real, synth_chb01_timeiso_win10=synth)
t = np.arange(1024) / FS
t2 = np.arange(512) / (FS // DS)
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
for c, ch in enumerate(['F7-T3', 'F8-T4']):
    axes[0, c].plot(t, real[c * 1024:(c + 1) * 1024], lw=0.6, color='steelblue')
    axes[0, c].set_title(f'Real ictal ({ch})'); axes[0, c].set_xlabel('s'); axes[0, c].set_ylabel('µV')
    axes[1, c].plot(t2, synth[c * 512:(c + 1) * 512], lw=0.6, color='indianred')
    axes[1, c].set_title(f'Synthetic ictal ({ch})'); axes[1, c].set_xlabel('s'); axes[1, c].set_ylabel('µV')
fig.suptitle('CHB-MIT: real vs EpilepsyGAN-synthetic ictal windows')
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'fig_real_vs_synth.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig_real_vs_synth.png regenerated (time-isolated synth)')

# ---- 图 4：谱相似度（时间隔离版数值）----
sim = json.load(open(os.path.join(RES_DIR, 'nnsim_timeiso.json'), encoding='utf-8'))['similarity']
keys = ['chb01', 'chb03', 'chb05', 'chb08']
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(keys, [sim[k]['real_real'] for k in keys], 'o-', label='real-real')
ax.plot(keys, [sim[k]['real_synth'] for k in keys], 's-', label='real-synthetic')
ax.set_ylabel('Spectral cosine similarity'); ax.legend()
ax.set_title('Spectral similarity per patient')
fig.tight_layout()
fig.savefig(os.path.join(RES_DIR, 'fig_similarity.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig_similarity.png regenerated (time-isolated synth)')
