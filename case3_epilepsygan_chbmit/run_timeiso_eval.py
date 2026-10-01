# -*- coding: utf-8 -*-
"""时间隔离版评估驱动：对 GAN_DIR 指向的合成数据运行 TSTR / UID / NN+谱相似度，
结果写入 OUT_JSON（不覆盖主 results/summary.json）。

用法:
  GAN_DIR=gan_results_timeiso MODE=uid   OUT_JSON=results/uid_timeiso.json  python run_timeiso_eval.py
  GAN_DIR=gan_results_timeiso MODE=nnsim OUT_JSON=results/nnsim_timeiso.json python run_timeiso_eval.py
  GAN_DIR=gan_results_timeiso MODE=tstr  OUT_JSON=results/tstr_timeiso.json  python run_timeiso_eval.py
  GAN_DIR=gan_results_timeiso MODE=all   OUT_JSON=results/summary_timeiso.json python run_timeiso_eval.py

v3（2026-09-30）：新增 MODE=tstr 与 MODE=all；MODE=all 运行全部三项并写出
论文最终数字所依据的 summary_timeiso.json（此前该文件无生成脚本，为评审
指出的复现链缺口）。
"""
import os, json
import numpy as np

MODE = os.environ.get('MODE', 'uid')
OUT_JSON = os.environ.get('OUT_JSON', 'results/timeiso_eval.json')

import evaluate_chbmit as ev

if MODE == 'uid':
    res = ev.cnn_reidentification()
elif MODE == 'nnsim':
    res = {'nn_audit': ev.nn_audit(), 'similarity': ev.spectral_similarity()}
elif MODE == 'tstr':
    res = {'tstr': ev.tstr_experiment()}
elif MODE == 'all':
    tstr = ev.tstr_experiment()
    uid = ev.cnn_reidentification()
    nnres = ev.nn_audit()
    sim = ev.spectral_similarity()
    res = dict(tstr=tstr, uid=uid, nn_audit=nnres, similarity=sim,
               mean_baseline=round(float(np.mean([r['baseline'] for r in tstr])), 2),
               mean_synthetic=round(float(np.mean([r['synthetic'] for r in tstr])), 2))
else:
    raise SystemExit('unknown MODE')

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('saved', OUT_JSON)
print(json.dumps(res, ensure_ascii=False)[:600])
