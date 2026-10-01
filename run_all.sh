#!/usr/bin/env bash
# 一键复现脚本（Linux/macOS/Git Bash）：依次运行三个案例
# 用法：bash run_all.sh
# 注意：案例一需先联网下载 eegmmidb（约 270 MB），案例三按需下载 CHB-MIT 片段（约 60 MB）；
#       案例二需先克隆上游仓库并设置 FEDEEG_DATA（见 case2_fedeeg/README.md）。
set -e
pip install -r requirements.txt

echo "== 案例一：用户级扰动（eegmmidb，CPU 约 2.5 小时） =="
cd case1_user_wise_perturbations
python download.py            # 下载 36 名被试的 R04/R08/R12 run
python preprocess_mi2.py      # 带通、切窗、逐 run 标准化
python run_perturbations.py   # 四类扰动 + 任务/UID 评估（5 种子）
python make_fig.py
cd ..

echo "== 案例二：联邦平均（FedEEG，CPU；正式配置约 45 分钟 + 成员推断抽查） =="
cd case2_fedeeg
python valsplit_run.py        # 论文报告配置：27 轮 × 3 本地 epoch，60:20:20 三划分
                              #   （轮数只按验证集曲线选取；可分臂分种子运行，见 README）
python merge_valsplit.py      # 合并生成 results/valsplit_summary.json（论文数字来源）
python make_fig.py            # 重绘图 2
# 可选：原始 5 轮协议复现——python fedegg_reimpl.py && python merge_results.py
cd ..

echo "== 案例三：合成数据发布（CHB-MIT，GPU 约 5 分钟训练 + 约 15 分钟评估） =="
cd case3_epilepsygan_chbmit
python download_segments.py   # HTTP Range 按需下载（可续跑）
python preprocess.py
EPOCHS=40 python epilepsygan_chbmit.py     # 4 折留一训练（逐折清单 manifest_<sid>.json）
python regen_timeiso.py                    # 时间隔离再生成（条件窗 = 间期池前 294 窗）
# 时间隔离版完整评估（论文最终数字，生成 results/summary_timeiso.json）：
GAN_DIR=gan_results_timeiso MODE=all OUT_JSON=results/summary_timeiso.json python run_timeiso_eval.py
GAN_DIR=gan_results_timeiso python tstr_td_features.py   # 时域特征 TSTR 复核
python attacker_power_control.py           # 同状态阳性对照
python make_fig_timeiso.py                 # 重绘图 3/4
cd ..
echo "全部完成。各案例结果见各自 results/ 目录。"
