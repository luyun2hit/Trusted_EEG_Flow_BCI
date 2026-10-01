# Trusted EEG Flow BCI — Companion Code Package

Companion code for the book chapter **"Trusted Flow of EEG Data in Brain-Computer
Interfaces: Three Privacy-Preserving Pathways and Verifiable Practices"**.

The chapter studies which artifacts may cross trust boundaries in EEG data flows
and how privacy claims can be *verified* rather than assumed. This repository
contains three self-contained, reproducible case studies — user-wise
perturbation, federated learning, and synthetic EEG release — together with the
result files produced by our own runs (under `results/`), so every number
reported in the chapter can be re-generated or inspected directly.

[中文完整说明见 README_zh.md](README_zh.md)

## Repository layout

```
Trusted_EEG_Flow_BCI/
├── README.md / README_zh.md         This file / full Chinese documentation
├── LICENSE                          MIT + upstream license notes
├── requirements.txt                 Pinned dependency versions (tested)
├── run_all.sh / run_all.bat         One-click reproduction of all three cases
├── screenshots/                     Run screenshots of the three cases and the environment
├── case1_user_wise_perturbations/   Case 1: pre-release user-identity perturbation
│   ├── README.md                    Detailed documentation (Chinese) and scope of interpretation
│   ├── download.py                  eegmmidb left/right-hand MI subset downloader (resumable)
│   ├── preprocess_mi2.py            Band-pass + epoching + per-run channel standardization
│   ├── eeg_models.py                Independent re-implementation of EEGNet / ShallowConvNet / heads
│   ├── run_perturbations.py         Four perturbations, task/UID evaluation (config-hash cache, resumable)
│   ├── make_fig.py                  Redraws Figure 1 (mean ± SD)
│   └── results/                     Measured results (CSV / figure)
├── case2_fedeeg/                    Case 2: federated averaging (FedAvg)
│   ├── README.md                    Protocol, corrections, license notes (Chinese)
│   ├── fedegg_reimpl.py             Protocol-matched PyTorch re-implementation (LSTM + FedAvg)
│   ├── membership_inference.py      Loss-based membership-inference spot check
│   ├── local_only_baseline.py       Local-only baseline (matched data-pass budget)
│   ├── sweep.py / final_run.py      Round/epoch/lr screening and the tuned 24-round configuration
│   ├── valsplit_run.py              Validation-split protocol (60:20:20) — the reported configuration
│   ├── merge_results.py / merge_tuned.py / merge_valsplit.py / mia_merge.py
│   ├── make_fig.py                  Redraws Figure 2
│   └── results/                     Per-round CSV, summary JSON, MIA JSON, per-seed partials
└── case3_epilepsygan_chbmit/        Case 3: synthetic ictal EEG release (EpilepsyGAN @ CHB-MIT)
    ├── README.md                    Detailed documentation, failure modes, trade-off table (Chinese)
    ├── download_segments.py         Segment-level downloader (HTTP Range; ~60 MB instead of 40 GB+)
    ├── preprocess.py                4-second windowing (ictal 3 s overlap / inter-ictal)
    ├── epilepsygan_chbmit.py        Conditional GAN, leave-one-patient-out training (PyTorch)
    ├── evaluate_chbmit.py           TSTR utility + re-identification + nearest-neighbor audit + spectra
    ├── attacker_power_control.py    Adversary-power positive control (same-state, non-overlapping)
    ├── regen_timeiso.py             Temporal-isolation regeneration (conditioning windows from training period)
    ├── run_timeiso_eval.py / tstr_td_features.py
    ├── gan_results/                 Trained leave-one-out generators + per-fold manifests (v5)
    ├── gan_results_timeiso/         Final release configuration: synthetic ictal windows (4 folds)
    ├── gan_results_timeiso_noadv/   Same protocol without debiasing (control)
    └── results/                     summary_timeiso.json (chapter values), training logs, per-seed UID logs, figures
```

All three cases are **independent re-implementations**; no upstream source files
are bundled. Upstream papers, repositories, and datasets are referenced in each
case README and must be used under their own licenses (see `LICENSE`).

## Environment

| | Cases 1 & 2 | Case 3 |
|---|---|---|
| Python | 3.12 (tested) | 3.11 (tested) |
| PyTorch | 2.9.1+cpu | 2.11.0+cu126 (GPU); CPU also works, only slower |
| Key deps | numpy, scipy, scikit-learn 1.9, pyedflib, matplotlib, pandas | same + PyWavelets 1.8 |
| Hardware tested | single CPU host (Windows 11) | single NVIDIA GTX 1660 SUPER |

Install: `pip install -r requirements.txt`

Approximate end-to-end runtimes (as run for the chapter): Case 1 ≈ 2.5 h (CPU),
Case 2 ≈ 1.5 h cumulative across configurations (CPU), Case 3 ≈ 5 min GAN
training + 15 min evaluation (GPU; CPU ≈ 1.5 h estimated for training).

## Quick start

```bash
pip install -r requirements.txt
bash run_all.sh        # Windows CMD: run_all.bat
```

Case 2 additionally needs the upstream FedEEG dataset (distributed with the
upstream repository, MIT license):

```bash
git clone https://github.com/AmanPriyanshu/FedEEG.git
export FEDEEG_DATA=/path/to/FedEEG-main/dataset_hand_movement   # Windows: set FEDEEG_DATA=...
```

## Case 1 — User-wise perturbation (eegmmidb)

Independent re-implementation of the four user-wise perturbations
(RAND / SN / EMIN / EMAX) from the algorithmic description of Chen et al.
(J. Neural Eng. 22(1):016040, 2025), evaluated on the left/right-hand
motor-imagery subset of PhysioNet eegmmidb (36 subjects; train on R04, test on
R08+R12; only the released/training side is perturbed).

```bash
cd case1_user_wise_perturbations
python download.py            # ~270 MB, resumable
python preprocess_mi2.py      # band-pass [4,32] Hz, post-cue [0,4] s, per-run standardization
python run_perturbations.py   # 4 perturbations x 5 seeds + cross-architecture attacker + time-shift checks
python make_fig.py            # Figure 1
```

Expected results (`results/perturbation_results.csv`, mean ± SD over 5 seeds;
UID chance level = 1/36 ≈ 2.78%):

| Perturbation | Task BCA | UID BCA (EEGNet) | UID BCA (ShallowConvNet) |
|---|---|---|---|
| None | 56.40 ± 1.50% | 42.31 ± 2.91% | 31.61 ± 1.93% |
| RAND | 54.86 ± 1.07% | 3.46 ± 0.80% | 5.07 ± 0.90% |
| SN   | 53.91 ± 1.33% | 3.11 ± 0.60% | 3.63 ± 0.79% |
| EMIN | 54.94 ± 1.90% | 2.72 ± 0.08% | 3.61 ± 0.69% |
| EMAX | 56.49 ± 2.89% | 3.37 ± 0.52% | 4.07 ± 0.73% |

Scope: the protocol verifies that identity linkage from released (perturbed)
records to clean cross-block records is suppressed to the chance level; it is
not full de-identification. See `case1_user_wise_perturbations/README.md` and
Section 4.1 of the chapter.

## Case 2 — Federated averaging (FedEEG, PyTorch re-implementation)

The original FedEEG implementation (Priyanshu, 2021) targets TensorFlow 1.x and
no longer runs on current stacks; this case re-implements the protocol line by
line in PyTorch with two protocol corrections: (1) normalization statistics are
computed by each client from its own training split only (the original used
full-dataset statistics from user_a); (2) the aggregated global model is
evaluated after each round (the original tested pre-aggregation local models).

```bash
cd case2_fedeeg
python fedegg_reimpl.py        # original 5-round configuration (or per-seed: 42 / 7 / 2024)
python merge_results.py && python make_fig.py
python membership_inference.py && python mia_merge.py     # privacy spot check
python valsplit_run.py sweep 42                            # round screening on validation only
python valsplit_run.py fed 42 27 && ...                    # six arms x 3 seeds (see README)
python merge_valsplit.py
```

Expected results (3 seeds; reported configuration: 27 rounds x 3 local epochs,
60:20:20 train/val/test, rounds selected on the validation set only):

- Federated global model **47.90% ± 1.0%**; matched-budget centralized baseline
  **54.92% ± 0.8%**; local-only baseline **63.79% ± 0.5%** (federated–centralized
  gap ≈ 7.0 pp; ranking unchanged from the original 5-round configuration).
- Membership-inference spot check (loss-based, Yeom-style AUC): under-trained
  5-round configuration stays at chance (final round 0.506 ± 0.001 — a negative
  result consistent with underfitting, not privacy evidence); the 27-round
  configuration reaches **0.547 ± 0.007**, below the matched-budget centralized
  model's **0.651 ± 0.007**. The two models differ in fitting level, so the AUC
  difference cannot be attributed to aggregation alone (Section 4.2).

## Case 3 — Synthetic ictal EEG (EpilepsyGAN @ CHB-MIT)

End-to-end port of EpilepsyGAN (Pascual et al., 2021) from the paid EPILEPSIAE
database to the public CHB-MIT Scalp EEG database, re-implemented in PyTorch:
a conditional GAN generates seizure windows conditioned on inter-ictal windows,
trained leave-one-patient-out over 4 subjects, with identity-adversarial
debiasing (gradient-reversal UID head, λ_adv = 2.0).

```bash
cd case3_epilepsygan_chbmit
python download_segments.py     # ~60 MB via HTTP Range, resumable
python preprocess.py            # 4-s windows
python epilepsygan_chbmit.py    # 4-fold LOPO training (GPU ≈ 5 min; DEVICE=cpu to force CPU)
                                # per-fold manifests bind config/data/output hashes
python regen_timeiso.py         # temporal-isolation regeneration (final configuration)
# temporal-isolation evaluation — the chapter's final numbers:
GAN_DIR=gan_results_timeiso MODE=all OUT_JSON=results/summary_timeiso.json python run_timeiso_eval.py
GAN_DIR=gan_results_timeiso python tstr_td_features.py   # temporal-feature TSTR cross-check
python attacker_power_control.py  # same-state adversary-power control
python make_fig_timeiso.py        # redraw Figures 3/4
```

Final results (temporal-isolation protocol, v5 corrected implementation;
`results/summary_timeiso.json`):

- Utility (TSTR, G-mean, Welch band-power features): real-data baseline
  **83.94%** vs. debiased synthetic **69.12%**; the no-debiasing control
  (λ_adv = 0) reaches **82.33%** → debiasing costs ≈ 13.2 pp. A temporal/
  nonlinear-feature cross-check (20 dims) gives 63.78% vs. 85.57%.
- Privacy (patient re-identification, balanced accuracy, chance = 25%,
  5 attacker-initialization seeds): synthetic ictal **49.90% ± 0.11%
  (2.00× chance)** with debiasing vs. **47.73% ± 1.21% (1.91×)** without and
  47.68% ± 0.14% (1.91×, 3 seeds) at λ_adv = 5 — gradient-reversal debiasing
  does not reduce re-identification under the corrected implementation. The
  cross-state control (real ictal, 18.53% ± 3.38%, 0.74× chance) does not
  exceed chance, so privacy conclusions are limited to the synthetic-vs-real
  relative comparison; because each patient's synthetic data come from a
  different leave-one-out fold, the signal is reported as a fold-associated
  re-identification signal (see Section 4.3.3).
- Adversary-power control (same-state): inter-ictal→inter-ictal 56.30% ± 4.55%,
  ictal→ictal 68.65% ± 1.39% (2.25×/2.75× chance) — the adversary can learn
  identity within each state; this does not by itself explain the below-chance
  cross-state figure.
- Nearest-neighbor memorization audit: synthetic/real median-distance ratio
  0.93–1.10 ≈ 1, near-duplicate rate 0 — no per-sample memorization observed
  under this metric.
- Spectral cosine similarity: real–real 0.653–0.737 vs. real–synthetic
  0.606–0.737 (intervals largely overlap; population-level plausibility only).

The `case3.../README.md` documents the four failure modes found while building
this pipeline (sigmoid-wired LSGAN discriminator collapse, random-pairing L1
median regression, specificity inflation from reusing inter-ictal windows, and
a fold-associated re-identification signal carried through the conditional
pathway that gradient-reversal debiasing failed to remove) — all discussed in
Section 4.3.

## Figure and table source data

| Chapter item | Source file(s) in this repository |
|---|---|
| Figure 1 | `case1_user_wise_perturbations/results/perturbation_results.csv` (redraw with `make_fig.py`) |
| Figure 2 | `case2_fedeeg/results/federated_rounds.csv`, `results/fig2_curve_data.json` |
| Figure 3 | `case3_epilepsygan_chbmit/results/fig_waveforms_source.npz` |
| Figure 4 | `case3_epilepsygan_chbmit/results/nnsim_timeiso.json` |
| Case 3 final results | `case3_epilepsygan_chbmit/results/summary_timeiso.json` |

## Datasets

| Dataset | Used in | Access |
|---|---|---|
| PhysioNet eegmmidb 1.0.0 (ODC-BY 1.0) | Case 1 | https://physionet.org/content/eegmmidb/1.0.0/ (script downloads automatically) |
| EEG data from hands movement | Case 2 | Distributed with the FedEEG repository (MIT) |
| CHB-MIT Scalp EEG 1.0.0 (ODC-BY 1.0) | Case 3 | https://physionet.org/content/chbmit/1.0.0/ (script downloads selected segments) |

No dataset is redistributed in this repository.

## License and citation

All scripts in this package are written by the chapter authors and released
under the MIT License; upstream resources keep their own licenses (see the
table in `LICENSE`). If you use this package, please cite the chapter and the
three original papers (Chen et al., J. Neural Eng. 2025; Priyanshu, 2021;
Pascual et al., 2021).
