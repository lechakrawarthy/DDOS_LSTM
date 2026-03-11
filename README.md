# DDoS Attack Detection — LSTM-Based Network Traffic Analysis

> **Status: Core pipeline complete — training, evaluation, and real-time simulation fully implemented.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Progress Report](#progress-report)
6. [How to Run](#how-to-run)
7. [Configuration Reference](#configuration-reference)
8. [Further Steps / Roadmap](#further-steps--roadmap)

---

## Project Overview

Binary and multi-class classification of network traffic flows as **Benign** or **DDoS** using a Bidirectional LSTM with temporal attention.

**Target Dataset:** [CIC-DDoS2019 / CICIDS-2017](https://www.unb.ca/cic/datasets/ddos-2019.html)  
A synthetic dataset with the same schema is auto-generated when the real CSV is absent.

**Key design principles:**
- Class-weighted loss to handle severe label imbalance
- RobustScaler to neutralise outlier-driven feature dominance
- Temporal attention to highlight which timesteps drive each prediction
- Optional SMOTE oversampling for minority-class augmentation
- Mixed-precision training (AMP) + gradient clipping for stability
- Early stopping + best-checkpoint saving

---

## Architecture

```
Input (batch, seq_len, n_features)
       │
[Optional] Input Projection  (Linear → LayerNorm → GELU)
       │
BiLSTM — stacked, N layers, dropout between layers
       │
Temporal Attention  (soft attention → context vector)
       │
MLP Head  (Linear → BN → GELU → Dropout → Linear)
       │
Output logits  (n_classes)
```

| Component | Detail |
|---|---|
| LSTM type | Bidirectional, stacked |
| Hidden size | 128 (per direction, configurable) |
| Layers | 2 (configurable) |
| Attention | Soft temporal (Bahdanau-style) |
| Optimizer | AdamW |
| Scheduler | StepLR |
| Loss | CrossEntropyLoss (class-weighted) |
| Sequence length | 20 timesteps (sliding window) |

---

## Repository Structure

```
DDOS_LSTM/
├── config.py                        # All hyperparameters and paths
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── network_traffic.csv      # Place CIC-DDoS2019 CSV here
│   └── processed/                   # Scaler, encoder, processed pkl (auto-generated)
│
├── models/                          # Saved checkpoints (.pt files)
├── results/                         # Evaluation plots and reports
├── logs/                            # TensorBoard logs
│
├── notebooks/
│   └── ddos_lstm_detection.ipynb    # Full end-to-end pipeline notebook
│
├── scripts/
│   └── generate_synthetic_data.py   # Generates synthetic CIC-schema data
│
└── src/
    ├── __init__.py
    ├── model.py        # DDoSLSTM, TemporalAttention, build_model()
    ├── preprocess.py   # load_and_clean, select_features, make_sequences, etc.
    ├── train.py        # Standalone training script
    ├── evaluate.py     # plot_confusion_matrix, plot_roc, plot_precision_recall
    └── utils.py        # set_seed, get_device, predict, smooth, checkpointing
```

---

## Environment Setup

```powershell
# Create and activate virtual environment (example with virtualenvwrapper)
mkvirtualenv ddos-lstm
workon ddos-lstm               # or: & .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**Python:** 3.10+  **PyTorch:** 2.1+  **CUDA:** optional but recommended

---

## Progress Report

### ✅ Completed

| # | Component | Status | Notes |
|---|---|---|---|
| 1 | Project scaffolding | Done | `src/`, `scripts/`, `config.py`, `requirements.txt` |
| 2 | Synthetic data generator | Done | `scripts/generate_synthetic_data.py` — CIC-DDoS2019 schema |
| 3 | Data loading & EDA | Done | Class distribution, missing-value check, dtypes |
| 4 | Preprocessing pipeline | Done | `src/preprocess.py` — clean, select features, encode labels |
| 5 | Feature engineering | Done | 78-feature CIC column set, variance-based importance plot |
| 6 | Train/Val/Test split | Done | Stratified, 70/15/15, `split_data()` |
| 7 | Scaling | Done | `RobustScaler`, fitted on train only, applied to all splits |
| 8 | Sequence generation | Done | Sliding window → `(N, SEQ_LEN, n_features)` tensors |
| 9 | SMOTE oversampling | Done | Optional via `cfg.OVERSAMPLE`, `oversample_sequences()` |
| 10 | LSTM model | Done | `DDoSLSTM` — BiLSTM + temporal attention + MLP head |
| 11 | Training loop | Done | AMP, gradient clipping, early stopping, checkpoint saving |
| 12 | Evaluation metrics | Done | Accuracy, F1, Cohen's Kappa, ROC-AUC, classification report |
| 13 | Visualisations | Done | Loss/accuracy curves, confusion matrices, ROC, PR curve |
| 14 | Real-time simulation | Done | `RealTimeDetector` — circular buffer streaming inference |
| 15 | Attention visualisation | Done | Per-timestep attention weights plotted for DDoS samples |
| 16 | Notebook | Done | `ddos_lstm_detection.ipynb` — 11-section end-to-end walkthrough |

---

## How to Run

### Option A — Notebook (recommended for exploration)

```powershell
cd notebooks
jupyter lab ddos_lstm_detection.ipynb
```

Run cells top-to-bottom. Synthetic data is auto-generated if no real CSV is present.

### Option B — Standalone script

```powershell
# From repo root
python src/train.py
```

---

## Configuration Reference

All settings live in `config.py`. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `SEQUENCE_LEN` | 20 | Sliding window size (timesteps per sample) |
| `HIDDEN_SIZE` | 128 | LSTM hidden units per direction |
| `NUM_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.3 | Dropout probability |
| `BIDIRECTIONAL` | True | Use BiLSTM |
| `EPOCHS` | 50 | Max training epochs |
| `BATCH_SIZE` | 256 | Mini-batch size |
| `LEARNING_RATE` | 1e-3 | AdamW initial LR |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `LR_STEP_SIZE` | 10 | StepLR step (epochs) |
| `LR_GAMMA` | 0.5 | StepLR decay factor |
| `PATIENCE` | 10 | Early stopping patience |
| `GRAD_CLIP` | 1.0 | Gradient norm clip |
| `OVERSAMPLE` | False | Enable SMOTE on training set |
| `RANDOM_STATE` | 42 | Global seed |

---

## Further Steps / Roadmap

### High Priority

- [ ] **Run with real CIC-DDoS2019 data** — download the dataset, place `network_traffic.csv` in `data/raw/`, and re-run the notebook to validate performance on real traffic.
- [ ] **Multi-class classification** — the model already supports `n_classes > 2`; set `binary=False` in `encode_labels()` and test against all 12 attack sub-types in CIC-DDoS2019.
- [ ] **Hyperparameter tuning** — grid/random search over `HIDDEN_SIZE`, `NUM_LAYERS`, `SEQUENCE_LEN`, `DROPOUT`. Consider using `optuna` for Bayesian optimisation.
- [ ] **TensorBoard integration** — `LOGS_DIR` is already created; wire up `SummaryWriter` in `src/train.py` to log loss, accuracy, and LR curves.

### Model Improvements

- [ ] **Transformer / TCN baseline** — compare BiLSTM against a Temporal Convolutional Network or a lightweight Transformer encoder to quantify the value of the recurrent architecture.
- [ ] **Threshold tuning** — the current decision threshold is 0.5; sweep thresholds on the val set to optimise F1 or a custom cost metric (false negatives are more costly than false positives in DDoS detection).
- [ ] **Export & deployment** — export trained model to ONNX (`torch.onnx.export`) or TorchScript for embedding in a network monitoring daemon or packet capture pipeline.
- [ ] **Quantisation / pruning** — INT8 post-training quantisation to reduce inference latency for real-time use.

### Engineering & Ops

- [ ] **`src/train.py` CLI flags** — wire `argparse` so `--epochs`, `--lr`, `--seq-len`, etc. can be overridden without editing `config.py`.
- [ ] **Unit tests** — add `tests/` with pytest coverage for `preprocess.py` (sequence shapes, scaler leakage check) and `model.py` (forward pass output shape, attention sum-to-one).
- [ ] **CI pipeline** — GitHub Actions workflow: lint (`ruff`), type-check (`mypy`), run unit tests on CPU.
- [ ] **Results persistence** — save final metrics JSON to `results/metrics.json` at end of training so runs are reproducible and comparable.
- [ ] **`results/` report** — auto-save all evaluation plots (currently displayed inline) to `results/` as PNG files using `src/evaluate.py` helpers.

### Data & Generalisation

- [ ] **Cross-dataset validation** — train on CIC-DDoS2019, test on CICIDS-2017 (or vice versa) to assess generalisation across network environments.
- [ ] **Feature ablation study** — systematically remove feature groups (IAT, flag counts, bulk rate, etc.) to identify the most discriminative subsets and reduce inference-time dimensionality.
- [ ] **Concept drift simulation** — insert temporal distribution shift mid-stream in the real-time simulation to test detector robustness and explore online adaptation strategies.
