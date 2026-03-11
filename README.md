# DDoS Attack Detection — LSTM-Based Network Traffic Analysis

> **Status: Core pipeline complete — training, evaluation, and real-time simulation fully implemented.**

---

## Table of Contents

1. [Title Justification](#title-justification)
2. [Existing System](#existing-system)
3. [Proposed System](#proposed-system)
4. [Project Overview](#project-overview)
5. [Architecture](#architecture)
6. [Repository Structure](#repository-structure)
7. [Roles & Technology Used](#roles--technology-used)
8. [Environment Setup](#environment-setup)
9. [Progress Report](#progress-report)
10. [How to Run](#how-to-run)
11. [Configuration Reference](#configuration-reference)
12. [Limitations](#limitations)
13. [Further Steps / Roadmap](#further-steps--roadmap)

---

## Title Justification

**Title:** "DDoS Attack Detection Using LSTM-Based Network Traffic Analysis"

Distributed Denial-of-Service (DDoS) attacks remain one of the most disruptive and economically damaging cyber threats. Attackers flood a target host or network segment with artificially generated traffic, exhausting bandwidth, CPU, and connection-table resources until legitimate users are denied access.

Traditional intrusion detection systems (IDS) rely on hand-crafted signature rules or shallow statistical thresholds. These approaches:
- Fail against novel or morphed attack variants
- Require constant manual rule maintenance
- Cannot model the temporal dynamics of a flow burst

Network traffic is inherently **sequential** — each flow follows another in time, and attack patterns manifest as anomalous *sequences* of flows rather than isolated packets. Long Short-Term Memory (LSTM) networks are specifically designed to learn patterns over time-ordered sequences, making them a natural architectural match for this problem.

The "LSTM-Based" qualifier in the title is therefore technically precise: the model ingests sliding windows of consecutive flow records and classifies whether the sequence represents a DDoS burst or benign traffic, leveraging temporal dependencies that static classifiers cannot capture.

---

## Existing System

Current production-grade DDoS detection systems broadly fall into three categories:

**a) Signature / Rule-Based IDS** (e.g., Snort, Suricata)
- Maintain a database of known attack patterns; generate alerts when traffic matches rules.
- **Limitations:** Zero-day attacks evade detection; rules require continuous manual updates; high false-positive rates under legitimate traffic spikes.

**b) Statistical / Threshold-Based Methods**
- Monitor metrics such as packet rate, SYN/ACK ratio, or flow duration; raise an alarm when a metric exceeds a fixed threshold.
- **Limitations:** Cannot distinguish a flash crowd from a DDoS; thresholds require per-network tuning; fail under low-and-slow attacks.

**c) Classical Machine Learning** (Random Forest, SVM, XGBoost)
- Extract hand-crafted features from flow records and train a static classifier.
- **Limitations:** Treat each flow independently — no temporal context; require expensive feature engineering; brittle across network topologies; poor handling of severe class imbalance without explicit corrections.

> None of these approaches model the *sequential, time-dependent* nature of attack traffic, leaving a fundamental capability gap.

---

## Proposed System

This project proposes an end-to-end deep learning pipeline centred on a **Bidirectional LSTM (BiLSTM) with Temporal Attention** for detecting DDoS attacks in real time.

**Core idea:** A sliding window of `SEQ_LEN` consecutive network flow records is treated as a time-series and fed to the model. The BiLSTM reads the sequence in both directions, capturing both the build-up and the tail of an attack burst. The Temporal Attention layer then learns *which timesteps* are most discriminative, producing an interpretable context vector fed to a classification head.

**System Pipeline:**

```
[Raw CSV / Synthetic Data]
        │
        ▼
[Data Cleaning & Feature Engineering]
  • Strip whitespace from column names
  • Remove NaN / Inf values
  • Select 78 CIC-standard flow features
        │
        ▼
[Label Encoding]
  • Binary : BENIGN=0 / DDoS=1
  • Multi  : up to 12 CIC-DDoS2019 sub-types (configurable)
        │
        ▼
[Train / Val / Test Split]  70% / 15% / 15% — stratified
        │
        ▼
[RobustScaler]  fitted on train only → applied to all splits
        │
        ▼
[Sliding Window Sequence Builder]  shape: (N, SEQ_LEN, n_features)
        │
        ▼  [Optional SMOTE oversampling on training sequences]
        ▼
[BiLSTM + Temporal Attention + MLP Head]
        │
        ▼
[Training]  AdamW | StepLR | AMP | Gradient Clip | Early Stopping
        │
        ▼
[Evaluation]  Accuracy | F1 | Kappa | ROC-AUC | Confusion Matrix
        │
        ▼
[Real-Time Streaming Simulation]  RealTimeDetector (circular buffer)
```

**Key innovations over the existing system:**
- **Temporal modelling** — BiLSTM captures burst dynamics across 20 timesteps
- **Interpretability** — attention weights expose which timesteps drove each prediction (explainable AI for security ops)
- **Imbalance handling** — class-weighted CrossEntropy + optional SMOTE
- **Outlier resilience** — RobustScaler instead of StandardScaler
- **Streaming inference** — circular buffer allows deployment in a live packet-capture or NetFlow collector pipeline

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

## Roles & Technology Used

| Role | Technology / Component |
|---|---|
| Programming Language | Python 3.10+ |
| Deep Learning Framework | PyTorch 2.1+ (model, training loop, AMP) |
| Model Architecture | `torch.nn.LSTM` (BiLSTM), `TemporalAttention`, `BatchNorm1d`, `LayerNorm`, GELU |
| Optimiser | `torch.optim.AdamW` |
| LR Scheduler | `torch.optim.lr_scheduler.StepLR` |
| Mixed-Precision Training | `torch.amp.GradScaler` + `torch.amp.autocast` |
| Data Manipulation | Pandas 2.0, NumPy 1.24+, PyArrow 13+ |
| Feature Scaling | scikit-learn `RobustScaler` |
| Label Encoding | scikit-learn `LabelEncoder` |
| Class Imbalance | imbalanced-learn (SMOTE) + class-weighted loss |
| Train/Val/Test Split | scikit-learn `train_test_split` (stratified) |
| Evaluation Metrics | scikit-learn (accuracy, F1, Kappa, ROC-AUC, confusion matrix) |
| Visualisation | Matplotlib 3.7+, Seaborn 0.12+ |
| Experiment Tracking | TensorBoard 2.14+ (`logs/` directory) |
| Serialisation | Joblib (scaler / encoder), `torch.save` (checkpoints) |
| Notebook Environment | Jupyter Lab (ipykernel, notebook 7+) |
| Progress / Utilities | tqdm 4.65+ |
| Target Dataset | CIC-DDoS2019 / CICIDS-2017 (or auto-generated synthetic equivalent) |

**Source file map:**

| File | Responsibility |
|---|---|
| `config.py` | All hyperparameters and directory paths |
| `src/model.py` | `DDoSLSTM`, `TemporalAttention`, `build_model()` |
| `src/preprocess.py` | `load_and_clean`, `select_features`, `make_sequences`, `oversample_sequences` |
| `src/train.py` | Standalone training script |
| `src/evaluate.py` | `plot_confusion_matrix`, `plot_roc`, `plot_precision_recall` |
| `src/utils.py` | `set_seed`, `get_device`, `predict`, `smooth`, save/load checkpoint |
| `scripts/generate_synthetic_data.py` | Synthetic CIC-schema data generator |
| `notebooks/ddos_lstm_detection.ipynb` | End-to-end 11-section walkthrough |

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
