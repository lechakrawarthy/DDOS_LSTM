# DDoS Attack Detection — LSTM-Based Network Traffic Analysis

Multiclass DDoS detection using a TensorFlow/Keras LSTM trained on the
[CIC-DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html) dataset.
Detects 6 attack types + benign traffic from 75 CIC-standard flow features,
with a live packet capture module for real-time inference.

**Evaluation results (held-out test set, 40,933 samples):**

| Metric | Value |
|---|---|
| Accuracy | 83.64% |
| F1 (Weighted) | 0.80 |
| Cohen's Kappa | 0.80 |
| MCC | 0.82 |
| ROC-AUC (OvR macro) | 0.97 |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [Environment Setup](#environment-setup)
6. [How to Run](#how-to-run)
7. [Configuration Reference](#configuration-reference)
8. [Results](#results)
9. [Limitations](#limitations)
10. [Technologies Used](#technologies-used)

---

## Project Overview

This project builds an end-to-end pipeline for detecting DDoS attacks from
network flow records. The pipeline goes from raw CIC-DDoS2019 CSV files
(10.8 million rows, 4.6 GB) through preprocessing, LSTM training, evaluation,
and live packet capture inference.

**7 classes:** BENIGN, DrDoS_DNS, DrDoS_NTP, DrDoS_SSDP, Syn, UDP-lag, WebDDoS

**Why LSTM?** Network traffic is sequential — DDoS attacks manifest as
anomalous patterns across consecutive flows, not as isolated packets. LSTM
networks are designed to model exactly this kind of time-ordered dependency.

---

## Architecture

```
Input: (batch_size, timesteps=1, features=75)
        │
  LSTM Layer 1  — 128 units, return_sequences=True, dropout=0.3
        │
  LSTM Layer 2  — 64 units, dropout=0.3
        │
  Dense Layer   — 64 units, ReLU
        │
  Dropout       — rate=0.3
        │
  Output Dense  — 7 units, Softmax
        │
Prediction: class label + confidence score
```

- **Framework:** TensorFlow 2.21 / Keras 3
- **Optimizer:** Adam, sparse categorical crossentropy loss
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint, ReduceLROnPlateau
- **Saved format:** `.keras` (1.9 MB)

---

## Repository Structure

```
DDOS_LSTM/
├── config.py                    # All hyperparameters and paths
├── requirements.txt
├── build_dataset.py             # Raw CSVs → data/multiclass_dataset.csv
│
├── dataset/
│   └── raw_dataset/             # Place CIC-DDoS2019 CSVs here
│       ├── DrDoS_DNS.csv
│       ├── DrDoS_NTP.csv
│       ├── DrDoS_SSDP.csv
│       ├── Syn.csv
│       └── UDPLag.csv
│
├── data/
│   ├── multiclass_dataset.csv   # 272,884 rows × 75 features (generated)
│   └── test_dataset.csv         # 40,933 held-out test samples (generated)
│
├── model/
│   ├── multiclass_model.keras   # Trained LSTM (1.9 MB)
│   ├── multiclass_scaler.pkl    # Fitted StandardScaler (75 features)
│   └── label_encoder.pkl        # LabelEncoder (7 classes)
│
├── results/
│   ├── fig1_system_pipeline.png
│   ├── fig2_model_architecture.png
│   ├── fig3_label_distribution.png
│   ├── fig4_preprocessing_workflow.png
│   ├── fig5_training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
└── src/
    ├── model_multi_final.py     # Train LSTM → model + scaler + test set
    ├── evaluation.py            # Full metrics on test set
    ├── evaluation_vis.py        # Confusion matrix, ROC, PR curve plots
    ├── prediction.py            # Single random sample inference
    └── live_capture.py          # Real-time pyshark capture + inference
```

---

## Dataset

**CIC-DDoS2019** — Canadian Institute for Cybersecurity, University of New
Brunswick. Traffic captured 1 December 2018 in a controlled lab testbed.
CICFlowMeter computed 88 bidirectional flow features per record.

| File | Attack Type | Records |
|---|---|---|
| DrDoS_DNS.csv | DNS Amplification | 5,074,413 |
| DrDoS_NTP.csv | NTP Amplification | 1,217,007 |
| DrDoS_SSDP.csv | SSDP Amplification | 2,611,374 |
| Syn.csv | SYN Flood | 1,582,681 |
| UDPLag.csv | UDP Lag Attack | 370,605 |
| **Total** | | **10,856,080** |

BENIGN and WebDDoS labels are embedded within these files. After stratified
sampling (50,000 rows per class cap), the training dataset is 272,884 rows
across 7 classes (BENIGN: 22,445; WebDDoS: 439 — all available rows retained).

> Raw CSVs are not committed to the repository (~4.6 GB). Place them in
> `dataset/raw_dataset/` before running `build_dataset.py`.

---

## Environment Setup

```bash
# Clone repo
git clone <repo-url>
cd DDOS_LSTM

# Create virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Linux / Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**For live capture only:** Install
[Wireshark](https://www.wireshark.org/download.html) (includes tshark +
Npcap). Run the live capture script as Administrator on Windows.

---

## How to Run

Run all commands from the **repo root** with `.venv` active.

### Step 1 — Build dataset (requires raw CSVs in `dataset/raw_dataset/`)

```bash
python build_dataset.py
```

Outputs: `data/multiclass_dataset.csv`, `model/label_encoder.pkl`

### Step 2 — Train model

```bash
python src/model_multi_final.py
```

Outputs: `model/multiclass_model.keras`, `model/multiclass_scaler.pkl`,
`data/test_dataset.csv`

### Step 3 — Evaluate

```bash
python src/evaluation.py      # prints all metrics to terminal
python src/evaluation_vis.py  # saves plots to results/
```

### Step 4 — Single sample prediction

```bash
python src/prediction.py
```

### Step 5 — Live capture (requires Wireshark + run as Administrator)

```bash
# List available network interfaces
python src/live_capture.py --list-ifaces

# Capture on Wi-Fi, classify every 10 packets
python src/live_capture.py --interface "Wi-Fi" --batch 10
```

Output example:
```
[20:42:11] Window #1 (10 packets):
  [01] BENIGN  | conf=0.934
  [02] BENIGN  | conf=0.891
  ...
```

---

## Configuration Reference

All settings are in `config.py`.

| Parameter | Value | Description |
|---|---|---|
| `HIDDEN_SIZE` | 128 | Units in first LSTM layer |
| `DROPOUT` | 0.3 | Dropout rate on LSTM and Dense layers |
| `EPOCHS` | 30 | Max training epochs (early stopping applies) |
| `BATCH_SIZE` | 256 | Mini-batch size |
| `LEARNING_RATE` | 1e-3 | Initial Adam learning rate |
| `PATIENCE` | 5 | Early stopping patience |
| `TEST_SIZE` | 0.15 | Test split fraction |
| `VAL_SIZE` | 0.15 | Validation split fraction |
| `RANDOM_STATE` | 42 | Global random seed |
| `CAPTURE_BATCH_SIZE` | 10 | Packets per inference window (live capture) |

---

## Results

### Per-class performance (test set, 40,933 samples)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| BENIGN | 0.98 | 0.98 | 0.98 | 3,367 |
| DrDoS_DNS | 0.96 | 0.97 | 0.97 | 7,500 |
| DrDoS_NTP | 0.99 | 0.99 | 0.99 | 7,500 |
| DrDoS_SSDP | 0.87 | 0.98 | 0.92 | 7,500 |
| Syn | 0.60 | 0.99 | 0.75 | 7,500 |
| UDP-lag | 0.94 | 0.19 | 0.32 | 7,500 |
| WebDDoS | 0.70 | 0.24 | 0.36 | 66 |

### Output plots (`results/`)

| File | Description |
|---|---|
| `fig1_system_pipeline.png` | Full end-to-end pipeline diagram |
| `fig2_model_architecture.png` | LSTM layer-by-layer architecture |
| `fig3_label_distribution.png` | Class distribution bar chart |
| `fig4_preprocessing_workflow.png` | Step-by-step preprocessing flow |
| `fig5_training_curves.png` | Loss and accuracy vs. epoch |
| `confusion_matrix.png` | 7×7 confusion matrix heatmap |
| `roc_curve.png` | ROC curves per class (one-vs-rest) |
| `precision_recall_curve.png` | PR curves per class |

---

## Limitations

**UDP-lag (recall 0.19):** Feature overlap with Syn flows — both are
high-volume, short-duration flows. ROC-AUC for this class is still above 0.90;
the issue is at the hard decision boundary, not in the learned representations.

**WebDDoS (recall 0.24):** Only 439 total samples in the raw dataset. With 373
training examples the model defaults to BENIGN for ambiguous cases.

**Sequence length = 1:** Each flow is classified independently. The LSTM's
temporal memory across multiple consecutive flows is not exploited. A sliding
window of 20 flows would be the highest-impact improvement.

**Live capture features:** `live_capture.py` computes per-packet
approximations of CICFlowMeter flow-level statistics. Integrating
CICFlowMeter into the live path would produce exact feature matches.

---

## Technologies Used

| Category | Tool | Version |
|---|---|---|
| Deep Learning | TensorFlow / Keras | 2.21.0 |
| Data | Pandas, NumPy | ≥2.0, ≥1.24 |
| ML Utilities | scikit-learn | ≥1.3 |
| Serialisation | Joblib | ≥1.3 |
| Visualisation | Matplotlib, Seaborn | ≥3.7, ≥0.12 |
| Packet Capture | PyShark | 0.6 |
| Packet Dissection | Wireshark / tshark | 4.4.3 |
| Language | Python | 3.10+ |
| Dataset | CIC-DDoS2019 | 2019 |
