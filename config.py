"""
Global configuration for DDoS Attack Detection using LSTM.
All hyperparameters, paths, and dataset settings are centralised here.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DATASET_DIR = ROOT_DIR / "dataset" / "raw_dataset"
PROC_DATASET_DIR= ROOT_DIR / "dataset" / "processed_dataset"
MODELS_DIR      = ROOT_DIR / "model"
RESULTS_DIR     = ROOT_DIR / "results"
LOGS_DIR        = ROOT_DIR / "logs"

for _dir in (DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Dataset Files ────────────────────────────────────────────────────────────
# Raw CIC-DDoS2019 CSVs (in dataset/raw_dataset/)
RAW_CSVS = [
    RAW_DATASET_DIR / "DrDoS_DNS.csv",
    RAW_DATASET_DIR / "DrDoS_NTP.csv",
    RAW_DATASET_DIR / "DrDoS_SSDP.csv",
    RAW_DATASET_DIR / "Syn.csv",
    RAW_DATASET_DIR / "UDPLag.csv",
]

# Intermediate & final processed files (in data/)
MULTICLASS_DATASET = DATA_DIR / "multiclass_dataset.csv"
TEST_DATASET       = DATA_DIR / "test_dataset.csv"

# ─── Model Artifact Files ─────────────────────────────────────────────────────
MODEL_FILE   = MODELS_DIR / "multiclass_model.keras"
SCALER_FILE  = MODELS_DIR / "multiclass_scaler.pkl"
ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"

# ─── Feature Engineering ──────────────────────────────────────────────────────
# CIC-DDoS2019 label column (leading space is intentional – matches raw header)
LABEL_COLUMN = " Label"

# 78 features selected after removing identifiers / constant / leaky columns
SELECTED_FEATURES = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    " Flow Bytes/s",
    " Flow Packets/s",
    " Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Total",
    " Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Total",
    " Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    "Fwd PSH Flags",
    " Fwd URG Flags",
    " Fwd Header Length",
    " Bwd Header Length",
    "Fwd Packets/s",
    " Bwd Packets/s",
    " Min Packet Length",
    " Max Packet Length",
    " Packet Length Mean",
    " Packet Length Std",
    " Packet Length Variance",
    "FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    " CWE Flag Count",
    " ECE Flag Count",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    " Fwd Avg Packets/Bulk",
    " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk",
    " Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Packets",
    " Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    " Init_Win_bytes_backward",
    " act_data_pkt_fwd",
    " min_seg_size_forward",
    "Active Mean",
    " Active Std",
    " Active Max",
    " Active Min",
    "Idle Mean",
    " Idle Std",
    " Idle Max",
    " Idle Min",
]

# ─── Preprocessing ────────────────────────────────────────────────────────────
TEST_SIZE    = 0.15          # 70 / 15 / 15 train-val-test split
VAL_SIZE     = 0.15
RANDOM_STATE = 42
OVERSAMPLE   = False         # Majority classes capped at 50k in build_dataset.py; SMOTE not used

# ─── Model ────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3

# ─── Training ─────────────────────────────────────────────────────────────────
EPOCHS       = 30
BATCH_SIZE   = 256
LEARNING_RATE= 1e-3
PATIENCE     = 5             # Early-stopping patience (epochs without improvement)

# ─── Evaluation ───────────────────────────────────────────────────────────────
SAVE_CONFUSION = True
SAVE_CURVES    = True

# ─── Live Capture (pyshark) ───────────────────────────────────────────────────
# Network interface to capture on.  Set to None to auto-detect.
CAPTURE_INTERFACE = None
# Number of packets to collect before running one inference pass.
CAPTURE_BATCH_SIZE = 10
