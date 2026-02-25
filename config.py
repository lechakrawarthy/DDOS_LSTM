"""
Global configuration for DDoS Attack Detection using LSTM.
All hyperparameters, paths, and dataset settings are centralised here.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_RAW_DIR   = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR  = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
RESULTS_DIR    = ROOT_DIR / "results"
LOGS_DIR       = ROOT_DIR / "logs"

for _dir in (DATA_RAW_DIR, DATA_PROC_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Expected raw CSV file (CIC-DDoS2019 / CICIDS-2017 or similar).
# Drop this CSV into data/raw/ before running the pipeline.
RAW_CSV        = DATA_RAW_DIR / "network_traffic.csv"
PROCESSED_FILE = DATA_PROC_DIR / "processed_data.pkl"
SCALER_FILE    = DATA_PROC_DIR / "scaler.pkl"
ENCODER_FILE   = DATA_PROC_DIR / "label_encoder.pkl"

# ─── Feature Engineering ──────────────────────────────────────────────────────
# CIC-DDoS2019 column name for the label (adjust if using a different dataset)
LABEL_COLUMN   = " Label"

# Features selected after removing identifiers / constant / leaky columns
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
TEST_SIZE      = 0.20           # 80 / 20 train-test split
VAL_SIZE       = 0.10           # 10% of training set used for validation
RANDOM_STATE   = 42
SEQUENCE_LEN   = 20             # Number of consecutive flows fed to the LSTM
OVERSAMPLE     = True           # Apply SMOTE on training sequences

# ─── Model ────────────────────────────────────────────────────────────────────
INPUT_SIZE     = len(SELECTED_FEATURES)   # set dynamically in train.py as well
HIDDEN_SIZE    = 128
NUM_LAYERS     = 2
DROPOUT        = 0.3
BIDIRECTIONAL  = True           # BiLSTM for richer context

# ─── Training ─────────────────────────────────────────────────────────────────
DEVICE         = "cuda"         # "cuda" or "cpu"  (auto-detected in train.py)
EPOCHS         = 30
BATCH_SIZE     = 256
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
PATIENCE       = 5              # Early-stopping patience (epochs)
GRAD_CLIP      = 1.0            # Gradient clipping max-norm
LR_STEP_SIZE   = 10             # StepLR: decay every N epochs
LR_GAMMA       = 0.5            # StepLR: multiply LR by this factor

# ─── Evaluation ───────────────────────────────────────────────────────────────
THRESHOLD      = 0.5            # Binary decision threshold (for binary mode)
SAVE_CONFUSION = True
SAVE_CURVES    = True
