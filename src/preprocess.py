"""
src/preprocess.py

End-to-end preprocessing pipeline for network-traffic CSV files
(CIC-DDoS2019 / CICIDS-2017 format).

Steps
-----
1. Load & clean raw CSV (handle inf / NaN, remove constant columns).
2. Encode class labels (binary: BENIGN vs. DDoS, or multi-class).
3. Stratified train / val / test split.
4. Robust / Standard scaling.
5. Sliding-window sequence construction for LSTM.
6. Optional SMOTE oversampling on training sequences.
7. Serialise processed artefacts to data/processed/.
"""

import config as cfg
import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Allow running this file directly from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─── Step 1: Load & Clean ─────────────────────────────────────────────────────

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV, strip column names, drop inf/NaN rows."""
    logger.info(f"Loading dataset from {csv_path} …")
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    logger.info(f"Raw shape: {df.shape}")

    # Replace ±inf with NaN then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    logger.info(f"Dropped {before - len(df)} rows with NaN/Inf.")

    # Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(f"Dropped {before - len(df)} duplicate rows.")

    # Strip leading/trailing spaces from the label column
    label_col = cfg.LABEL_COLUMN.strip()
    if label_col in df.columns:
        df[label_col] = df[label_col].str.strip()
    elif cfg.LABEL_COLUMN in df.columns:
        df[cfg.LABEL_COLUMN] = df[cfg.LABEL_COLUMN].str.strip()

    return df


# ─── Step 2: Feature Selection & Encoding ─────────────────────────────────────

def encode_labels(df: pd.DataFrame, binary: bool = True) -> tuple[np.ndarray, LabelEncoder]:
    """
    Extract & encode the label column.

    binary=True  → BENIGN=0, everything else=1
    binary=False → multi-class integer encoding
    """
    label_col = cfg.LABEL_COLUMN.strip()
    raw_labels = df[label_col].values

    le = LabelEncoder()
    if binary:
        binary_labels = np.where(raw_labels == "BENIGN", 0, 1)
        le.classes_ = np.array(["BENIGN", "DDoS"])
        logger.info(f"Binary labels  → BENIGN: {(binary_labels == 0).sum()}, "
                    f"DDoS: {(binary_labels == 1).sum()}")
        return binary_labels.astype(np.int64), le

    encoded = le.fit_transform(raw_labels)
    for cls, idx in zip(le.classes_, range(len(le.classes_))):
        logger.info(
            f"  Class {idx}: {cls}  ({(encoded == idx).sum()} samples)")
    return encoded.astype(np.int64), le


def select_features(df: pd.DataFrame) -> np.ndarray:
    """Return a numpy array of the chosen feature columns."""
    # Keep only columns that are actually present in the dataframe
    available = [c.strip() for c in df.columns]
    selected = []
    missing = []
    for feat in cfg.SELECTED_FEATURES:
        name = feat.strip()
        if name in available:
            selected.append(name)
        else:
            missing.append(name)

    if missing:
        logger.warning(f"{len(missing)} configured features not found in CSV "
                       f"and will be skipped: {missing[:5]} …")

    if not selected:
        raise ValueError(
            "No features matched. Check SELECTED_FEATURES in config.py.")

    logger.info(f"Using {len(selected)} features.")
    return df[selected].values.astype(np.float32), selected


# ─── Step 3: Train / Val / Test Split ─────────────────────────────────────────

def split_data(X: np.ndarray, y: np.ndarray):
    """Stratified 80/10/10 split."""
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )
    val_ratio = cfg.VAL_SIZE / (1 - cfg.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=cfg.RANDOM_STATE,
    )
    logger.info(
        f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Step 4: Scaling ──────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def scale(scaler: RobustScaler, *arrays: np.ndarray):
    return [scaler.transform(arr) for arr in arrays]


# ─── Step 5: Sliding-Window Sequences ─────────────────────────────────────────

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = cfg.SEQUENCE_LEN):
    """
    Build overlapping windows of length `seq_len`.

    Returns
    -------
    X_seq : (N, seq_len, n_features)
    y_seq : (N,)  – label of the LAST timestep in each window
    """
    n_samples = len(X) - seq_len + 1
    n_features = X.shape[1]

    X_seq = np.lib.stride_tricks.sliding_window_view(X, (seq_len, n_features))
    X_seq = X_seq.reshape(-1, seq_len, n_features)   # (N, T, F)
    y_seq = y[seq_len - 1:]                           # label of last frame

    return X_seq.astype(np.float32), y_seq.astype(np.int64)


# ─── Step 6: SMOTE Oversampling ───────────────────────────────────────────────

def oversample_sequences(X_seq: np.ndarray, y_seq: np.ndarray):
    """
    Flatten sequences → SMOTE → reshape back.
    Only applied when minority class ratio < 0.4.
    """
    n, t, f = X_seq.shape
    X_flat = X_seq.reshape(n, t * f)

    unique, counts = np.unique(y_seq, return_counts=True)
    ratio = counts.min() / counts.max()
    if ratio >= 0.4:
        logger.info(f"Class ratio {ratio:.2f} is balanced – skipping SMOTE.")
        return X_seq, y_seq

    logger.info(f"Applying SMOTE  (minority ratio = {ratio:.2f}) …")
    sm = SMOTE(random_state=cfg.RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_flat, y_seq)
    X_res = X_res.reshape(-1, t, f)
    logger.info(f"After SMOTE: {X_res.shape[0]} training sequences.")
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ─── Step 7: Serialise ────────────────────────────────────────────────────────

def save_artefacts(data_dict: dict, scaler, label_encoder, feature_names):
    """Persist processed data and pipeline objects."""
    import pickle
    cfg.DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Processed tensors
    with open(cfg.PROCESSED_FILE, "wb") as f:
        pickle.dump(data_dict, f)
    # Scaler
    joblib.dump(scaler, cfg.SCALER_FILE)
    # Label encoder
    joblib.dump(label_encoder, cfg.ENCODER_FILE)
    # Feature names
    feat_path = cfg.DATA_PROC_DIR / "feature_names.pkl"
    joblib.dump(feature_names, feat_path)

    logger.info(f"Artefacts saved to {cfg.DATA_PROC_DIR}")


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline(csv_path: Path = cfg.RAW_CSV, binary: bool = True):
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = load_and_clean(csv_path)
    X, feature_names = select_features(df)
    y, le = encode_labels(df, binary=binary)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    scaler = fit_scaler(X_train)
    X_train, X_val, X_test = scale(scaler, X_train, X_val, X_test)

    logger.info("Building sliding-window sequences …")
    X_train_s, y_train_s = make_sequences(X_train, y_train)
    X_val_s,   y_val_s = make_sequences(X_val,   y_val)
    X_test_s,  y_test_s = make_sequences(X_test,  y_test)

    if cfg.OVERSAMPLE:
        X_train_s, y_train_s = oversample_sequences(X_train_s, y_train_s)

    data = dict(
        X_train=X_train_s, y_train=y_train_s,
        X_val=X_val_s,     y_val=y_val_s,
        X_test=X_test_s,   y_test=y_test_s,
        n_classes=len(le.classes_),
        class_names=list(le.classes_),
        n_features=X_train_s.shape[2],
    )

    save_artefacts(data, scaler, le, feature_names)
    logger.info("Preprocessing complete.")
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Preprocess network traffic CSV.")
    parser.add_argument("--csv",    type=str, default=str(cfg.RAW_CSV),
                        help="Path to raw CSV file.")
    parser.add_argument("--multi",  action="store_true",
                        help="Use multi-class labels instead of binary.")
    args = parser.parse_args()
    run_pipeline(Path(args.csv), binary=not args.multi)
