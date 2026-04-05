"""
Legacy preprocessing utilities (not the active pipeline).

The active dataset build script is build_dataset.py (repo root).
build_dataset.py uses 75 confirmed features, stratified sampling to 50k rows
per class, and does not apply SMOTE.

This file is retained for reference only.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

import config

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _load_csv(path: "Path") -> pd.DataFrame:
    """Load a CIC-DDoS2019 CSV, strip column whitespace, keep needed columns."""
    print(f"  Loading {path.name} ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Rename stripped label column back to the config name (stripped)
    label_stripped = config.LABEL_COLUMN.strip()
    if label_stripped in df.columns:
        df.rename(columns={label_stripped: config.LABEL_COLUMN.strip()}, inplace=True)

    # Keep only selected features + label
    label_col = config.LABEL_COLUMN.strip()
    feature_cols = [f.strip() for f in config.SELECTED_FEATURES]

    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"    WARNING: {len(missing)} feature(s) not found in {path.name}: {missing}")

    cols_to_keep = available_features + ([label_col] if label_col in df.columns else [])
    return df[cols_to_keep]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Verify raw CSVs exist
    available = [p for p in config.RAW_CSVS if p.exists()]
    if not available:
        raise FileNotFoundError(
            f"No raw CSVs found in {config.RAW_DATASET_DIR}. "
            "Place the CIC-DDoS2019 CSV files there and retry."
        )
    print(f"Found {len(available)} raw CSV file(s).")

    # ── Load & Concatenate ──────────────────────────────────────────────────
    frames = [_load_csv(p) for p in available]
    df = pd.concat(frames, ignore_index=True)
    print(f"\nCombined shape before cleaning: {df.shape}")

    # ── Clean Column Names ─────────────────────────────────────────────────
    df.columns = df.columns.str.strip()
    label_col  = config.LABEL_COLUMN.strip()

    if label_col not in df.columns:
        raise KeyError(
            f"Label column '{label_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    # ── Drop Inf / NaN ─────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != label_col]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN/Inf values.")

    # ── Label Distribution ──────────────────────────────────────────────────
    print("\nLabel distribution (raw):")
    print(df[label_col].value_counts())

    # ── Encode Labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    df["Label"] = le.fit_transform(df[label_col])
    if label_col != "Label":
        df.drop(columns=[label_col], inplace=True)

    joblib.dump(le, config.ENCODER_FILE)
    print(f"\nLabel encoder saved to {config.ENCODER_FILE}")
    print("Class mapping:")
    for i, cls in enumerate(le.classes_):
        print(f"  {i}: {cls}")

    # ── Optional SMOTE ─────────────────────────────────────────────────────
    if config.OVERSAMPLE:
        try:
            from imblearn.over_sampling import SMOTE
            X = df.drop("Label", axis=1).values
            y = df["Label"].values
            print(f"\nApplying SMOTE (original size: {len(y)})...")
            sm = SMOTE(random_state=config.RANDOM_STATE)
            X_res, y_res = sm.fit_resample(X, y)
            df = pd.DataFrame(X_res, columns=df.drop("Label", axis=1).columns)
            df["Label"] = y_res
            print(f"After SMOTE: {df.shape}")
        except ImportError:
            print(
                "WARNING: imbalanced-learn not installed. Skipping SMOTE.\n"
                "         Install with:  pip install imbalanced-learn"
            )

    # ── Save Final Dataset ─────────────────────────────────────────────────
    df.to_csv(config.MULTICLASS_DATASET, index=False)
    print(f"\nMulticlass dataset saved to {config.MULTICLASS_DATASET}")
    print(f"Final shape : {df.shape}")
    print("\nLabel distribution (encoded):")
    print(df["Label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
