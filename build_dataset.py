"""
One-shot script: raw CSVs -> data/multiclass_dataset.csv + model/label_encoder.pkl

Reads all 5 CIC-DDoS2019 raw CSVs, selects the 75 good features, encodes labels,
takes a stratified sample so the file is manageable, and saves the result.

Run from repo root:  python build_dataset.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RAW_FILES = [
    "dataset/raw_dataset/DrDoS_DNS.csv",
    "dataset/raw_dataset/DrDoS_NTP.csv",
    "dataset/raw_dataset/DrDoS_SSDP.csv",
    "dataset/raw_dataset/Syn.csv",
    "dataset/raw_dataset/UDPLag.csv",
]

# 75 features confirmed from git history (feature_names.pkl)
FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Fwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# Max rows per class to keep dataset manageable (~300k total)
ROWS_PER_CLASS = 50_000
RANDOM_STATE   = 42

os.makedirs("data",  exist_ok=True)
os.makedirs("model", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
frames = []
for path in RAW_FILES:
    print(f"Loading {os.path.basename(path)} ...", flush=True)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    # keep only needed columns
    available = [f for f in FEATURES if f in df.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        print(f"  WARNING: {len(missing)} features missing — filling with 0")
        for m in missing:
            df[m] = 0.0

    label_col = "Label"
    df = df[available + [label_col]].copy()
    df["Label"] = df["Label"].str.strip()
    frames.append(df)
    print(f"  {len(df):,} rows | labels: {df['Label'].value_counts().to_dict()}")

df = pd.concat(frames, ignore_index=True)
print(f"\nCombined: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
before = len(df)
df.dropna(inplace=True)
print(f"Dropped {before - len(df):,} NaN/Inf rows -> {len(df):,} remaining")

# ── Stratified sample ─────────────────────────────────────────────────────────
print("\nLabel distribution before sampling:")
print(df["Label"].value_counts())

sampled = []
for label, grp in df.groupby("Label"):
    n = min(len(grp), ROWS_PER_CLASS)
    sampled.append(grp.sample(n=n, random_state=RANDOM_STATE))
df = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)

print("\nLabel distribution after sampling:")
print(df["Label"].value_counts())

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
df["Label"] = le.fit_transform(df["Label"])
joblib.dump(le, "model/label_encoder.pkl")

print(f"\nClass encoding:")
for i, cls in enumerate(le.classes_):
    print(f"  {i} -> {cls}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = "data/multiclass_dataset.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}  shape={df.shape}")
print("Done.")
