"""
scripts/generate_synthetic_data.py

Generates a synthetic network-traffic CSV that mimics CIC-DDoS2019 column
names and statistics.  Useful for testing the pipeline end-to-end without
downloading the real dataset.

Output  →  data/raw/network_traffic.csv
"""

import config as cfg
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


RANDOM_SEED = 42
N_BENIGN = 50_000
N_DDOS = 15_000    # intentionally imbalanced (30% minority)


def _benign(n: int, rng: np.random.Generator) -> dict:
    """Typical benign traffic: lower packet rates, larger flow durations."""
    return {
        "Flow Duration":            rng.exponential(scale=500_000, size=n),
        "Total Fwd Packets":        rng.integers(1,   100, size=n).astype(float),
        "Total Backward Packets":   rng.integers(1,   80,  size=n).astype(float),
        "Total Length of Fwd Packets": rng.exponential(scale=5_000, size=n),
        "Total Length of Bwd Packets": rng.exponential(scale=3_000, size=n),
        "Fwd Packet Length Max":    rng.uniform(64,  1500, size=n),
        "Fwd Packet Length Min":    rng.uniform(40,  200,  size=n),
        "Fwd Packet Length Mean":   rng.uniform(100, 800,  size=n),
        "Fwd Packet Length Std":    rng.uniform(0,   400,  size=n),
        "Bwd Packet Length Max":    rng.uniform(64,  1500, size=n),
        "Bwd Packet Length Min":    rng.uniform(40,  200,  size=n),
        "Bwd Packet Length Mean":   rng.uniform(100, 800,  size=n),
        "Bwd Packet Length Std":    rng.uniform(0,   400,  size=n),
        "Flow Bytes/s":             rng.exponential(scale=50_000, size=n),
        "Flow Packets/s":           rng.exponential(scale=200,    size=n),
        "Flow IAT Mean":            rng.exponential(scale=50_000, size=n),
        "Flow IAT Std":             rng.exponential(scale=30_000, size=n),
        "Flow IAT Max":             rng.exponential(scale=200_000, size=n),
        "Flow IAT Min":             rng.uniform(0,   1_000, size=n),
        "Fwd IAT Total":            rng.exponential(scale=200_000, size=n),
        "Fwd IAT Mean":             rng.exponential(scale=50_000, size=n),
        "Fwd IAT Std":              rng.exponential(scale=30_000, size=n),
        "Fwd IAT Max":              rng.exponential(scale=200_000, size=n),
        "Fwd IAT Min":              rng.uniform(0,   1_000, size=n),
        "Bwd IAT Total":            rng.exponential(scale=200_000, size=n),
        "Bwd IAT Mean":             rng.exponential(scale=50_000, size=n),
        "Bwd IAT Std":              rng.exponential(scale=30_000, size=n),
        "Bwd IAT Max":              rng.exponential(scale=200_000, size=n),
        "Bwd IAT Min":              rng.uniform(0,   1_000, size=n),
        "Fwd PSH Flags":            rng.integers(0, 2, size=n).astype(float),
        "Fwd URG Flags":            rng.integers(0, 2, size=n).astype(float),
        "Fwd Header Length":        rng.integers(20, 60, size=n).astype(float),
        "Bwd Header Length":        rng.integers(20, 60, size=n).astype(float),
        "Fwd Packets/s":            rng.exponential(scale=100, size=n),
        "Bwd Packets/s":            rng.exponential(scale=80,  size=n),
        "Min Packet Length":        rng.uniform(40,  200,  size=n),
        "Max Packet Length":        rng.uniform(64, 1500,  size=n),
        "Packet Length Mean":       rng.uniform(100, 800,  size=n),
        "Packet Length Std":        rng.uniform(0,   400,  size=n),
        "Packet Length Variance":   rng.uniform(0, 160_000, size=n),
        "FIN Flag Count":           rng.integers(0, 2, size=n).astype(float),
        "SYN Flag Count":           rng.integers(0, 2, size=n).astype(float),
        "RST Flag Count":           rng.integers(0, 2, size=n).astype(float),
        "PSH Flag Count":           rng.integers(0, 4, size=n).astype(float),
        "ACK Flag Count":           rng.integers(0, 5, size=n).astype(float),
        "URG Flag Count":           rng.integers(0, 2, size=n).astype(float),
        "CWE Flag Count":           rng.integers(0, 2, size=n).astype(float),
        "ECE Flag Count":           rng.integers(0, 2, size=n).astype(float),
        "Down/Up Ratio":            rng.uniform(0, 5, size=n),
        "Average Packet Size":      rng.uniform(100, 800, size=n),
        "Avg Fwd Segment Size":     rng.uniform(100, 800, size=n),
        "Avg Bwd Segment Size":     rng.uniform(100, 800, size=n),
        "Fwd Header Length.1":      rng.integers(20, 60, size=n).astype(float),
        "Fwd Avg Bytes/Bulk":       rng.exponential(scale=1_000, size=n),
        "Fwd Avg Packets/Bulk":     rng.exponential(scale=5,     size=n),
        "Fwd Avg Bulk Rate":        rng.exponential(scale=10_000, size=n),
        "Bwd Avg Bytes/Bulk":       rng.exponential(scale=1_000, size=n),
        "Bwd Avg Packets/Bulk":     rng.exponential(scale=5,     size=n),
        "Bwd Avg Bulk Rate":        rng.exponential(scale=10_000, size=n),
        "Subflow Fwd Packets":      rng.integers(1, 50, size=n).astype(float),
        "Subflow Fwd Bytes":        rng.exponential(scale=2000, size=n),
        "Subflow Bwd Packets":      rng.integers(1, 40, size=n).astype(float),
        "Subflow Bwd Bytes":        rng.exponential(scale=1500, size=n),
        "Init_Win_bytes_forward":   rng.integers(0, 65535, size=n).astype(float),
        "Init_Win_bytes_backward":  rng.integers(0, 65535, size=n).astype(float),
        "act_data_pkt_fwd":         rng.integers(0, 50,  size=n).astype(float),
        "min_seg_size_forward":     rng.integers(20, 40, size=n).astype(float),
        "Active Mean":              rng.exponential(scale=100_000, size=n),
        "Active Std":               rng.exponential(scale=50_000,  size=n),
        "Active Max":               rng.exponential(scale=300_000, size=n),
        "Active Min":               rng.uniform(0, 10_000, size=n),
        "Idle Mean":                rng.exponential(scale=1_000_000, size=n),
        "Idle Std":                 rng.exponential(scale=500_000,   size=n),
        "Idle Max":                 rng.exponential(scale=3_000_000, size=n),
        "Idle Min":                 rng.uniform(0, 50_000, size=n),
        "Label":                    ["BENIGN"] * n,
    }


def _ddos(n: int, rng: np.random.Generator) -> dict:
    """DDoS traffic: very high packet rates, tiny flow durations, many SYN flags."""
    d = _benign(n, rng)                 # start with benign template and override
    d["Flow Duration"] = rng.exponential(scale=5_000,  size=n)   # much shorter
    d["Total Fwd Packets"] = rng.integers(100, 10_000, size=n).astype(float)
    d["Flow Bytes/s"] = rng.exponential(scale=5_000_000, size=n)
    d["Flow Packets/s"] = rng.exponential(scale=50_000,    size=n)
    d["Flow IAT Mean"] = rng.uniform(0,  500, size=n)             # tiny gaps
    d["SYN Flag Count"] = rng.integers(5, 500, size=n).astype(float)
    d["RST Flag Count"] = rng.integers(0, 100, size=n).astype(float)
    d["Fwd Packets/s"] = rng.exponential(scale=40_000, size=n)
    d["Bwd Packets/s"] = rng.exponential(scale=40_000, size=n)
    d["Min Packet Length"] = rng.uniform(
        40, 80, size=n)              # tiny packets
    d["Max Packet Length"] = rng.uniform(40, 200, size=n)
    d["Packet Length Mean"] = rng.uniform(40, 80,  size=n)
    d["Packet Length Std"] = rng.uniform(0,  20,  size=n)
    d["Init_Win_bytes_forward"] = rng.integers(0, 1024, size=n).astype(float)
    d["Label"] = ["DDoS"] * n
    return d


def generate(out_path: Path = cfg.RAW_CSV):
    rng = np.random.default_rng(RANDOM_SEED)

    benign = pd.DataFrame(_benign(N_BENIGN, rng))
    ddos = pd.DataFrame(_ddos(N_DDOS, rng))

    df = pd.concat([benign, ddos], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add leading/trailing spaces to column names to mimic CIC-DDoS2019 format
    rename = {c: f" {c}" for c in df.columns if c != "Label"}
    rename["Label"] = " Label"
    df.rename(columns=rename, inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved → {out_path}")
    print(f"  Total rows : {len(df):,}")
    print(f"  BENIGN     : {(df[' Label'] == 'BENIGN').sum():,}")
    print(f"  DDoS       : {(df[' Label'] == 'DDoS').sum():,}")
    print(f"  Columns    : {len(df.columns)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate synthetic traffic CSV.")
    parser.add_argument("--out", type=str, default=str(cfg.RAW_CSV))
    args = parser.parse_args()
    generate(Path(args.out))
