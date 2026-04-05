"""
Real-time DDoS detection using pyshark live packet capture.

How it works
------------
1. Capture packets from a network interface with pyshark.
2. Extract per-packet features that map to the 75-feature CIC-style vector
   the trained LSTM model expects.
3. Accumulate packets into a batch (config.CAPTURE_BATCH_SIZE).
4. Scale the batch with the saved StandardScaler and run LSTM inference.
5. Print the prediction (BENIGN / attack type) and confidence per packet.

Requirements
------------
- Wireshark / tshark must be installed and on PATH.
- On Windows run as Administrator; on Linux run as root or with CAP_NET_RAW.
- pip install pyshark

Run from the repo root:
    python src/live_capture.py [--interface <iface>] [--batch <n>] [--list-ifaces]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import joblib

import config

# ── Exact 75 features the model was trained on (order matters) ────────────────
LIVE_FEATURES = [
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

_N_FEATURES = len(LIVE_FEATURES)
_FEAT_IDX   = {name: i for i, name in enumerate(LIVE_FEATURES)}


def _f(val, default=0.0):
    """Safe float conversion."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def extract_features(pkt) -> np.ndarray:
    """
    Map a single pyshark packet to the 75-dim feature vector.

    Many CIC flow-level features (IAT statistics, bulk rates, etc.) require a
    complete bidirectional flow and cannot be computed from one packet alone.
    Those positions are set to 0.  Fields that ARE accessible per-packet
    (length, TCP flags, header length, window size) are filled in correctly.
    """
    vec = np.zeros(_N_FEATURES, dtype=np.float32)
    idx = _FEAT_IDX

    pkt_len = _f(getattr(pkt, "length", None))

    # ── Packet length features ─────────────────────────────────────────────
    for key in ("Min Packet Length", "Max Packet Length",
                "Packet Length Mean", "Average Packet Size"):
        if key in idx:
            vec[idx[key]] = pkt_len

    # ── TCP-specific ───────────────────────────────────────────────────────
    if hasattr(pkt, "tcp"):
        tcp = pkt.tcp

        # Flags
        try:
            iflags = int(getattr(tcp, "flags", "0x0"), 16)
        except (ValueError, TypeError):
            iflags = 0

        flag_map = {
            "FIN Flag Count": 0x001,
            "SYN Flag Count": 0x002,
            "RST Flag Count": 0x004,
            "PSH Flag Count": 0x008,
            "ACK Flag Count": 0x010,
            "URG Flag Count": 0x020,
            "ECE Flag Count": 0x040,
            "CWE Flag Count": 0x080,
            "Fwd PSH Flags":  0x008,
            "Fwd URG Flags":  0x020,
        }
        for fname, mask in flag_map.items():
            if fname in idx:
                vec[idx[fname]] = float(bool(iflags & mask))

        hdr_len = _f(getattr(tcp, "hdr_len", None))
        for key in ("Fwd Header Length", "Bwd Header Length", "Fwd Header Length.1"):
            if key in idx:
                vec[idx[key]] = hdr_len

        win = _f(getattr(tcp, "window_size_value", None))
        if "Init_Win_bytes_forward" in idx:
            vec[idx["Init_Win_bytes_forward"]] = win

        seg = _f(getattr(tcp, "len", None))
        if "min_seg_size_forward" in idx:
            vec[idx["min_seg_size_forward"]] = seg

        payload = _f(getattr(tcp, "len", None))
        for key in ("Total Length of Fwd Packets", "Fwd Packet Length Max",
                    "Fwd Packet Length Min", "Fwd Packet Length Mean",
                    "Avg Fwd Segment Size"):
            if key in idx:
                vec[idx[key]] = payload

        if "act_data_pkt_fwd" in idx:
            vec[idx["act_data_pkt_fwd"]] = 1.0 if payload > 0 else 0.0

    # ── UDP-specific ───────────────────────────────────────────────────────
    elif hasattr(pkt, "udp"):
        udp_len = _f(getattr(pkt.udp, "length", None))
        if "Total Length of Fwd Packets" in idx:
            vec[idx["Total Length of Fwd Packets"]] = udp_len

    # ── IP-level ───────────────────────────────────────────────────────────
    if hasattr(pkt, "ip"):
        ttl = _f(getattr(pkt.ip, "ttl", None))
        if "Flow Duration" in idx:
            vec[idx["Flow Duration"]] = ttl

    return vec


# ── Inference ─────────────────────────────────────────────────────────────────
def load_artifacts():
    for path, label in [
        (config.MODEL_FILE,  "Model"),
        (config.SCALER_FILE, "Scaler"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                "Train the model first:  python src/model_multi_final.py"
            )

    from tensorflow.keras.models import load_model as tf_load
    import warnings
    warnings.filterwarnings("ignore")

    model  = tf_load(str(config.MODEL_FILE))
    scaler = joblib.load(config.SCALER_FILE)

    # Verify feature count matches
    if scaler.n_features_in_ != _N_FEATURES:
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features "
            f"but live_capture builds {_N_FEATURES}. "
            "Retrain the model or update LIVE_FEATURES."
        )

    try:
        label_encoder = joblib.load(config.ENCODER_FILE)
        class_names   = label_encoder.classes_.tolist()
    except FileNotFoundError:
        class_names = None

    return model, scaler, class_names


def predict_batch(model, scaler, batch: list, class_names) -> None:
    X = np.stack(batch, axis=0)
    X = scaler.transform(X)
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    probs  = model.predict(X, verbose=0)
    labels = np.argmax(probs, axis=1)

    for i, (label, prob_row) in enumerate(zip(labels, probs)):
        name       = class_names[label] if class_names else str(label)
        confidence = float(prob_row[label])
        status     = "BENIGN" if name == "BENIGN" else "ATTACK"
        print(f"  [{i+1:02d}] {status:6s} | {name:15s} | conf={confidence:.3f}")


# ── List interfaces ────────────────────────────────────────────────────────────
def list_interfaces():
    try:
        import pyshark
        cap = pyshark.LiveCapture()
        print("Available interfaces:")
        for iface in cap.interfaces:
            print(f"  {iface}")
    except Exception as e:
        print(f"Could not list interfaces: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live DDoS detection with pyshark")
    parser.add_argument("--interface", "-i", default=None,
                        help="Network interface (default: auto)")
    parser.add_argument("--batch", "-b", type=int,
                        default=config.CAPTURE_BATCH_SIZE,
                        help=f"Packets per inference window (default: {config.CAPTURE_BATCH_SIZE})")
    parser.add_argument("--list-ifaces", action="store_true",
                        help="List available interfaces and exit")
    args = parser.parse_args()

    if args.list_ifaces:
        list_interfaces()
        return

    try:
        import pyshark
    except ImportError:
        print(
            "ERROR: pyshark not installed.\n"
            "       pip install pyshark\n"
            "       Also install Wireshark and ensure tshark is on PATH."
        )
        sys.exit(1)

    print("Loading model artifacts...")
    model, scaler, class_names = load_artifacts()

    interface = args.interface or config.CAPTURE_INTERFACE
    print(f"Capturing on: {interface or 'auto-detected'}")
    print(f"Batch size  : {args.batch} packets per inference window")
    print("Press Ctrl+C to stop.\n")

    capture = pyshark.LiveCapture(interface=interface, bpf_filter="ip")

    batch = []
    window_count = 0

    try:
        for pkt in capture.sniff_continuously():
            try:
                feat = extract_features(pkt)
                batch.append(feat)
            except Exception:
                continue

            if len(batch) >= args.batch:
                window_count += 1
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] Window #{window_count} ({len(batch)} packets):")
                predict_batch(model, scaler, batch, class_names)
                batch = []

    except KeyboardInterrupt:
        print("\nCapture stopped.")
        if batch:
            print(f"Processing remaining {len(batch)} packet(s):")
            predict_batch(model, scaler, batch, class_names)


if __name__ == "__main__":
    main()
