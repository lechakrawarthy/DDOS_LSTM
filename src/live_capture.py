"""
Real-time DDoS detection using pyshark live packet capture.

How it works
------------
1. Capture packets from a network interface with pyshark.
2. Extract per-packet features that approximate CIC-style flow statistics.
3. Accumulate packets into a small window (config.CAPTURE_BATCH_SIZE).
4. Scale the feature vector and run one inference pass through the trained LSTM.
5. Print the prediction for each window.

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

# ─── Feature Extraction ───────────────────────────────────────────────────────
# We extract a simplified feature vector from individual packets.
# In production these would be computed over complete network flows
# (like CICFlowMeter does), but live single-packet features are a reasonable
# approximation for demonstration purposes.

_ZERO_VEC = np.zeros(len(config.SELECTED_FEATURES), dtype=np.float32)


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def extract_features(pkt) -> np.ndarray:
    """
    Map a pyshark packet to a feature vector aligned with config.SELECTED_FEATURES.

    Many CIC features cannot be computed from a single packet (they require a
    complete flow), so those positions are filled with 0.  The resulting vector
    still gives the model usable signal from length, timing, and flag fields.
    """
    vec = _ZERO_VEC.copy()

    feature_names = [f.strip() for f in config.SELECTED_FEATURES]
    idx = {name: i for i, name in enumerate(feature_names)}

    # ── Packet length ──────────────────────────────────────────────────────
    pkt_len = _safe_float(getattr(pkt, "length", None))

    if "Max Packet Length" in idx:
        vec[idx["Max Packet Length"]] = pkt_len
    if "Min Packet Length" in idx:
        vec[idx["Min Packet Length"]] = pkt_len
    if "Packet Length Mean" in idx:
        vec[idx["Packet Length Mean"]] = pkt_len
    if "Average Packet Size" in idx:
        vec[idx["Average Packet Size"]] = pkt_len

    # ── TCP-specific ───────────────────────────────────────────────────────
    if hasattr(pkt, "tcp"):
        tcp = pkt.tcp
        flags = _safe_float(getattr(tcp, "flags", None))

        # Extract individual TCP flags from the flags integer
        iflags = int(flags) if flags else 0
        if "FIN Flag Count" in idx:
            vec[idx["FIN Flag Count"]] = float(bool(iflags & 0x01))
        if "SYN Flag Count" in idx:
            vec[idx["SYN Flag Count"]] = float(bool(iflags & 0x02))
        if "RST Flag Count" in idx:
            vec[idx["RST Flag Count"]] = float(bool(iflags & 0x04))
        if "PSH Flag Count" in idx:
            vec[idx["PSH Flag Count"]] = float(bool(iflags & 0x08))
        if "ACK Flag Count" in idx:
            vec[idx["ACK Flag Count"]] = float(bool(iflags & 0x10))
        if "URG Flag Count" in idx:
            vec[idx["URG Flag Count"]] = float(bool(iflags & 0x20))

        hdr_len = _safe_float(getattr(tcp, "hdr_len", None))
        if "Fwd Header Length" in idx:
            vec[idx["Fwd Header Length"]] = hdr_len
        if "Bwd Header Length" in idx:
            vec[idx["Bwd Header Length"]] = hdr_len

        win = _safe_float(getattr(tcp, "window_size_value", None))
        if "Init_Win_bytes_forward" in idx:
            vec[idx["Init_Win_bytes_forward"]] = win

        payload_len = _safe_float(getattr(tcp, "len", None))
        if "Total Length of Fwd Packets" in idx:
            vec[idx["Total Length of Fwd Packets"]] = payload_len
        if "Fwd Packet Length Max" in idx:
            vec[idx["Fwd Packet Length Max"]] = payload_len
        if "Fwd Packet Length Mean" in idx:
            vec[idx["Fwd Packet Length Mean"]] = payload_len

    # ── UDP-specific ───────────────────────────────────────────────────────
    elif hasattr(pkt, "udp"):
        udp_len = _safe_float(getattr(pkt.udp, "length", None))
        if "Total Length of Fwd Packets" in idx:
            vec[idx["Total Length of Fwd Packets"]] = udp_len

    # ── IP-level ───────────────────────────────────────────────────────────
    if hasattr(pkt, "ip"):
        ttl = _safe_float(getattr(pkt.ip, "ttl", None))
        if "Flow Duration" in idx:
            # Use TTL as a rough flow-duration proxy
            vec[idx["Flow Duration"]] = ttl

    return vec


# ─── Inference ────────────────────────────────────────────────────────────────
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

    from tensorflow.keras.models import load_model as tf_load_model
    model  = tf_load_model(str(config.MODEL_FILE))
    scaler = joblib.load(config.SCALER_FILE)

    try:
        label_encoder = joblib.load(config.ENCODER_FILE)
        class_names   = label_encoder.classes_.tolist()
    except FileNotFoundError:
        label_encoder = None
        class_names   = None

    return model, scaler, class_names


def predict_batch(model, scaler, batch: list, class_names) -> None:
    """Scale a batch of feature vectors and run one inference pass."""
    X = np.stack(batch, axis=0)             # (batch, n_features)
    X = scaler.transform(X)                 # scale
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # (batch, 1, features)

    probs  = model.predict(X, verbose=0)    # (batch, n_classes)
    labels = np.argmax(probs, axis=1)

    for i, (label, prob_row) in enumerate(zip(labels, probs)):
        name       = class_names[label] if class_names else str(label)
        confidence = float(prob_row[label])
        status     = "ATTACK" if label != 0 else "BENIGN"
        print(
            f"  [{i+1:02d}] {status:6s} | class={name:20s} | conf={confidence:.3f}"
        )


# ─── List Interfaces ──────────────────────────────────────────────────────────
def list_interfaces():
    try:
        import pyshark
        ifaces = pyshark.LiveCapture().interfaces
        print("Available interfaces:")
        for iface in ifaces:
            print(f"  {iface}")
    except Exception as e:
        print(f"Could not list interfaces: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live DDoS detection with pyshark")
    parser.add_argument("--interface", "-i", default=None,
                        help="Network interface to capture on (default: auto)")
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
            "ERROR: pyshark is not installed.\n"
            "       Install it with:  pip install pyshark\n"
            "       Also ensure Wireshark/tshark is installed and on PATH."
        )
        sys.exit(1)

    print("Loading model artifacts...")
    model, scaler, class_names = load_artifacts()

    interface = args.interface or config.CAPTURE_INTERFACE
    print(f"Starting live capture on interface: {interface or 'auto-detected'}")
    print(f"Inference window: {args.batch} packet(s)\n")
    print("Press Ctrl+C to stop.\n")

    capture = pyshark.LiveCapture(
        interface=interface,
        bpf_filter="ip",          # Only capture IPv4 traffic
    )

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
        print("\nCapture stopped by user.")
        if batch:
            print(f"Processing remaining {len(batch)} packet(s):")
            predict_batch(model, scaler, batch, class_names)


if __name__ == "__main__":
    main()
