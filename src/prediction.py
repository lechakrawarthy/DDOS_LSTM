"""
Run inference on a single random sample from the saved test dataset.
Run from the repo root:  python src/prediction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

import config

# ─── Load Artifacts ───────────────────────────────────────────────────────────
for path, label in [
    (config.MODEL_FILE,   "Model"),
    (config.SCALER_FILE,  "Scaler"),
    (config.TEST_DATASET, "Test dataset"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

print("Loading model and scaler...")
model  = load_model(str(config.MODEL_FILE))
scaler = joblib.load(config.SCALER_FILE)

# Optional label encoder
try:
    label_encoder = joblib.load(config.ENCODER_FILE)
    class_names = label_encoder.classes_
except FileNotFoundError:
    label_encoder = None
    class_names = None

# ─── Load Test Data ───────────────────────────────────────────────────────────
df = pd.read_csv(config.TEST_DATASET)

# ─── Random Sample ────────────────────────────────────────────────────────────
sample  = df.sample(n=1, random_state=None)
X_sample = sample.drop("Label", axis=1).values
y_true   = sample["Label"].values[0]

# ─── Scale & Reshape ─────────────────────────────────────────────────────────
# The test dataset is already scaled by model_multi_final.py.
# Reshape to (1, timesteps=1, features).
X_input = X_sample.reshape((1, 1, X_sample.shape[1]))

# ─── Predict ──────────────────────────────────────────────────────────────────
y_pred_prob = model.predict(X_input, verbose=0)
y_pred      = int(np.argmax(y_pred_prob, axis=1)[0])
confidence  = float(np.max(y_pred_prob))

# ─── Output ───────────────────────────────────────────────────────────────────
def _class_name(idx):
    if class_names is not None:
        return class_names[idx]
    return str(idx)

print("\n===== RANDOM SAMPLE PREDICTION =====")
print(f"Actual Class    : {_class_name(int(y_true))}")
print(f"Predicted Class : {_class_name(y_pred)}")
print(f"Confidence      : {confidence:.4f}")
print(f"Correct         : {int(y_true) == y_pred}")

print("\nClass Probabilities:")
for i, prob in enumerate(y_pred_prob[0]):
    marker = " <--" if i == y_pred else ""
    print(f"  {_class_name(i):20s}: {prob:.4f}{marker}")
