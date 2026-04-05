"""
Comprehensive model evaluation on the saved test dataset.
Run from the repo root:  python src/evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

print("Loading model, scaler, and test data...")
model  = load_model(str(config.MODEL_FILE))
scaler = joblib.load(config.SCALER_FILE)

try:
    label_encoder = joblib.load(config.ENCODER_FILE)
    class_names   = label_encoder.classes_
except FileNotFoundError:
    label_encoder = None
    class_names   = None

# ─── Load Test Data ───────────────────────────────────────────────────────────
df     = pd.read_csv(config.TEST_DATASET)
X_test = df.drop("Label", axis=1).values
y_test = df["Label"].values.astype(int)

# The test CSV already contains scaled values (saved by model_multi_final.py).
# Reshape to (N, timesteps=1, features).
X_input = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ─── Predict ──────────────────────────────────────────────────────────────────
print("Running predictions...")
y_pred_probs = model.predict(X_input, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

# ─── Basic Metrics ────────────────────────────────────────────────────────────
accuracy           = accuracy_score(y_test, y_pred)
precision_macro    = precision_score(y_test, y_pred, average="macro",    zero_division=0)
recall_macro       = recall_score(y_test, y_pred,    average="macro",    zero_division=0)
f1_macro           = f1_score(y_test, y_pred,        average="macro",    zero_division=0)
precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall_weighted    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1_weighted        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

print("\n===== BASIC METRICS =====")
print(f"Accuracy               : {accuracy:.4f}")
print(f"Precision (Macro)      : {precision_macro:.4f}")
print(f"Recall    (Macro)      : {recall_macro:.4f}")
print(f"F1 Score  (Macro)      : {f1_macro:.4f}")
print(f"Precision (Weighted)   : {precision_weighted:.4f}")
print(f"Recall    (Weighted)   : {recall_weighted:.4f}")
print(f"F1 Score  (Weighted)   : {f1_weighted:.4f}")

# ─── Advanced Metrics ─────────────────────────────────────────────────────────
kappa     = cohen_kappa_score(y_test, y_pred)
mcc       = matthews_corrcoef(y_test, y_pred)
logloss   = log_loss(y_test, y_pred_probs)
error_rate = 1 - accuracy

print("\n===== ADVANCED METRICS =====")
print(f"Cohen Kappa Score      : {kappa:.4f}")
print(f"Matthews Corrcoef (MCC): {mcc:.4f}")
print(f"Log Loss               : {logloss:.4f}")
print(f"Error Rate             : {error_rate:.4f}")

# ─── ROC-AUC ──────────────────────────────────────────────────────────────────
print("\n===== ROC-AUC =====")
try:
    roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class="ovr", average="macro")
    print(f"ROC-AUC (OvR, Macro)   : {roc_auc:.4f}")
except ValueError as e:
    print(f"ROC-AUC not computed: {e}")

# ─── Classification Report ────────────────────────────────────────────────────
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test, y_pred,
    target_names=class_names,
    zero_division=0,
))

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ─── Class-wise Accuracy ──────────────────────────────────────────────────────
print("\n===== CLASS-WISE ACCURACY =====")
class_accuracy = cm.diagonal() / cm.sum(axis=1)
labels = np.unique(y_test)
for i, (label, ca) in enumerate(zip(labels, class_accuracy)):
    name = class_names[label] if class_names is not None else f"Class {label}"
    print(f"  {name:25s}: {ca:.4f}  (n={cm[i].sum()})")

# ─── Samples Per Class ────────────────────────────────────────────────────────
print("\n===== SAMPLES PER CLASS =====")
counts = np.bincount(y_test)
for i, cnt in enumerate(counts):
    name = class_names[i] if class_names is not None else f"Class {i}"
    print(f"  {name:25s}: {cnt}")
