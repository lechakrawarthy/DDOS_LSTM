"""
Generate and save evaluation visualisations (confusion matrix, ROC, PR curves).
Run from the repo root:  python src/evaluation_vis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

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

print("Loading model and test data...")
model  = load_model(str(config.MODEL_FILE))
scaler = joblib.load(config.SCALER_FILE)

try:
    label_encoder = joblib.load(config.ENCODER_FILE)
    class_names   = label_encoder.classes_.tolist()
except FileNotFoundError:
    label_encoder = None
    class_names   = None

df     = pd.read_csv(config.TEST_DATASET)
X_test = df.drop("Label", axis=1).values
y_test = df["Label"].values.astype(int)

X_input      = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
y_pred_probs = model.predict(X_input, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

classes = np.unique(y_test)
if class_names is None:
    class_names = [str(c) for c in classes]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# ─── Graph 1: Confusion Matrix Heatmap ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()

out_path = config.RESULTS_DIR / "confusion_matrix.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Graph 2: ROC Curve (one-vs-rest) ─────────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=classes)

fig, ax = plt.subplots(figsize=(8, 6))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Multiclass OvR)")
ax.legend(fontsize=8)
plt.tight_layout()

out_path = config.RESULTS_DIR / "roc_curve.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

# ─── Graph 3: Precision-Recall Curve ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for i, cls in enumerate(classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, label=f"{class_names[i]} (AUC={pr_auc:.2f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve")
ax.legend(fontsize=8)
plt.tight_layout()

out_path = config.RESULTS_DIR / "precision_recall_curve.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

print(f"\nAll graphs saved in '{config.RESULTS_DIR}/'")
