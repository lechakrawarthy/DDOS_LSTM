import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model

# ---------------- CREATE OUTPUT FOLDER ----------------
os.makedirs("results", exist_ok=True)

# ---------------- LOAD ----------------
print("Loading model...")

model = load_model("model/multiclass_model.keras")
scaler = joblib.load("model/multiclass_scaler.pkl")

df = pd.read_csv("data/test_dataset.csv")

X_test = df.drop("Label", axis=1)
y_test = df["Label"].values

# ---------------- PREPROCESS ----------------
X_test = scaler.transform(X_test)
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ---------------- PREDICT ----------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ---------------- REPORT ----------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 🔥 GRAPH 1: CONFUSION MATRIX HEATMAP
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig("results/confusion_matrix.png")
plt.show()

# ---------------- ROC CURVE ----------------
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure()

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multiclass)")
plt.legend()

plt.savefig("results/roc_curve.png")
plt.show()

# ---------------- PRECISION-RECALL CURVE ----------------
plt.figure()

for i in range(len(classes)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(recall, precision, label=f"Class {i}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.savefig("results/precision_recall_curve.png")
plt.show()

print("\n✅ Graphs saved in 'results/' folder")