import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss
)

from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
print("Loading model, scaler, and encoder...")

model = load_model("model/multiclass_model.keras")
scaler = joblib.load("model/multiclass_scaler.pkl")

# Optional (if using label encoder)
try:
    label_encoder = joblib.load("model/label_encoder.pkl")
    use_encoder = True
except:
    use_encoder = False

# ---------------- LOAD TEST DATA ----------------
df = pd.read_csv("data/test_dataset.csv")

X_test = df.drop("Label", axis=1)
y_test = df["Label"].values

# ---------------- SCALE ----------------
X_test = scaler.transform(X_test)

# ---------------- RESHAPE FOR LSTM ----------------
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ---------------- PREDICT ----------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ---------------- BASIC METRICS ----------------
print("\n===== BASIC METRICS =====")

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy               : {accuracy:.4f}")
print(f"Precision (Macro)      : {precision_macro:.4f}")
print(f"Recall (Macro)         : {recall_macro:.4f}")
print(f"F1 Score (Macro)       : {f1_macro:.4f}")
print(f"Precision (Weighted)   : {precision_weighted:.4f}")
print(f"Recall (Weighted)      : {recall_weighted:.4f}")
print(f"F1 Score (Weighted)    : {f1_weighted:.4f}")

# ---------------- ADVANCED METRICS ----------------
print("\n===== ADVANCED METRICS =====")

kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
loss = log_loss(y_test, y_pred_probs)

print(f"Cohen Kappa Score      : {kappa:.4f}")
print(f"Matthews Corrcoef (MCC): {mcc:.4f}")
print(f"Log Loss               : {loss:.4f}")

# ---------------- ROC-AUC ----------------
print("\n===== ROC-AUC =====")

try:
    roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
    print(f"ROC-AUC (OvR)          : {roc_auc:.4f}")
except:
    print("ROC-AUC not computed (check class format)")

# ---------------- CLASSIFICATION REPORT ----------------
print("\n===== CLASSIFICATION REPORT =====")

if use_encoder:
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))
else:
    print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(y_test, y_pred)
print(cm)

# ---------------- CLASS-WISE ACCURACY ----------------
print("\n===== CLASS-WISE ACCURACY =====")

class_accuracy = cm.diagonal() / cm.sum(axis=1)

if use_encoder:
    for label, acc in zip(label_encoder.classes_, class_accuracy):
        print(f"{label}: {acc:.4f}")
else:
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i}: {acc:.4f}")

# ---------------- EXTRA METRICS ----------------
print("\n===== EXTRA INSIGHTS =====")

# Error rate
error_rate = 1 - accuracy
print(f"Error Rate             : {error_rate:.4f}")

# Per-class support
print("\nSamples per class:")
print(np.bincount(y_test))