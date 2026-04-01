import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL & SCALER ----------------
print("Loading model and scaler...")

model = load_model("model/multiclass_model.keras")
scaler = joblib.load("model/multiclass_scaler.pkl")

# ---------------- LOAD TEST DATA ----------------
df = pd.read_csv("data/test_dataset.csv")

# ---------------- PICK RANDOM SAMPLE ----------------
sample = df.sample(n=1)

X_sample = sample.drop("Label", axis=1)
y_true = sample["Label"].values[0]

# ---------------- SCALE ----------------
X_sample_scaled = scaler.transform(X_sample)

# ---------------- RESHAPE FOR LSTM ----------------
X_sample_scaled = X_sample_scaled.reshape((1, 1, X_sample_scaled.shape[1]))

# ---------------- PREDICT ----------------
y_pred_prob = model.predict(X_sample_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)[0]

# ---------------- OUTPUT ----------------
print("\n===== RANDOM SAMPLE PREDICTION =====")
print("Actual Class   :", y_true)
print("Predicted Class:", y_pred)
print("Prediction Confidence:", np.max(y_pred_prob))

# Optional: show probabilities for all classes
print("\nClass Probabilities:")
for i, prob in enumerate(y_pred_prob[0]):
    print(f"Class {i}: {prob:.4f}")