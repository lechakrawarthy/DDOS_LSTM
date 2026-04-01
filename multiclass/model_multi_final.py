import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
df = pd.read_csv("data/multiclass_dataset.csv")

print("Dataset shape:", df.shape)

# ---------------- SPLIT FEATURES & LABEL ----------------
X = df.drop("Label", axis=1)
y = df["Label"]

# ---------------- TRAIN / VAL / TEST SPLIT ----------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ---------------- SAVE TEST DATASET ----------------
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df["Label"] = y_test.values

os.makedirs("data", exist_ok=True)
test_df.to_csv("data/test_dataset.csv", index=False)

print("Test dataset saved at data/test_dataset.csv")

# ---------------- SCALING ----------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/multiclass_scaler.pkl")

# ---------------- RESHAPE FOR LSTM ----------------
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ---------------- NUMBER OF CLASSES ----------------
num_classes = len(np.unique(y))
print("Number of classes:", num_classes)

# ---------------- BUILD MODEL ----------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- TRAIN ----------------
print("Training model...")

history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=512,
    validation_data=(X_val, y_val)
)

# ---------------- EVALUATE ----------------
print("Evaluating model...")

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# ---------------- PREDICTIONS ----------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# ---------------- CLASS-WISE ACCURACY ----------------
print("\nClass-wise Accuracy:")

class_accuracy = cm.diagonal() / cm.sum(axis=1)

labels = np.unique(y_test)

for label, acc in zip(labels, class_accuracy):
    print(f"Class {label} Accuracy: {acc:.4f}")

# ---------------- SAVE MODEL ----------------
model.save("model/multiclass_model.keras")

print("\nModel saved at model/multiclass_model.keras")
print("Scaler saved at model/multiclass_scaler.pkl")
