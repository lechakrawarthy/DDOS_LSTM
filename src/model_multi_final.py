"""
Train a multiclass LSTM model for DDoS attack detection.
Run from the repo root:  python src/model_multi_final.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading dataset...")
if not config.MULTICLASS_DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {config.MULTICLASS_DATASET}\n"
        "Run  python src/preprocess.py  first to generate it."
    )

df = pd.read_csv(config.MULTICLASS_DATASET)
print(f"Dataset shape: {df.shape}")

# ─── Split Features & Label ───────────────────────────────────────────────────
X = df.drop("Label", axis=1).values
y = df["Label"].values

# ─── Train / Val / Test Split ─────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=config.TEST_SIZE + config.VAL_SIZE,
    stratify=y,
    random_state=config.RANDOM_STATE,
)
val_fraction = config.VAL_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=1.0 - val_fraction,
    stratify=y_temp,
    random_state=config.RANDOM_STATE,
)

print(f"Train : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

# ─── Scaling ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, config.SCALER_FILE)
print(f"Scaler saved to {config.SCALER_FILE}")

# ─── Save Test Dataset (scaled values + labels) ───────────────────────────────
feature_cols = df.drop("Label", axis=1).columns.tolist()
test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df["Label"] = y_test
test_df.to_csv(config.TEST_DATASET, index=False)
print(f"Test dataset saved to {config.TEST_DATASET}")

# ─── Reshape for LSTM  (samples, timesteps=1, features) ──────────────────────
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val   = X_val.reshape((X_val.shape[0],   1, X_val.shape[1]))
X_test  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

num_classes = len(np.unique(y))
n_features  = X_train.shape[2]
print(f"Classes : {num_classes}  |  Features : {n_features}")

# ─── Build Model ──────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(config.HIDDEN_SIZE, return_sequences=True,
         input_shape=(1, n_features), dropout=config.DROPOUT),
    LSTM(config.HIDDEN_SIZE // 2, dropout=config.DROPOUT),
    Dense(64, activation="relu"),
    Dropout(config.DROPOUT),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ─── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=str(config.MODEL_FILE),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ─── Train ────────────────────────────────────────────────────────────────────
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {acc:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClass-wise Accuracy:")
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for label, ca in zip(np.unique(y_test), class_accuracy):
    print(f"  Class {label}: {ca:.4f}")

print(f"\nModel saved to {config.MODEL_FILE}")
