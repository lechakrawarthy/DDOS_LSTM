"""
src/train.py

Training loop for the DDoS LSTM model.

Features
--------
* Mixed-precision training (torch.cuda.amp) when a GPU is available.
* TensorBoard logging (loss, accuracy, learning-rate per epoch).
* Early stopping with best-model checkpointing.
* Class-weighted cross-entropy to handle class imbalance.
* StepLR learning-rate scheduler.
* Gradient clipping.

Usage
-----
    python src/train.py                          # use preprocessed data
    python src/train.py --csv data/raw/net.csv   # preprocess on-the-fly
    python src/train.py --multi                  # multi-class mode
"""

from src.utils import (
    set_seed,
    get_device,
    compute_class_weights,
    format_elapsed,
    save_checkpoint,
    load_checkpoint,
)
from src.model import build_model
import config as cfg
import sys
import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def make_loaders(data: dict, batch_size: int = cfg.BATCH_SIZE):
    def _loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, pin_memory=True,
        )

    return (
        _loader(data["X_train"], data["y_train"], shuffle=True),
        _loader(data["X_val"],   data["y_val"],   shuffle=False),
        _loader(data["X_test"],  data["y_test"],  shuffle=False),
    )


# ─── Single epoch ─────────────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, scaler_amp, device, training: bool):
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler_amp.step(optimizer)
                scaler_amp.update()

            bs = y_batch.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += bs

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ─── Main training function ───────────────────────────────────────────────────

def train(data: dict, binary: bool = True):
    set_seed(cfg.RANDOM_STATE)
    device = get_device()
    logger.info(f"Training on device: {device}")

    # Model
    n_features = data["n_features"]
    n_classes = data["n_classes"]
    model = build_model(n_features, n_classes).to(device)

    # Data loaders
    train_loader, val_loader, test_loader = make_loaders(data)

    # Loss: weighted cross-entropy for imbalance
    weights = compute_class_weights(data["y_train"], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimiser & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.LEARNING_RATE,
                      weight_decay=cfg.WEIGHT_DECAY)
    scheduler = StepLR(
        optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)

    # AMP gradient scaler
    amp_scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # TensorBoard
    writer = SummaryWriter(log_dir=str(cfg.LOGS_DIR / "runs"))

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = cfg.MODELS_DIR / "best_model.pt"

    logger.info("=" * 60)
    logger.info(f"  Epochs: {cfg.EPOCHS}  |  Batch: {cfg.BATCH_SIZE}  "
                f"|  LR: {cfg.LEARNING_RATE}  |  Classes: {n_classes}")
    logger.info("=" * 60)

    t0 = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, amp_scaler, device, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, amp_scaler, device, training=False
        )
        scheduler.step()

        # TensorBoard
        writer.add_scalars(
            "Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars(
            "Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch:3d}/{cfg.EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"[{format_elapsed(t0)}]"
        )

        # Checkpoint & early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt)
            logger.info(
                f"  ✔ Best model saved  (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                logger.info(
                    f"  Early stopping triggered after {epoch} epochs.")
                break

    writer.close()
    logger.info(f"Training complete in {format_elapsed(t0)}.")

    # Reload best weights for final evaluation
    load_checkpoint(model, best_ckpt, device)
    logger.info("Best checkpoint loaded for final evaluation.")
    return model, test_loader


# ─── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDoS LSTM model.")
    parser.add_argument("--csv",   type=str, default=None,
                        help="Raw CSV path. If omitted, uses pre-processed data.")
    parser.add_argument("--multi", action="store_true",
                        help="Multi-class mode (default: binary).")
    args = parser.parse_args()

    # Load or generate processed data
    if cfg.PROCESSED_FILE.exists() and args.csv is None:
        logger.info(f"Loading preprocessed data from {cfg.PROCESSED_FILE} …")
        with open(cfg.PROCESSED_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        from src.preprocess import run_pipeline
        csv_path = Path(args.csv) if args.csv else cfg.RAW_CSV
        data = run_pipeline(csv_path, binary=not args.multi)

    model, test_loader = train(data, binary=not args.multi)

    # Run evaluation immediately after training
    from src.evaluate import evaluate_model
    evaluate_model(model, test_loader, data["class_names"])
