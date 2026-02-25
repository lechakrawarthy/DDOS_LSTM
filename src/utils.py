"""
src/utils.py

Shared utility functions used across the pipeline.
"""

import os
import time
import random
import logging
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Device Selection ─────────────────────────────────────────────────────────

def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU.")
    return device


# ─── Class Weights ────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weighting.
    w_c = N / (n_classes * count_c)
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)          # avoid division by zero
    weights = len(y) / (n_classes * counts)
    logger.info(f"Class weights: {dict(enumerate(weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, val_loss: float, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, path: Path, device: torch.device, optimizer=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logger.info(f"Checkpoint loaded: epoch {ckpt.get('epoch', '?')}, "
                f"val_loss={ckpt.get('val_loss', '?'):.4f}")
    return ckpt


# ─── Time formatting ──────────────────────────────────────────────────────────

def format_elapsed(start: float) -> str:
    elapsed = int(time.time() - start)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


# ─── Training-curve smoothing ─────────────────────────────────────────────────

def smooth(values: list[float], weight: float = 0.85) -> list[float]:
    """Exponential moving average (like TensorBoard smoothing)."""
    smoothed, last = [], 0.0
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


# ─── Prediction helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device: torch.device):
    """
    Run inference over a DataLoader.

    Returns
    -------
    all_probs  : (N, n_classes) softmax probabilities
    all_preds  : (N,) integer class predictions
    all_labels : (N,) true labels
    """
    model.eval()
    probs_list, preds_list, labels_list = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        p = torch.softmax(logits, dim=1).cpu()
        probs_list.append(p)
        preds_list.append(p.argmax(dim=1))
        labels_list.append(y_batch)

    all_probs = torch.cat(probs_list,  dim=0).numpy()
    all_preds = torch.cat(preds_list,  dim=0).numpy()
    all_labels = torch.cat(labels_list, dim=0).numpy()
    return all_probs, all_preds, all_labels
