"""
src/evaluate.py

Comprehensive evaluation of the trained DDoS LSTM model.

Metrics produced
----------------
* Accuracy, Precision, Recall, F1 (macro & weighted)
* Cohen's Kappa
* ROC-AUC (binary: standard, multi-class: one-vs-rest)
* Confusion matrix (saved as PNG)
* Per-class classification report
* ROC curve(s) (saved as PNG)
* Precision-Recall curve (saved as PNG)
* Attention-weight visualisation for a sample batch

Usage
-----
    python src/evaluate.py                      # loads best checkpoint automatically
    python src/evaluate.py --ckpt models/best_model.pt
"""

from src.utils import get_device, load_checkpoint, predict
from src.model import build_model
import config as cfg
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless rendering

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _save(fig, filename: str):
    path = cfg.RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", linewidths=0.5, ax=ax,
        )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)

    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


def plot_roc(y_true, y_probs, class_names, n_classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        auc = roc_auc_score(y_true, y_probs[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
            ax.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "roc_curve.png")


def plot_precision_recall(y_true, y_probs, class_names, n_classes):
    fig, ax = plt.subplots(figsize=(8, 6))
    if n_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_probs[:, 1])
        ap = average_precision_score(y_true, y_probs[:, 1])
        ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, cls in enumerate(class_names):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
            ap = average_precision_score(y_bin[:, i], y_probs[:, i])
            ax.plot(rec, prec, lw=1.5, label=f"{cls} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "pr_curve.png")


def plot_attention(model, X_sample: np.ndarray, y_sample: np.ndarray,
                   class_names, device, n_show: int = 8):
    """Visualise temporal attention weights for a few test samples."""
    model.eval()
    X_t = torch.tensor(X_sample[:n_show], dtype=torch.float32).to(device)
    with torch.no_grad():
        _, weights = model.forward_with_attention(X_t)
    weights = weights.cpu().numpy()

    fig, axes = plt.subplots(n_show, 1, figsize=(12, 2 * n_show), sharex=True)
    for i, ax in enumerate(axes):
        ax.bar(range(weights.shape[1]), weights[i],
               color="steelblue", alpha=0.8)
        label = class_names[y_sample[i]] if y_sample[i] < len(
            class_names) else str(y_sample[i])
        ax.set_title(f"Sample {i}  |  True: {label}", fontsize=9)
        ax.set_ylabel("α", fontsize=8)
        ax.set_ylim(0, weights[i].max() * 1.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("LSTM Temporal Attention Weights", fontsize=13)
    fig.tight_layout()
    _save(fig, "attention_weights.png")


# ─── Core evaluation ──────────────────────────────────────────────────────────

def evaluate_model(model, test_loader: DataLoader, class_names: list[str]):
    device = get_device()
    model.to(device)

    logger.info("Running inference on test set …")
    all_probs, all_preds, all_labels = predict(model, test_loader, device)

    n_classes = len(class_names)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    prec_m = precision_score(all_labels, all_preds,
                             average="macro",    zero_division=0)
    prec_w = precision_score(all_labels, all_preds,
                             average="weighted", zero_division=0)
    rec_m = recall_score(all_labels, all_preds,
                         average="macro",    zero_division=0)
    rec_w = recall_score(all_labels, all_preds,
                         average="weighted", zero_division=0)
    f1_m = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    try:
        if n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(all_labels, classes=list(range(n_classes)))
            auc = roc_auc_score(
                y_bin, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")

    sep = "─" * 55
    logger.info(sep)
    logger.info("  EVALUATION RESULTS  (test set)")
    logger.info(sep)
    logger.info(f"  Accuracy            : {acc:.4f}")
    logger.info(f"  Precision (macro)   : {prec_m:.4f}")
    logger.info(f"  Precision (weighted): {prec_w:.4f}")
    logger.info(f"  Recall    (macro)   : {rec_m:.4f}")
    logger.info(f"  Recall    (weighted): {rec_w:.4f}")
    logger.info(f"  F1        (macro)   : {f1_m:.4f}")
    logger.info(f"  F1        (weighted): {f1_w:.4f}")
    logger.info(f"  Cohen's Kappa       : {kappa:.4f}")
    logger.info(f"  ROC-AUC             : {auc:.4f}")
    logger.info(sep)
    logger.info("\n" + classification_report(all_labels, all_preds,
                                             target_names=class_names, zero_division=0))

    # ── Save metrics to txt ───────────────────────────────────────────────────
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_txt = cfg.RESULTS_DIR / "metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"Accuracy            : {acc:.4f}\n")
        f.write(f"Precision (macro)   : {prec_m:.4f}\n")
        f.write(f"Precision (weighted): {prec_w:.4f}\n")
        f.write(f"Recall    (macro)   : {rec_m:.4f}\n")
        f.write(f"Recall    (weighted): {rec_w:.4f}\n")
        f.write(f"F1        (macro)   : {f1_m:.4f}\n")
        f.write(f"F1        (weighted): {f1_w:.4f}\n")
        f.write(f"Cohen's Kappa       : {kappa:.4f}\n")
        f.write(f"ROC-AUC             : {auc:.4f}\n\n")
        f.write(classification_report(all_labels, all_preds,
                                      target_names=class_names, zero_division=0))
    logger.info(f"Metrics written → {metrics_txt}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if cfg.SAVE_CONFUSION:
        plot_confusion_matrix(all_labels, all_preds, class_names)
    if cfg.SAVE_CURVES:
        plot_roc(all_labels, all_probs, class_names, n_classes)
        plot_precision_recall(all_labels, all_probs, class_names, n_classes)

    # Attention plot (using the first batch of the loader)
    try:
        X_batch, y_batch = next(iter(test_loader))
        plot_attention(model, X_batch.numpy(), y_batch.numpy(), class_names,
                       device, n_show=min(8, len(X_batch)))
    except Exception as e:
        logger.warning(f"Attention plot skipped: {e}")

    return {
        "accuracy": acc, "f1_macro": f1_m, "f1_weighted": f1_w,
        "roc_auc": auc, "kappa": kappa,
    }


# ─── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DDoS LSTM checkpoint.")
    parser.add_argument("--ckpt", type=str,
                        default=str(cfg.MODELS_DIR / "best_model.pt"))
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load processed test data
    with open(cfg.PROCESSED_FILE, "rb") as f:
        data = pickle.load(f)

    device = get_device()
    model = build_model(data["n_features"], data["n_classes"]).to(device)
    load_checkpoint(model, ckpt_path, device)

    ds = TensorDataset(
        torch.tensor(data["X_test"], dtype=torch.float32),
        torch.tensor(data["y_test"], dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    evaluate_model(model, loader, data["class_names"])
