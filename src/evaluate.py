"""
Evaluation metrics for ordinal DR grading.

Primary metric is Quadratic Weighted Kappa (QWK), the official APTOS metric.
Accuracy is reported but is misleading here: predicting "No DR" for every image
scores ~49% accuracy while being clinically worthless. QWK penalizes predictions
that are further from the true grade quadratically, and corrects for chance agreement.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, accuracy_score, classification_report,
)

from src.data.dataset import CLASS_NAMES, NUM_CLASSES


def quadratic_weighted_kappa(y_true, y_pred):
    """Cohen's kappa with quadratic weights. 1.0 = perfect, 0.0 = chance."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def per_class_sensitivity(cm):
    """
    Sensitivity (recall) per class = TP / (TP + FN).
    This is the clinically important number: a model that misses Severe cases is
    dangerous no matter how good its overall accuracy looks.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(sens)


def per_class_specificity(cm):
    """Specificity per class = TN / (TN + FP)."""
    spec = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.array(spec)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """
    Run the model over a loader and return a dict of metrics plus raw predictions.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        if criterion is not None:
            total_loss += criterion(logits, labels).item()
            n_batches += 1

        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "loss": (total_loss / n_batches) if n_batches else None,
        "confusion_matrix": cm,
        "sensitivity": per_class_sensitivity(cm),
        "specificity": per_class_specificity(cm),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_report(metrics, title="Validation"):
    """Human-readable dump, safe to paste straight into the presentation."""
    print("\n" + "=" * 62)
    print(f"  {title} results")
    print("=" * 62)
    print(f"  Quadratic Weighted Kappa : {metrics['qwk']:.4f}   <-- primary metric")
    print(f"  Accuracy                 : {metrics['accuracy']:.4f}")
    if metrics.get("loss") is not None:
        print(f"  Loss                     : {metrics['loss']:.4f}")

    print("\n  Per-class sensitivity (recall) and specificity:")
    print(f"  {'Grade':<16}{'Sensitivity':>13}{'Specificity':>13}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:<14}{metrics['sensitivity'][i]:>13.3f}"
              f"{metrics['specificity'][i]:>13.3f}")

    print("\n  Confusion matrix (rows = true, cols = predicted):")
    header = "        " + "".join(f"{i:>7}" for i in range(NUM_CLASSES))
    print(header)
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"  true {i}" + "".join(f"{v:>7}" for v in row))
    print("=" * 62 + "\n")
