"""
Ordinal regression head with optimized decision thresholds.

Motivation. The evaluation metric is Quadratic Weighted Kappa, which is ordinal: it
penalises predicting grade 0 for a grade-4 patient far more than predicting grade 3.
But a 5-way softmax classifier treats the grades as five unrelated categories. It has
no notion that grade 3 sits between 2 and 4, so it cannot exploit the very structure
the metric rewards.

This module replaces the classification head with a single continuous output trained
with MSE, then learns four decision thresholds on the validation set that convert that
score into a grade. Because the thresholds are fitted to maximise QWK directly, this
optimises the actual objective rather than a proxy.

This is the standard approach among strong APTOS solutions, and it is the natural
follow-up to the midterm observation that the grades are ordered.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


class OrdinalHead(nn.Module):
    """Single continuous output. The model predicts severity on a 0 to 4 scale."""

    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def apply_thresholds(scores, thresholds):
    """
    Convert continuous scores to integer grades.

    thresholds is a sorted list of 4 cut points. A score below the first becomes
    grade 0, between the first and second becomes grade 1, and so on.
    """
    t = np.sort(np.asarray(thresholds))
    return np.digitize(np.asarray(scores), t).astype(int)


def optimize_thresholds(scores, labels, init=(0.5, 1.5, 2.5, 3.5)):
    """
    Fit the four cut points that maximise QWK on the validation set.

    Nelder-Mead is used because the objective (QWK after digitisation) is piecewise
    constant, so it has no useful gradient. The initial guess is the naive rounding
    boundaries, which is what simple rounding would do.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def neg_qwk(t):
        preds = apply_thresholds(scores, t)
        return -cohen_kappa_score(labels, preds, weights="quadratic")

    res = minimize(neg_qwk, np.array(init, dtype=float), method="nelder-mead",
                   options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 2000})

    best_t = np.sort(res.x)
    best_qwk = -res.fun

    naive_qwk = cohen_kappa_score(labels, apply_thresholds(scores, init),
                                  weights="quadratic")

    return {
        "thresholds": best_t.tolist(),
        "qwk_optimized": float(best_qwk),
        "qwk_naive_rounding": float(naive_qwk),
        "gain_from_thresholds": float(best_qwk - naive_qwk),
    }


@torch.no_grad()
def collect_scores(model, loader, device):
    """Run the model over a loader and return (continuous scores, true labels)."""
    model.eval()
    scores, labels = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        scores.append(out.detach().cpu().numpy().ravel())
        labels.append(ys.numpy().ravel())
    return np.concatenate(scores), np.concatenate(labels)
