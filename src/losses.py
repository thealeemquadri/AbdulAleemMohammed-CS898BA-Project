"""
Loss functions for imbalanced ordinal DR grading.

The midterm identified the binding constraint: Severe (grade 3) sensitivity was 0.385,
meaning the model missed roughly 6 in 10 severe cases despite a weighted sampler and
class-weighted cross-entropy. Both of those reweight by CLASS frequency, which does
nothing about the fact that most individual examples are easy.

Focal loss reweights by example DIFFICULTY instead, down-weighting the well-classified
majority so gradient signal concentrates on the hard, rare cases.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017), multi-class form.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The modulating factor (1 - p_t)^gamma shrinks the loss for confident correct
    predictions. With gamma=2, an example predicted correctly at p=0.9 contributes
    100x less loss than one at p=0.5, so training focuses on what it is getting wrong.

    Args:
        alpha: optional per-class weight tensor. Combines class-frequency weighting
            with difficulty weighting.
        gamma: focusing strength. 0 reduces to cross-entropy; 2 is the standard value.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Per-example cross-entropy (no reduction) with optional class weights.
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")

        # p_t = probability the model assigned to the TRUE class.
        # exp(-ce) recovers it directly, and is numerically stabler than softmax+gather.
        pt = torch.exp(-ce)

        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(name, class_weights=None, gamma=2.0):
    """
    Factory used by the training script.

    Args:
        name: "weighted_ce" or "focal".
        class_weights: per-class weight tensor (inverse frequency).
    """
    if name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    return nn.CrossEntropyLoss(weight=class_weights)
