"""
EfficientNet-B3 backbone with a classification head for 5-class DR grading.

Why transfer learning: with only ~2,900 training images, a from-scratch CNN cannot
learn robust low-level filters (edges, textures, blobs). ImageNet pretraining supplies
those, and we fine-tune the whole network on fundus data.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import torch
import torch.nn as nn
import timm


class DRModel(nn.Module):
    """
    Args:
        backbone: any timm model name. Default efficientnet_b3.
        num_classes: 5 severity grades.
        pretrained: load ImageNet weights (the whole point of the design choice).
        dropout: regularization before the classifier, important on a small dataset.
    """

    def __init__(self, backbone="efficientnet_b3", num_classes=5,
                 pretrained=True, dropout=0.3):
        super().__init__()
        # num_classes=0 strips timm's own classifier and gives us pooled features.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)      # global-pooled feature vector
        return self.head(feats)       # logits, (B, 5)


def build_model(backbone="efficientnet_b3", num_classes=5, pretrained=True,
                dropout=0.3, device="cuda"):
    model = DRModel(backbone, num_classes, pretrained, dropout)
    return model.to(device)
