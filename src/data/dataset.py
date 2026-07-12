"""
APTOS 2019 dataset loading, stratified splitting, and class imbalance handling.

The dataset is severely imbalanced (roughly 1805 / 370 / 999 / 193 / 295 across
grades 0-4). Left untouched, a model can score ~49% accuracy by predicting "No DR"
for everything while being clinically useless. We address this in two independent ways:

  1. WeightedRandomSampler   - rebalances what the model SEES during training.
  2. Class-weighted loss     - rebalances how much each mistake COSTS.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.preprocessing.fundus import preprocess, preprocess_rgb

# ImageNet statistics: the EfficientNet backbone was pretrained with these,
# so inputs must be normalized the same way for the transferred features to be valid.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


class APTOSDataset(Dataset):
    """
    Args:
        df: DataFrame with columns [id_code, diagnosis].
        img_dir: directory holding <id_code>.png files.
        preprocess_cfg: dict of toggles passed to preprocess(). This is what makes
            the ablation study possible: the same Dataset class serves every condition.
        size: output resolution.
        train: if True, apply light augmentation AFTER the domain processing.
            Augmentation is a supplement here, not the processing strategy itself.
    """

    def __init__(self, df, img_dir, preprocess_cfg, size=300, train=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.cfg = preprocess_cfg
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.df)

    def _augment(self, img):
        """Light geometric augmentation. Fundus images have no canonical orientation,
        so flips and small rotations are label-preserving."""
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 0)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['id_code']}.png")

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Domain image processing (the graded part of this project).
        cfg = dict(self.cfg)
        if cfg.pop("rgb", False):
            img = preprocess_rgb(img, size=self.size, **cfg)
        else:
            img = preprocess(img, size=self.size, **cfg)

        # Augmentation, applied only at training time and only on top of processing.
        if self.train:
            img = self._augment(img)

        # To normalized CHW float tensor.
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))

        label = int(row["diagnosis"])
        return torch.from_numpy(img).float(), torch.tensor(label, dtype=torch.long)


def stratified_split(csv_path, val_size=0.2, seed=42):
    """
    Stratified split preserves the class ratio in both sets. With only 193 Severe
    images, a random split could leave the validation set with almost none, making
    the metric meaningless.
    """
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["diagnosis"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_sampler(train_df):
    """
    WeightedRandomSampler: sample each image with probability inversely proportional
    to its class frequency, so each batch is roughly class-balanced. This fixes the
    model's tendency to collapse onto the majority class.
    """
    counts = train_df["diagnosis"].value_counts().sort_index().values
    class_weights = 1.0 / counts
    sample_weights = class_weights[train_df["diagnosis"].values]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_df),
        replacement=True,
    )


def class_weights_tensor(train_df, device):
    """Inverse-frequency weights for the loss function, normalized to mean 1."""
    counts = train_df["diagnosis"].value_counts().sort_index().values.astype(np.float32)
    w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def build_loaders(csv_path, img_dir, preprocess_cfg, size=300, batch_size=16,
                  val_size=0.2, seed=42, balanced=True, num_workers=8):
    """Returns (train_loader, val_loader, train_df, val_df)."""
    train_df, val_df = stratified_split(csv_path, val_size=val_size, seed=seed)

    train_ds = APTOSDataset(train_df, img_dir, preprocess_cfg, size=size, train=True)
    val_ds = APTOSDataset(val_df, img_dir, preprocess_cfg, size=size, train=False)

    sampler = make_sampler(train_df) if balanced else None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_df, val_df
