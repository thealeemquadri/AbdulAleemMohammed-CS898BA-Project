"""
Generate presentation figures from REAL data and REAL training results.

Produces:
    figures/pipeline_demo.png    - the actual preprocessing chain on a real fundus image
    figures/class_dist.png       - real class distribution from train.csv
    figures/training_curves.png  - QWK/loss per epoch from a results json
    figures/confusion_matrix.png - confusion matrix from a results json

Usage:
    python -m src.figures --csv data/train.csv --img_dir data/train_images \
        --results outputs/full_pipeline_results.json

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing.fundus import (
    circular_crop, ben_graham, green_channel, apply_clahe, tophat_bothat, denoise,
)
from src.data.dataset import CLASS_NAMES

# Presentation palette (matches the slide deck).
OX = "#2A1318"; GOLD = "#E6AE4F"; CORAL = "#D85A3C"; BONE = "#F0E7D8"
MUTE = "#AC9089"; HAIR = "#5A3A3E"

plt.rcParams.update({
    "figure.facecolor": OX, "axes.facecolor": OX, "savefig.facecolor": OX,
    "text.color": BONE, "axes.labelcolor": BONE,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": HAIR, "font.family": "serif",
})

FIG_DIR = "figures"


def fig_pipeline_demo(csv_path, img_dir, severity=3):
    """
    THE key figure: run the real pipeline on a real fundus image, stage by stage.
    Picks an image of the requested severity so lesions are actually visible.
    """
    df = pd.read_csv(csv_path)
    subset = df[df["diagnosis"] == severity]
    if len(subset) == 0:
        subset = df
    row = subset.iloc[0]

    path = os.path.join(img_dir, f"{row['id_code']}.png")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cropped = circular_crop(img)
    bg = ben_graham(cropped)
    g = green_channel(bg)
    cl = apply_clahe(g)
    dn = denoise(cl)
    th, bh = tophat_bothat(dn)

    panels = [
        (img, None, "Raw fundus", "As captured: uneven light, low lesion contrast"),
        (cropped, None, "1  Circular crop", "Black borders removed, retina isolated"),
        (bg, None, "2  Ben Graham", "Illumination normalized across the retina"),
        (cl, "gray", "3  Green + CLAHE", "Lesion contrast amplified locally"),
        (th, "hot", "4  Top-hat", "Bright lesions (exudates) isolated"),
        (bh, "hot", "5  Bottom-hat", "Dark lesions (hemorrhages) isolated"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(19, 4.1))
    for ax, (im, cm, title, desc) in zip(axes, panels):
        ax.imshow(im, cmap=cm) if cm else ax.imshow(im)
        ax.set_title(title, color=GOLD, fontsize=13, fontweight="bold", loc="left", pad=6)
        ax.text(0.0, -0.06, desc, transform=ax.transAxes, ha="left", va="top",
                color=MUTE, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(HAIR)

    fig.suptitle(f"Image processing pipeline on a real grade-{severity} fundus image "
                 f"(id: {row['id_code']})",
                 color=BONE, fontsize=14, y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "pipeline_demo.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def fig_class_dist(csv_path):
    """Real class distribution, the class-imbalance roadblock."""
    df = pd.read_csv(csv_path)
    counts = df["diagnosis"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = [MUTE, GOLD, GOLD, CORAL, CORAL]
    bars = ax.bar(range(len(counts)), counts.values, color=colors,
                  edgecolor=BONE, linewidth=0.6)
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c + max(counts) * 0.015, str(c),
                ha="center", color=BONE, fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{i}\n{CLASS_NAMES[i]}" for i in counts.index], fontsize=10)
    ax.set_ylabel("Training images")
    ax.set_title("APTOS 2019 class distribution", color=BONE, fontsize=13, pad=10)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "class_dist.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}  counts={counts.values.tolist()}")


def fig_training_curves(results_path):
    """QWK and loss per epoch, from the real run."""
    with open(results_path) as f:
        r = json.load(f)
    h = r["history"]
    ep = [x["epoch"] for x in h]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))

    a1.plot(ep, [x["train_loss"] for x in h], color=MUTE, marker="o",
            ms=4, label="train loss")
    a1.plot(ep, [x["val_loss"] for x in h], color=CORAL, marker="o",
            ms=4, label="val loss")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss")
    a1.set_title("Loss", color=BONE, fontsize=12)
    a1.legend(facecolor=OX, edgecolor=HAIR, labelcolor=BONE)
    a1.grid(alpha=0.12, color=MUTE)

    a2.plot(ep, [x["val_qwk"] for x in h], color=GOLD, marker="o", ms=5,
            label="val QWK")
    best = max(x["val_qwk"] for x in h)
    a2.axhline(best, color=HAIR, ls="--", lw=1)
    a2.text(ep[0], best, f" best {best:.4f}", color=GOLD, fontsize=10, va="bottom")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Quadratic Weighted Kappa")
    a2.set_title("Validation QWK", color=BONE, fontsize=12)
    a2.grid(alpha=0.12, color=MUTE)

    for ax in (a1, a2):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def fig_confusion(results_path):
    """Confusion matrix from the real run."""
    with open(results_path) as f:
        r = json.load(f)
    cm = np.array(r["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap="inferno")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v < cm.max() * 0.6 else "black", fontsize=10)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(range(5)); ax.set_yticklabels(range(5))
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
    ax.set_title(f"Confusion matrix  (QWK {r['final_qwk']:.4f})",
                 color=BONE, fontsize=12, pad=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/train.csv")
    p.add_argument("--img_dir", default="data/train_images")
    p.add_argument("--results", default=None,
                   help="Path to a *_results.json to plot curves and confusion matrix.")
    p.add_argument("--severity", type=int, default=3)
    args = p.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    fig_class_dist(args.csv)
    fig_pipeline_demo(args.csv, args.img_dir, severity=args.severity)

    if args.results and os.path.exists(args.results):
        fig_training_curves(args.results)
        fig_confusion(args.results)
    else:
        print("No results json given, skipping curves and confusion matrix.")


if __name__ == "__main__":
    main()
