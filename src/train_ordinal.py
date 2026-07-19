"""
Train the ordinal regression variant.

Same backbone, same data, same schedule as the classification model. The only change
is the head (single continuous output, MSE loss) and the decision rule (thresholds
fitted to maximise QWK on the validation set).

Usage:
    python -m src.train_ordinal --epochs 15 --tag ordinal

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import json
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from src.preprocessing.fundus import ABLATION_CONFIGS
from src.data.dataset import build_loaders, CLASS_NAMES, NUM_CLASSES
from src.ordinal import OrdinalHead, optimize_thresholds, apply_thresholds, collect_scores
from src.evaluate import per_class_sensitivity, per_class_specificity


class OrdinalModel(nn.Module):
    def __init__(self, backbone="efficientnet_b3", pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = OrdinalHead(self.backbone.num_features, dropout)

    def forward(self, x):
        return self.head(self.backbone(x))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/train.csv")
    p.add_argument("--img_dir", default="data/cache")
    p.add_argument("--config", default="baseline_resize_only")
    p.add_argument("--backbone", default="efficientnet_b3")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="ordinal")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ABLATION_CONFIGS[args.config]

    print(f"\nDevice   : {device}")
    print(f"Head     : ordinal regression (1 output, MSE) + fitted thresholds")
    print(f"Config   : {args.config}")
    print(f"Epochs   : {args.epochs}  Batch: {args.batch_size}  LR: {args.lr}\n")

    train_loader, val_loader, train_df, val_df = build_loaders(
        args.csv, args.img_dir, cfg, size=args.size,
        batch_size=args.batch_size, seed=args.seed, balanced=True,
    )
    print(f"Train {len(train_df)}  Val {len(val_df)}\n")

    model = OrdinalModel(args.backbone, True, args.dropout).to(device)

    # MSE on the grade value. Being 1 grade off costs 1, being 2 off costs 4,
    # which mirrors the quadratic weighting of the evaluation metric.
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_qwk, best_thresholds = -1.0, [0.5, 1.5, 2.5, 3.5]
    history = []
    ckpt = os.path.join(args.out_dir, f"{args.tag}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                preds = model(imgs)
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n += 1

        scheduler.step()

        scores, labels = collect_scores(model, val_loader, device)
        opt = optimize_thresholds(scores, labels)
        qwk = opt["qwk_optimized"]
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs}  mse {running/max(n,1):.4f}  "
              f"QWK(naive) {opt['qwk_naive_rounding']:.4f}  "
              f"QWK(opt) {qwk:.4f}  ({dt:.0f}s)")

        history.append({
            "epoch": epoch,
            "train_mse": running / max(n, 1),
            "val_qwk_naive": opt["qwk_naive_rounding"],
            "val_qwk": qwk,
        })

        if qwk > best_qwk:
            best_qwk = qwk
            best_thresholds = opt["thresholds"]
            torch.save(model.state_dict(), ckpt)
            print(f"           new best QWK {best_qwk:.4f}  "
                  f"thresholds {[round(t,3) for t in best_thresholds]}")

    # Final evaluation with the best checkpoint and its thresholds.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    scores, labels = collect_scores(model, val_loader, device)
    opt = optimize_thresholds(scores, labels)
    preds = apply_thresholds(scores, opt["thresholds"])

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    sens = per_class_sensitivity(cm)
    spec = per_class_specificity(cm)
    acc = accuracy_score(labels, preds)
    qwk = opt["qwk_optimized"]

    print("\n" + "=" * 64)
    print("  ORDINAL REGRESSION, FINAL")
    print("=" * 64)
    print(f"  QWK (optimized thresholds) : {qwk:.4f}")
    print(f"  QWK (naive rounding)       : {opt['qwk_naive_rounding']:.4f}")
    print(f"  Gain from threshold fitting: {opt['gain_from_thresholds']:+.4f}")
    print(f"  Accuracy                   : {acc:.4f}")
    print(f"  Thresholds                 : {[round(t,3) for t in opt['thresholds']]}")
    print("\n  Per-class sensitivity / specificity:")
    for i, nm in enumerate(CLASS_NAMES):
        print(f"    {i} {nm:<14}{sens[i]:>8.3f}{spec[i]:>10.3f}")
    print("\n  Confusion matrix (rows true, cols predicted):")
    for i, row in enumerate(cm):
        print(f"    true {i}" + "".join(f"{v:>7}" for v in row))
    print("=" * 64 + "\n")

    json.dump({
        "config": args.config,
        "head": "ordinal_regression",
        "backbone": args.backbone,
        "epochs": args.epochs,
        "lr": args.lr,
        "size": args.size,
        "dropout": args.dropout,
        "final_qwk": float(qwk),
        "qwk_naive_rounding": float(opt["qwk_naive_rounding"]),
        "gain_from_thresholds": float(opt["gain_from_thresholds"]),
        "thresholds": opt["thresholds"],
        "final_accuracy": float(acc),
        "sensitivity": sens.tolist(),
        "specificity": spec.tolist(),
        "confusion_matrix": cm.tolist(),
        "history": history,
    }, open(os.path.join(args.out_dir, f"{args.tag}_results.json"), "w"), indent=2)

    print(f"Results -> {args.out_dir}/{args.tag}_results.json")


if __name__ == "__main__":
    main()
