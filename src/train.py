"""
Training entry point for DR grading.

Usage (baseline, no domain processing):
    python -m src.train --config baseline_resize_only --epochs 12 --tag baseline

Usage (full image processing pipeline):
    python -m src.train --config full_pipeline --epochs 12 --tag full

The --config flag selects a preprocessing configuration from ABLATION_CONFIGS.
Running the same model and schedule under different configs is what isolates the
contribution of the image processing, which is the central claim of this project.

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.preprocessing.fundus import ABLATION_CONFIGS
from src.data.dataset import build_loaders, class_weights_tensor
from src.models.model import build_model
from src.evaluate import evaluate, print_report


def parse_args():
    p = argparse.ArgumentParser(description="Train DR grading model")
    p.add_argument("--csv", default="data/train.csv")
    p.add_argument("--img_dir", default="data/train_images")
    p.add_argument("--config", default="full_pipeline",
                   choices=list(ABLATION_CONFIGS.keys()))
    p.add_argument("--backbone", default="efficientnet_b3")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balanced", action="store_true", default=True,
                   help="Use the weighted sampler to counter class imbalance.")
    p.add_argument("--no_balanced", dest="balanced", action="store_false")
    p.add_argument("--tag", default="run", help="Name for the output files.")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


def set_seed(seed):
    """Reproducibility. The professor should be able to rerun and get our numbers."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ABLATION_CONFIGS[args.config]

    print(f"\nDevice           : {device}")
    print(f"Preprocess config: {args.config}")
    print(f"  {cfg}")
    print(f"Backbone         : {args.backbone}")
    print(f"Epochs           : {args.epochs}   Batch: {args.batch_size}   LR: {args.lr}")
    print(f"Class balancing  : {args.balanced}\n")

    train_loader, val_loader, train_df, val_df = build_loaders(
        args.csv, args.img_dir, cfg,
        size=args.size, batch_size=args.batch_size,
        seed=args.seed, balanced=args.balanced,
    )
    print(f"Train images: {len(train_df)}   Val images: {len(val_df)}")
    print(f"Train class counts: {train_df['diagnosis'].value_counts().sort_index().tolist()}\n")

    model = build_model(args.backbone, pretrained=True,
                        dropout=args.dropout, device=device)

    # Class-weighted loss: rebalances the COST of errors (the sampler rebalances what
    # the model sees). Using both is deliberate and addresses the imbalance roadblock.
    weights = class_weights_tensor(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # Cosine annealing: high LR early to adapt the head, low LR late to fine-tune
    # the pretrained features without destroying them.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_qwk = -1.0
    history = []
    ckpt_path = os.path.join(args.out_dir, f"{args.tag}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            n += 1

        scheduler.step()
        train_loss = running / max(n, 1)

        m = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0

        print(f"Epoch {epoch:2d}/{args.epochs}  "
              f"train_loss {train_loss:.4f}  "
              f"val_loss {m['loss']:.4f}  "
              f"val_acc {m['accuracy']:.4f}  "
              f"val_QWK {m['qwk']:.4f}  ({dt:.0f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": m["loss"],
            "val_acc": float(m["accuracy"]),
            "val_qwk": float(m["qwk"]),
        })

        if m["qwk"] > best_qwk:
            best_qwk = m["qwk"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"           new best QWK {best_qwk:.4f}, checkpoint saved")

    # Final report using the best checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    final = evaluate(model, val_loader, device, criterion)
    print_report(final, title=f"FINAL ({args.config})")

    results = {
        "config": args.config,
        "preprocess_flags": cfg,
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "balanced": args.balanced,
        "best_qwk": float(best_qwk),
        "final_qwk": float(final["qwk"]),
        "final_accuracy": float(final["accuracy"]),
        "sensitivity": final["sensitivity"].tolist(),
        "specificity": final["specificity"].tolist(),
        "confusion_matrix": final["confusion_matrix"].tolist(),
        "history": history,
    }
    res_path = os.path.join(args.out_dir, f"{args.tag}_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {res_path}")
    print(f"Checkpoint at      {ckpt_path}")


if __name__ == "__main__":
    main()
