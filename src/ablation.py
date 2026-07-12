"""
Ablation study: measure what each image processing step actually contributes.

This is the evidence for the central claim of the project, that domain image
processing (not augmentation) drives the performance. It trains the identical model
and schedule under different preprocessing configurations and compares QWK.

The key comparison is:
    baseline_resize_only   vs   full_pipeline

For the midterm, running just those two is enough. The leave-one-out configs
(no_clahe, no_ben_graham, ...) attribute credit to individual steps for the final.

Usage:
    python -m src.ablation --configs baseline_resize_only full_pipeline --epochs 12

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Run preprocessing ablation study")
    p.add_argument("--configs", nargs="+",
                   default=["baseline_resize_only", "full_pipeline"],
                   help="Which preprocessing configs to train and compare.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--csv", default="data/train.csv")
    p.add_argument("--img_dir", default="data/train_images")
    p.add_argument("--out_dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Train each configuration as a separate run.
    for cfg in args.configs:
        print("\n" + "#" * 70)
        print(f"#  ABLATION RUN: {cfg}")
        print("#" * 70)
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", cfg,
            "--tag", cfg,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--csv", args.csv,
            "--img_dir", args.img_dir,
            "--out_dir", args.out_dir,
        ]
        subprocess.run(cmd, check=True)

    # Collect results into one comparison table.
    rows = []
    for cfg in args.configs:
        path = os.path.join(args.out_dir, f"{cfg}_results.json")
        if not os.path.exists(path):
            print(f"WARNING: missing results for {cfg}")
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "config": cfg,
            "qwk": r["final_qwk"],
            "accuracy": r["final_accuracy"],
            "sens_severe": r["sensitivity"][3],
            "sens_prolif": r["sensitivity"][4],
        })

    baseline = next((r for r in rows if r["config"] == "baseline_resize_only"), None)

    print("\n" + "=" * 78)
    print("  ABLATION SUMMARY")
    print("=" * 78)
    print(f"  {'Configuration':<24}{'QWK':>9}{'Acc':>9}{'Sev.Sens':>11}{'Prol.Sens':>11}{'dQWK':>10}")
    print("  " + "-" * 74)
    for r in rows:
        d = ""
        if baseline and r["config"] != "baseline_resize_only":
            d = f"{r['qwk'] - baseline['qwk']:+.4f}"
        print(f"  {r['config']:<24}{r['qwk']:>9.4f}{r['accuracy']:>9.4f}"
              f"{r['sens_severe']:>11.3f}{r['sens_prolif']:>11.3f}{d:>10}")
    print("=" * 78)
    print("  dQWK = change in QWK relative to the resize-only baseline.")
    print("  A positive dQWK for full_pipeline is direct evidence that the domain")
    print("  image processing, not augmentation, is doing the work.\n")

    out = os.path.join(args.out_dir, "ablation_summary.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
