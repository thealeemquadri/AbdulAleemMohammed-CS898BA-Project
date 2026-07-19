"""
Image analysis evaluation: measure the contribution of each processing step.

Design note. The midterm found the full pipeline UNDERPERFORMED the plain-resize
baseline. Running a leave-one-out ablation from that config would only tell us how to
make a bad configuration slightly less bad. Instead this runs an ADDITIVE ablation:
start from the baseline and enable one processing step at a time.

That directly answers the question the final milestone asks, namely how each
pre-processing step contributed to the model's performance, with a signed number per
step rather than a claim.

Conditions:
    baseline_resize_only  plain resize                       (control)
    rgb_crop_only         + circular crop
    rgb_ben_graham        + crop + Ben Graham normalization
    rgb_clahe             + crop + CLAHE
    rgb_full              + crop + Ben Graham + CLAHE + denoise
    full_pipeline         grayscale + morphology stacking    (midterm config)

Usage:
    python -m src.ablation_analysis --run --epochs 8
    python -m src.ablation_analysis --report

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
FIG = "figures"

# (config name, human label, what it adds)
CONDITIONS = [
    ("baseline_resize_only", "Resize only", "control, no domain processing"),
    ("rgb_crop_only",        "+ Circular crop", "removes black borders"),
    ("rgb_ben_graham",       "+ Ben Graham", "illumination normalization"),
    ("rgb_clahe",            "+ CLAHE", "local contrast enhancement"),
    ("rgb_full",             "+ Full colour pipeline", "crop, Ben Graham, CLAHE, denoise"),
    ("full_pipeline",        "Grayscale + morphology", "midterm config, channel stacking"),
]

OX = "#2A1318"; GOLD = "#E6AE4F"; CORAL = "#D85A3C"; BONE = "#F0E7D8"
MUTE = "#AC9089"; HAIR = "#5A3A3E"; GREEN = "#7FA650"


def run_all(epochs, batch_size, img_dir="data/cache"):
    for cfg, label, _ in CONDITIONS:
        path = os.path.join(OUT, f"abl_{cfg}_results.json")
        if os.path.exists(path):
            print(f"  skip {cfg} (already run)")
            continue
        # Reuse the midterm runs where they exist rather than retraining.
        for existing in [f"{OUT}/baseline_results.json", f"{OUT}/full_results.json"]:
            pass
        print("\n" + "#" * 70)
        print(f"#  ABLATION: {cfg}  ({label})")
        print("#" * 70)
        subprocess.run([
            sys.executable, "-m", "src.train",
            "--config", cfg, "--tag", f"abl_{cfg}",
            "--epochs", str(epochs), "--batch_size", str(batch_size),
            "--img_dir", img_dir,
        ], check=True)


def collect():
    """Gather results, preferring dedicated ablation runs, falling back to midterm runs."""
    rows = []
    fallback = {
        "baseline_resize_only": f"{OUT}/baseline_results.json",
        "full_pipeline": f"{OUT}/full_results.json",
        "rgb_full": f"{OUT}/rgb_full_results.json",
    }
    for cfg, label, desc in CONDITIONS:
        path = os.path.join(OUT, f"abl_{cfg}_results.json")
        if not os.path.exists(path):
            path = fallback.get(cfg, "")
        if not path or not os.path.exists(path):
            print(f"  missing results for {cfg}")
            continue
        r = json.load(open(path))
        rows.append({
            "config": cfg, "label": label, "desc": desc,
            "qwk": r["final_qwk"], "acc": r["final_accuracy"],
            "sens": r.get("sensitivity", [None] * 5),
        })
    return rows


def report(rows):
    if not rows:
        print("No results found. Run with --run first.")
        return

    base = next((r for r in rows if r["config"] == "baseline_resize_only"), None)
    b = base["qwk"] if base else None

    print("\n" + "=" * 84)
    print("  IMAGE ANALYSIS EVALUATION: contribution of each processing step")
    print("=" * 84)
    print(f"  {'Condition':<26}{'QWK':>9}{'Acc':>9}{'dQWK':>10}   {'Verdict':<12}")
    print("  " + "-" * 80)
    for r in rows:
        if b is not None and r["config"] != "baseline_resize_only":
            d = r["qwk"] - b
            verdict = "HELPS" if d > 0.002 else ("HURTS" if d < -0.002 else "neutral")
            ds = f"{d:+.4f}"
        else:
            ds, verdict = "-", "control"
        print(f"  {r['label']:<26}{r['qwk']:>9.4f}{r['acc']:>9.4f}{ds:>10}   {verdict:<12}")
    print("=" * 84)

    helped = [r for r in rows if b and r["qwk"] - b > 0.002]
    if helped:
        best = max(helped, key=lambda r: r["qwk"])
        print(f"  Only step(s) that improved on the baseline: "
              f"{', '.join(r['label'] for r in helped)}")
        print(f"  Best configuration: {best['label']} at QWK {best['qwk']:.4f}")
    else:
        print("  No processing step improved on the plain-resize baseline.")
        print("  Interpretation: on a strong ImageNet-pretrained backbone with clean")
        print("  clinical images, the enhancement removes more information than it adds.")
    print("=" * 84 + "\n")

    json.dump(rows, open(os.path.join(OUT, "ablation_analysis.json"), "w"), indent=2)
    return rows


def figure(rows):
    """Waterfall-style bar chart of each step's effect relative to the baseline."""
    if not rows:
        return
    base = next((r for r in rows if r["config"] == "baseline_resize_only"), None)
    if not base:
        return
    b = base["qwk"]

    plt.rcParams.update({
        "figure.facecolor": OX, "axes.facecolor": OX, "savefig.facecolor": OX,
        "text.color": BONE, "axes.labelcolor": BONE,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "axes.edgecolor": HAIR, "font.family": "serif",
    })

    others = [r for r in rows if r["config"] != "baseline_resize_only"]
    labels = [r["label"] for r in others]
    deltas = [r["qwk"] - b for r in others]
    colors = [GREEN if d > 0.002 else (CORAL if d < -0.002 else MUTE) for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(others)), deltas, color=colors,
                   edgecolor=BONE, linewidth=0.6, height=0.6)
    for i, (bar, d, r) in enumerate(zip(bars, deltas, others)):
        off = 0.0008 if d >= 0 else -0.0008
        ha = "left" if d >= 0 else "right"
        ax.text(d + off, i, f"{d:+.4f}  (QWK {r['qwk']:.4f})",
                va="center", ha=ha, color=BONE, fontsize=10)

    ax.axvline(0, color=GOLD, lw=1.5)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Change in QWK relative to the plain-resize baseline")
    ax.set_title(f"Measured contribution of each image processing step\n"
                 f"(baseline QWK {b:.4f})", color=BONE, fontsize=13, pad=12)
    lim = max(abs(min(deltas)), abs(max(deltas))) * 1.55
    ax.set_xlim(-lim, lim)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.12, color=MUTE)

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "ablation_contribution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Train all ablation conditions.")
    ap.add_argument("--report", action="store_true", help="Print table and build figure.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_dir", default="data/cache")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    if args.run:
        run_all(args.epochs, args.batch_size, args.img_dir)
    rows = collect()
    report(rows)
    figure(rows)


if __name__ == "__main__":
    main()
