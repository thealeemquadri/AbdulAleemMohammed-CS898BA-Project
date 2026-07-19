"""
Hyperparameter optimization for DR grading.

Runs a staged sweep rather than a blind grid. Each stage varies ONE factor while
holding everything else fixed, so the measured change in QWK is attributable to that
factor. The best value found in a stage is carried into the next stage.

This is deliberately not a random or exhaustive search: with roughly 25 minutes per
run on a single Colab GPU, a staged sweep buys interpretable attribution (which the
final report needs) for a fraction of the compute.

Search stages:
    1. learning rate      1e-4, 3e-4, 1e-3
    2. input resolution   300, 380
    3. loss function      weighted cross-entropy, focal loss
    4. dropout            0.3, 0.5

Usage:
    python -m src.hpo --stage all --epochs 8
    python -m src.hpo --stage lr  --epochs 8      # single stage
    python -m src.hpo --final --epochs 15         # retrain best config, longer

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import json
import os
import subprocess
import sys
import time

RESULTS = "outputs"
STATE = os.path.join(RESULTS, "hpo_state.json")


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    # Defaults are the midterm baseline configuration.
    return {
        "config": "baseline_resize_only",
        "lr": 3e-4,
        "size": 300,
        "loss": "weighted_ce",
        "dropout": 0.3,
        "batch_size": 16,
        "img_dir": "data/cache",
        "history": [],
    }


def save_state(s):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=2)


def run_one(tag, state, epochs, **overrides):
    """Train one configuration and return its final QWK."""
    p = dict(state)
    p.update(overrides)

    cmd = [
        sys.executable, "-m", "src.train",
        "--config", p["config"],
        "--tag", tag,
        "--epochs", str(epochs),
        "--batch_size", str(p["batch_size"]),
        "--lr", str(p["lr"]),
        "--size", str(p["size"]),
        "--dropout", str(p["dropout"]),
        "--loss", p["loss"],
        "--img_dir", p.get("img_dir", "data/cache"),
    ]

    print("\n" + "#" * 72)
    print(f"#  HPO RUN: {tag}")
    print(f"#  lr={p['lr']}  size={p['size']}  loss={p['loss']}  dropout={p['dropout']}")
    print("#" * 72)

    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = (time.time() - t0) / 60

    res_path = os.path.join(RESULTS, f"{tag}_results.json")
    r = json.load(open(res_path))
    qwk = r["final_qwk"]
    print(f"\n  -> {tag}: QWK {qwk:.4f}  ({dt:.1f} min)")
    return qwk, r


def stage_sweep(state, name, param, values, epochs):
    """Vary one parameter, keep the best, record every trial."""
    print("\n" + "=" * 72)
    print(f"  STAGE: {name}   ({param} in {values})")
    print("=" * 72)

    trials = []
    for v in values:
        tag = f"hpo_{param}_{str(v).replace('.', '').replace('-', '')}"
        qwk, _ = run_one(tag, state, epochs, **{param: v})
        trials.append({"param": param, "value": v, "qwk": qwk, "tag": tag})

    best = max(trials, key=lambda t: t["qwk"])
    baseline_val = state[param]
    baseline_trial = next((t for t in trials if t["value"] == baseline_val), None)

    print("\n  " + "-" * 66)
    print(f"  {name} results:")
    for t in trials:
        mark = "  <-- BEST" if t is best else ""
        delta = ""
        if baseline_trial and t is not baseline_trial:
            delta = f"  ({t['qwk'] - baseline_trial['qwk']:+.4f})"
        print(f"    {param}={str(t['value']):<14} QWK {t['qwk']:.4f}{delta}{mark}")
    print("  " + "-" * 66)

    state[param] = best["value"]
    state["history"].append({
        "stage": name, "param": param, "trials": trials,
        "chosen": best["value"], "best_qwk": best["qwk"],
    })
    save_state(state)
    print(f"  Carrying forward: {param} = {best['value']}\n")
    return state


def main():
    ap = argparse.ArgumentParser(description="Hyperparameter optimization")
    ap.add_argument("--stage", default="all",
                    choices=["all", "lr", "size", "loss", "dropout"])
    ap.add_argument("--epochs", type=int, default=8,
                    help="Epochs per search run. Short is fine for ranking configs.")
    ap.add_argument("--final", action="store_true",
                    help="Retrain the best-found config for longer.")
    ap.add_argument("--final_epochs", type=int, default=15)
    ap.add_argument("--reset", action="store_true", help="Clear saved HPO state.")
    args = ap.parse_args()

    if args.reset and os.path.exists(STATE):
        os.remove(STATE)
        print("HPO state cleared.")

    state = load_state()

    if args.final:
        print("\nRetraining the best configuration found:")
        for k in ["lr", "size", "loss", "dropout"]:
            print(f"    {k}: {state[k]}")
        qwk, r = run_one("final_optimized", state, args.final_epochs)

        base_path = os.path.join(RESULTS, "baseline_results.json")
        if os.path.exists(base_path):
            b = json.load(open(base_path))["final_qwk"]
            print("\n" + "=" * 72)
            print("  FINAL OPTIMIZED vs MIDTERM BASELINE")
            print("=" * 72)
            print(f"  Midterm baseline QWK : {b:.4f}")
            print(f"  Final optimized QWK  : {qwk:.4f}")
            print(f"  Improvement          : {qwk - b:+.4f}")
            print("=" * 72)
        return

    stages = {
        "lr":      ("Learning rate",     "lr",      [1e-4, 3e-4, 1e-3]),
        "size":    ("Input resolution",  "size",    [300, 380]),
        "loss":    ("Loss function",     "loss",    ["weighted_ce", "focal"]),
        "dropout": ("Dropout",           "dropout", [0.3, 0.5]),
    }

    order = ["lr", "size", "loss", "dropout"] if args.stage == "all" else [args.stage]
    for key in order:
        name, param, values = stages[key]
        state = stage_sweep(state, name, param, values, args.epochs)

    print("\n" + "=" * 72)
    print("  HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 72)
    for k in ["lr", "size", "loss", "dropout"]:
        print(f"    {k:<10}: {state[k]}")
    print("\n  Now run:  python -m src.hpo --final --final_epochs 15")
    print("=" * 72)


if __name__ == "__main__":
    main()
