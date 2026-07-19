"""
Pre-resize the dataset once to eliminate the data loading bottleneck.

Problem this solves. APTOS images are up to 3200x2100. Every epoch, every worker was
re-reading those multi-megabyte PNGs from disk and running circular_crop over roughly
6.7 million pixels, just to produce a 300x300 tensor. The GPU sat idle waiting. Epoch
time was 636 seconds, almost all of it data loading rather than computation.

Fix. Run the expensive, deterministic, per-image work exactly once and write the result
to a cache: circular crop, then downscale to CACHE_SIZE. Every later epoch reads a small
file. Nothing downstream changes, because the pipeline was going to downscale to 300
anyway, so no information the model ever saw is lost.

CACHE_SIZE is deliberately larger than the network input (512 vs 300) so that later
resolution experiments (for example --size 380) still have headroom.

Usage:
    python -m src.cache_images
    python -m src.cache_images --cache_size 512 --workers 4

Author: Abdul Aleem Mohammed
CS 898BA, Wichita State University
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd

from src.preprocessing.fundus import circular_crop

CACHE_DIR = "data/cache"


def process_one(args):
    src, dst, size = args
    if os.path.exists(dst):
        return "skip"
    img = cv2.imread(src)
    if img is None:
        return "fail"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # The two expensive, deterministic operations, done once instead of every epoch.
    img = circular_crop(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    cv2.imwrite(dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--img_dir", default="data/train_images")
    ap.add_argument("--cache_dir", default=CACHE_DIR)
    ap.add_argument("--cache_size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    jobs = [
        (
            os.path.join(args.img_dir, f"{r.id_code}.png"),
            os.path.join(args.cache_dir, f"{r.id_code}.png"),
            args.cache_size,
        )
        for r in df.itertuples()
    ]

    print(f"Caching {len(jobs)} images at {args.cache_size}x{args.cache_size} "
          f"using {args.workers} workers...")
    t0 = time.time()

    counts = {"ok": 0, "skip": 0, "fail": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(process_one, jobs, chunksize=16), 1):
            counts[res] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}  ({time.time() - t0:.0f}s)")

    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s   written={counts['ok']}  "
          f"already cached={counts['skip']}  failed={counts['fail']}")
    print(f"Cache at: {args.cache_dir}")
    print("\nTrain against the cache with:  --img_dir data/cache")


if __name__ == "__main__":
    main()
