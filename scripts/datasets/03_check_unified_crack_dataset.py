#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a materialized unified crack dataset.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "unified_crack")
    parser.add_argument("--sample-limit", type=int, default=2000)
    args = parser.parse_args()

    manifest = args.dataset_root / "manifest.csv"
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    missing = []
    bad_values = []
    bad_size = []
    nonempty = 0
    pixels = 0
    crack_pixels = 0
    by_dataset = Counter()
    by_split = Counter()

    for i, row in enumerate(rows):
        image = args.dataset_root / row["image_path"]
        mask = args.dataset_root / row["mask_path"]
        by_dataset[row["dataset"]] += 1
        by_split[row["split"]] += 1
        if not image.exists() or not mask.exists():
            missing.append((str(image), str(mask)))
            continue
        if i < args.sample_limit:
            with Image.open(image) as im, Image.open(mask) as ma:
                if im.size != ma.size:
                    bad_size.append((str(image), im.size, ma.size))
                arr = np.array(ma.convert("L"))
                values = set(np.unique(arr).tolist())
                if not values <= {0, 1}:
                    bad_values.append((str(mask), sorted(values)[:20]))
                crack = int((arr > 0).sum())
                pixels += int(arr.size)
                crack_pixels += crack
                if crack:
                    nonempty += 1

    print(f"samples: {len(rows)}")
    print(f"by dataset: {dict(by_dataset)}")
    print(f"by split: {dict(by_split)}")
    print(f"checked masks: {min(len(rows), args.sample_limit)}")
    print(f"nonempty checked masks: {nonempty}")
    if pixels:
        print(f"checked crack pixel ratio: {crack_pixels / pixels:.6f}")
    print(f"missing files: {len(missing)}")
    print(f"bad mask values: {len(bad_values)}")
    print(f"bad image/mask sizes: {len(bad_size)}")
    for title, items in (("missing", missing), ("bad_values", bad_values), ("bad_size", bad_size)):
        if items:
            print(f"\nfirst {title}:")
            for item in items[:10]:
                print(item)


if __name__ == "__main__":
    main()

