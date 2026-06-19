#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from dataset_common import ROOT, binary_mask_from_image, csb_json_mask, dacl_crack_mask, read_manifest, save_binary_png


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create normalized crack dataset from a manifest.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "crack_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "unified_crack")
    parser.add_argument("--image-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N rows for smoke tests.")
    parser.add_argument("--limit-per-dataset", type=int, default=0, help="Optional first-N rows from each dataset.")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    if args.limit_per_dataset:
        kept = []
        counts: dict[str, int] = {}
        for row in rows:
            dataset = row["dataset"]
            if counts.get(dataset, 0) >= args.limit_per_dataset:
                continue
            kept.append(row)
            counts[dataset] = counts.get(dataset, 0) + 1
        rows = kept

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_root / "manifest.csv"
    output_rows = []

    for idx, row in enumerate(rows):
        src_image = ROOT / row["image_path"]
        src_label = ROOT / row["label_path"]
        split = row["split"]
        stem = f"{idx:08d}_{row['dataset']}_{src_image.stem}".replace("/", "_")
        out_image = args.output_root / "images" / split / f"{stem}{src_image.suffix.lower()}"
        out_mask = args.output_root / "masks" / split / f"{stem}.png"

        link_or_copy(src_image, out_image, args.image_mode)

        if row["label_kind"] == "mask_image":
            mask = binary_mask_from_image(src_label)
        elif row["label_kind"] == "dacl_json_polygon":
            mask = dacl_crack_mask(src_label)
        elif row["label_kind"] == "csb_json_pixels":
            mask = csb_json_mask(src_label, src_image)
        else:
            raise ValueError(f"unknown label_kind: {row['label_kind']}")
        save_binary_png(mask, out_mask)

        output_rows.append(
            {
                "dataset": row["dataset"],
                "split": split,
                "image_path": str(out_image.relative_to(args.output_root)),
                "mask_path": str(out_mask.relative_to(args.output_root)),
                "source_image": row["image_path"],
                "source_label": row["label_path"],
                "width": row["width"],
                "height": row["height"],
            }
        )

    with output_manifest.open("w", newline="") as f:
        import csv

        fields = ["dataset", "split", "image_path", "mask_path", "source_image", "source_label", "width", "height"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"materialized {len(output_rows)} samples under {args.output_root}")
    print(f"wrote {output_manifest}")


if __name__ == "__main__":
    main()
