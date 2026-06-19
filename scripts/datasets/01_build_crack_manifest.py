#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dataset_common import (
    CONG_ROOT,
    CRACK_LABELS,
    CSB_ROOT,
    DACL_ROOT,
    LCW_ROOT,
    ROOT,
    SEGCODE_ROOT,
    image_size,
    rel,
    write_manifest,
)


def add_row(rows, dataset, split, image_path, label_path, label_kind, mask_strategy):
    width, height = image_size(image_path)
    rows.append(
        {
            "dataset": dataset,
            "split": split,
            "image_path": rel(image_path),
            "label_path": rel(label_path),
            "label_kind": label_kind,
            "mask_strategy": mask_strategy,
            "width": width,
            "height": height,
        }
    )


def dacl_rows(rows) -> None:
    for split in ("train", "validation"):
        ann_dir = DACL_ROOT / "annotations" / split
        img_dir = DACL_ROOT / "images" / split
        for ann in sorted(ann_dir.glob("*.json")):
            data = json.loads(ann.read_text())
            if not any(s.get("label") in CRACK_LABELS for s in data.get("shapes", [])):
                continue
            image = img_dir / data["imageName"]
            if image.exists():
                out_split = "train" if split == "train" else "val"
                add_row(rows, "dacl10k_crack", out_split, image, ann, "dacl_json_polygon", "dacl_crack_acrack")


def lcw_rows(rows, variant: str) -> None:
    for src_split, out_split in (("Train", "train"), ("Test", "test")):
        base = LCW_ROOT / variant / src_split
        for image in sorted((base / "images").glob("*.jpeg")):
            mask = base / "masks" / f"{image.stem}.png"
            if mask.exists():
                add_row(rows, f"lcw_{variant}", out_split, image, mask, "mask_image", "threshold_127")


def conglomerate_rows(rows) -> None:
    for src_split, out_split in (("Train", "train"), ("Test", "test")):
        base = CONG_ROOT / src_split
        for image in sorted((base / "images").glob("*")):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            mask = base / "masks" / image.name
            if mask.exists():
                add_row(rows, "conglomerate", out_split, image, mask, "mask_image", "threshold_127")


def segcode_rows(rows, val_fraction: float, seed: int) -> None:
    images = sorted(
        p
        for p in (SEGCODE_ROOT / "images").glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    val_n = int(round(len(shuffled) * val_fraction))
    val_set = {p.name for p in shuffled[:val_n]}
    for image in images:
        mask = SEGCODE_ROOT / "gts" / f"{image.stem}.png"
        if mask.exists():
            split = "val" if image.name in val_set else "train"
            add_row(rows, "segcodebrim", split, image, mask, "mask_image", "threshold_127")


def csb_entire_rows(rows) -> None:
    mapping = {
        "crack_train": "train",
        "nocrack_train": "train",
        "crack_test": "test",
        "nocrack_test": "test",
    }
    entire = CSB_ROOT / "entire images"
    for folder, out_split in mapping.items():
        src = entire / folder
        for ann in sorted(src.glob("*.json")):
            image = ann.with_suffix(".JPG")
            if not image.exists():
                image = ann.with_suffix(".jpg")
            if image.exists():
                add_row(rows, "csb_entire", out_split, image, ann, "csb_json_pixels", "csb_crack_pixels")


def csb_patch_rows(rows, variant: str) -> None:
    roots: list[tuple[Path, str]] = []
    patch_root = CSB_ROOT / "patch datasets"
    if variant == "70_30_corrected":
        roots = [
            (patch_root / "512_512" / "70_30" / "train_corrected", "train"),
            (patch_root / "512_512" / "70_30" / "test_corrected", "test"),
        ]
    elif variant == "full_512":
        roots = [
            (patch_root / "512_512" / "full" / "train", "train"),
            (patch_root / "512_512" / "full" / "test", "test"),
        ]
    elif variant in {"128_128", "256_256", "384_384"}:
        roots = [(patch_root / variant / "train", "train"), (patch_root / variant / "test", "test")]
    else:
        raise ValueError(f"unsupported CSB patch variant: {variant}")

    for root, split in roots:
        if not root.exists():
            continue
        for image in sorted(root.rglob("*.jpg")):
            if image.name.lower().endswith("mask.jpg"):
                continue
            mask = image.with_name(f"{image.stem}mask.jpg")
            if mask.exists():
                add_row(rows, f"csb_patch_{variant}", split, image, mask, "mask_image", "threshold_127")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a crack-segmentation manifest without copying data.")
    parser.add_argument("--output", type=Path, default=ROOT / "manifests" / "crack_manifest.csv")
    parser.add_argument("--lcw-variant", default="512x512", choices=["512x512", "original", "512x512_dilated_no_blanks"])
    parser.add_argument("--csb", default="patch_70_30_corrected", choices=["none", "entire", "patch_70_30_corrected", "patch_full_512", "patch_128_128", "patch_256_256", "patch_384_384"])
    parser.add_argument("--include-dacl", action="store_true", help="Include DACL10K Crack/ACrack polygons as binary crack masks.")
    parser.add_argument("--segcode-val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = []
    lcw_rows(rows, args.lcw_variant)
    conglomerate_rows(rows)
    segcode_rows(rows, args.segcode_val_fraction, args.seed)
    if args.include_dacl:
        dacl_rows(rows)
    if args.csb == "entire":
        csb_entire_rows(rows)
    elif args.csb.startswith("patch_"):
        csb_patch_rows(rows, args.csb.removeprefix("patch_"))

    write_manifest(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    by_dataset: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for row in rows:
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    print("by dataset:", by_dataset)
    print("by split:", by_split)


if __name__ == "__main__":
    main()
