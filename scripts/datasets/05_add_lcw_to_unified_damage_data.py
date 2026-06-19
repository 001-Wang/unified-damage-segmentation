from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(os.environ.get("DAMAGE_DATA_ROOT", Path(__file__).resolve().parents[2]))
LCW_ROOT = ROOT / "LCW Concrete Crack Detection" / "LCW Concrete Crack Detection"
DEFAULT_OUTPUT_ROOT = ROOT / "unified_damage_data"
NUM_CLASSES = 20
CRACK_CLASS_ID = 0


def rel(path: Path, root: Path) -> str:
    return str(path.absolute().relative_to(root.absolute()))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "dataset",
        "split",
        "id",
        "image_path",
        "target_path",
        "valid_mask_path",
        "source_image",
        "source_label",
        "width",
        "height",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def save_npz(path: Path, array: np.ndarray, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, mask=array.astype(np.uint8, copy=False))
    generated = Path(str(tmp) + ".npz")
    generated.replace(path)


def symlink_image(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.symlink_to(src.resolve())


def lcw_target(mask_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask = np.array(Image.open(mask_path).convert("L"))
    height, width = mask.shape
    target = np.zeros((NUM_CLASSES, height, width), dtype=np.uint8)
    valid = np.zeros((NUM_CLASSES, height, width), dtype=np.uint8)
    target[CRACK_CLASS_ID] = (mask > 127).astype(np.uint8)
    valid[CRACK_CLASS_ID] = 1
    return target, valid, width, height


def build_lcw_rows(output_root: Path, variant: str, overwrite: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    split_map = {"Train": "train", "Test": "test"}
    for source_split, split in split_map.items():
        image_dir = LCW_ROOT / variant / source_split / "images"
        mask_dir = LCW_ROOT / variant / source_split / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(f"LCW split is missing images or masks: {image_dir}, {mask_dir}")

        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"LCW mask missing for {image_path}: {mask_path}")

            item_id = f"lcw_{variant}_{split}_{image_path.stem}"
            image_dst = output_root / "images" / "lcw" / split / image_path.name
            target_dst = output_root / "masks_multilabel" / "lcw" / split / f"{item_id}.npz"
            valid_dst = output_root / "valid_masks" / "lcw" / split / f"{item_id}.npz"

            target, valid, width, height = lcw_target(mask_path)
            symlink_image(image_path, image_dst, overwrite)
            save_npz(target_dst, target, overwrite)
            save_npz(valid_dst, valid, overwrite)

            rows.append(
                {
                    "dataset": "lcw",
                    "split": split,
                    "id": item_id,
                    "image_path": rel(image_dst, output_root),
                    "target_path": rel(target_dst, output_root),
                    "valid_mask_path": rel(valid_dst, output_root),
                    "source_image": rel(image_path, ROOT),
                    "source_label": rel(mask_path, ROOT),
                    "width": str(width),
                    "height": str(height),
                }
            )
    return rows


def without_lcw(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("dataset") != "lcw"]


def write_summary(output_root: Path, all_rows: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    splits = {"train": 0, "val": 0, "test": 0}
    for row in all_rows:
        dataset = row["dataset"]
        split = row["split"]
        counts[dataset] = counts.get(dataset, 0) + 1
        splits[split] = splits.get(split, 0) + 1
    summary = {"classes": NUM_CLASSES, "counts": counts, "splits": splits}
    tmp = output_root / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(output_root / "summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add LCW crack-only masks to unified_damage_data.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--variant",
        choices=("original", "512x512", "512x512_dilated_no_blanks"),
        default="original",
    )
    parser.add_argument("--overwrite-lcw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    lcw_rows = build_lcw_rows(output_root, args.variant, args.overwrite_lcw)
    by_split: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "val", "test"):
        existing = without_lcw(read_csv(output_root / f"{split}.csv"))
        additions = [row for row in lcw_rows if row["split"] == split]
        by_split[split] = existing + additions
        write_csv_atomic(output_root / f"{split}.csv", by_split[split])

    all_rows = [row for split in ("train", "val", "test") for row in by_split[split]]
    write_csv_atomic(output_root / "all.csv", all_rows)
    write_summary(output_root, all_rows)

    print(json.dumps({
        "added_lcw": len(lcw_rows),
        "splits": {split: len([row for row in lcw_rows if row["split"] == split]) for split in ("train", "val", "test")},
        "output_root": str(output_root),
    }, indent=2))


if __name__ == "__main__":
    main()
