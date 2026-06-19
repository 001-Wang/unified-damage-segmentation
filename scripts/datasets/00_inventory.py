#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path

from dataset_common import CONG_ROOT, CSB_ROOT, DACL_ROOT, LCW_ROOT, ROOT, SEGCODE_ROOT


def count_files(root: Path) -> Counter:
    counts: Counter = Counter()
    if not root.exists():
        return counts
    for path in root.rglob("*"):
        if path.is_file():
            counts[path.suffix.lower() or "<no_ext>"] += 1
    return counts


def print_counts(name: str, root: Path) -> None:
    print(f"\n[{name}]")
    print(f"root: {root.relative_to(ROOT) if root.exists() else root}")
    if not root.exists():
        print("status: MISSING")
        return
    counts = count_files(root)
    print("file types:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def count_pair_dirs(name: str, image_dir: Path, mask_dir: Path, image_exts=(".jpg", ".jpeg", ".png")) -> None:
    images = [p for p in image_dir.glob("*") if p.suffix.lower() in image_exts]
    masks = [p for p in mask_dir.glob("*") if p.suffix.lower() in image_exts]
    image_stems = {p.stem for p in images}
    mask_stems = {p.stem for p in masks}
    print(f"{name}: images={len(images)} masks={len(masks)} paired_stems={len(image_stems & mask_stems)}")


def main() -> None:
    print(f"dataset root: {ROOT}")
    print_counts("DACL10K", DACL_ROOT)
    print_counts("LCW", LCW_ROOT)
    print_counts("Conglomerate", CONG_ROOT)
    print_counts("CSB", CSB_ROOT)
    print_counts("SegCODEBRIM", SEGCODE_ROOT)

    print("\n[pair checks]")
    if DACL_ROOT.exists():
        for split in ("train", "validation"):
            imgs = list((DACL_ROOT / "images" / split).glob("*.jpg"))
            anns = list((DACL_ROOT / "annotations" / split).glob("*.json"))
            print(f"DACL10K/{split}: images={len(imgs)} json={len(anns)}")
        print(f"DACL10K/testdev: images={len(list((DACL_ROOT / 'images' / 'testdev').glob('*.jpg')))} labels=0")

    if LCW_ROOT.exists():
        for variant in ("512x512", "512x512_dilated_no_blanks", "original"):
            for split in ("Train", "Test"):
                base = LCW_ROOT / variant / split
                if (base / "images").exists() and (base / "masks").exists():
                    count_pair_dirs(f"LCW/{variant}/{split}", base / "images", base / "masks")

    if CONG_ROOT.exists():
        for split in ("Train", "Test"):
            count_pair_dirs(f"Conglomerate/{split}", CONG_ROOT / split / "images", CONG_ROOT / split / "masks")

    if SEGCODE_ROOT.exists():
        count_pair_dirs("SegCODEBRIM", SEGCODE_ROOT / "images", SEGCODE_ROOT / "gts")

    if CSB_ROOT.exists():
        entire = CSB_ROOT / "entire images"
        for split in ("crack_train", "crack_test", "nocrack_train", "nocrack_test"):
            files = list((entire / split).glob("*")) if (entire / split).exists() else []
            print(
                f"CSB/entire/{split}: images={sum(p.suffix.lower() in ('.jpg', '.jpeg') for p in files)} "
                f"json={sum(p.suffix.lower() == '.json' for p in files)}"
            )
        patch_root = CSB_ROOT / "patch datasets"
        patch_images = 0
        patch_masks = 0
        for path in patch_root.rglob("*.jpg"):
            if path.name.lower().endswith("mask.jpg"):
                patch_masks += 1
            else:
                patch_images += 1
        print(f"CSB/patches: images={patch_images} masks={patch_masks}")


if __name__ == "__main__":
    main()



