from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(os.environ.get("DAMAGE_DATA_ROOT", Path(__file__).resolve().parents[2]))
DACL_ROOT = ROOT / "dacl10k_v2_devphase"
HRCDS_ROOT = ROOT / "HRCDS"
S2DS_ROOT = ROOT / "s2ds"

UNIFIED_CLASSES = [
    "crack",
    "alligator_crack",
    "wetspot",
    "efflorescence",
    "rust_corrosion",
    "rockpocket",
    "hollow_area",
    "cavity",
    "spalling",
    "graffiti",
    "weathering",
    "restformwork",
    "exposed_rebar",
    "bearing",
    "expansion_joint",
    "drainage",
    "protective_equipment",
    "joint_tape",
    "washout_concrete_corrosion",
    "vegetation",
]
CID = {name: idx for idx, name in enumerate(UNIFIED_CLASSES)}
C = len(UNIFIED_CLASSES)

DACL_LABEL_MAP = {
    "Crack": "crack",
    "ACrack": "alligator_crack",
    "Wetspot": "wetspot",
    "Efflorescence": "efflorescence",
    "Rust": "rust_corrosion",
    "Rockpocket": "rockpocket",
    "Hollowareas": "hollow_area",
    "Cavity": "cavity",
    "Spalling": "spalling",
    "Graffiti": "graffiti",
    "Weathering": "weathering",
    "Restformwork": "restformwork",
    "ExposedRebars": "exposed_rebar",
    "Bearing": "bearing",
    "EJoint": "expansion_joint",
    "Drainage": "drainage",
    "PEquipment": "protective_equipment",
    "JTape": "joint_tape",
    "WConccor": "washout_concrete_corrosion",
}

HRCDS_ID_MAP = {
    1: "crack",
    2: "spalling",
    3: "rust_corrosion",
    4: "exposed_rebar",
}

# S2DS labels are color-coded PNGs. These defaults match the local masks by
# comparing per-color image/pixel counts against the S2DS README split table.
# Override with --s2ds-color-map-json if your copy differs.
S2DS_COLOR_MAP = {
    (255, 255, 255): "crack",
    (255, 0, 0): "spalling",
    (255, 255, 0): "rust_corrosion",
    (0, 255, 255): "efflorescence",
    (0, 255, 0): "vegetation",
    (0, 0, 255): None,  # Control Point: ignored for the main model.
    (0, 0, 0): None,  # Background.
}

SPLIT_ALIASES = {
    "validation": "val",
}


def rel(path: Path, root: Path = ROOT) -> str:
    return str(path.absolute().relative_to(root.absolute()))


def split_name(name: str) -> str:
    return SPLIT_ALIASES.get(name, name)


def load_s2ds_color_map(path: Path | None) -> dict[tuple[int, int, int], str | None]:
    if path is None:
        return dict(S2DS_COLOR_MAP)
    raw = json.loads(path.read_text())
    out: dict[tuple[int, int, int], str | None] = {}
    for key, value in raw.items():
        parts = key.replace("(", "").replace(")", "").split(",")
        if len(parts) != 3:
            raise ValueError(f"bad RGB key in {path}: {key!r}")
        out[tuple(int(p.strip()) for p in parts)] = value
    return out


def write_classes(path: Path) -> None:
    payload = [
        {"id": idx, "name": name}
        for idx, name in enumerate(UNIFIED_CLASSES)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=array.astype(np.uint8, copy=False))


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"unknown image mode: {mode}")


def empty_target(height: int, width: int) -> np.ndarray:
    return np.zeros((C, height, width), dtype=np.uint8)


def valid_for(class_names: list[str], height: int, width: int) -> np.ndarray:
    valid = np.zeros((C, height, width), dtype=np.uint8)
    for name in class_names:
        valid[CID[name], :, :] = 1
    return valid


def dacl_target(annotation_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    data = json.loads(annotation_path.read_text())
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    target = empty_target(height, width)
    valid = valid_for(list(DACL_LABEL_MAP.values()), height, width)

    by_class: dict[int, Image.Image] = {}
    for shape in data.get("shapes", []):
        unified = DACL_LABEL_MAP.get(shape.get("label"))
        if unified is None:
            continue
        points = [tuple(point) for point in shape.get("points", [])]
        if len(points) < 3:
            continue
        cid = CID[unified]
        mask = by_class.get(cid)
        if mask is None:
            mask = Image.new("L", (width, height), 0)
            by_class[cid] = mask
        ImageDraw.Draw(mask).polygon(points, fill=1)

    for cid, mask in by_class.items():
        target[cid] = np.array(mask, dtype=np.uint8)
    return target, valid, width, height


def hrcds_target(mask_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask = np.array(Image.open(mask_path).convert("L"))
    height, width = mask.shape
    target = empty_target(height, width)
    valid = valid_for(list(HRCDS_ID_MAP.values()), height, width)
    for source_id, unified in HRCDS_ID_MAP.items():
        target[CID[unified]] = (mask == source_id).astype(np.uint8)
    return target, valid, width, height


def s2ds_target(
    mask_path: Path,
    color_map: dict[tuple[int, int, int], str | None],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    rgb = np.array(Image.open(mask_path).convert("RGB"))
    height, width = rgb.shape[:2]
    target = empty_target(height, width)
    valid_classes = [
        "crack",
        "spalling",
        "rust_corrosion",
        "efflorescence",
        "vegetation",
    ]
    valid = valid_for(valid_classes, height, width)

    seen_colors = {tuple(map(int, c)) for c in np.unique(rgb.reshape(-1, 3), axis=0)}
    unknown = seen_colors - set(color_map)
    if unknown:
        raise ValueError(f"{mask_path} has unknown S2DS colors: {sorted(unknown)}")

    for color, unified in color_map.items():
        if unified is None:
            continue
        target[CID[unified]] |= np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
    return target, valid, width, height


def add_record(
    rows_by_split: dict[str, list[dict[str, str]]],
    output_root: Path,
    dataset: str,
    split: str,
    image_path: Path,
    label_path: Path,
    target: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    image_mode: str,
) -> None:
    item_id = image_path.stem
    if dataset == "s2ds" and item_id.endswith("_lab"):
        item_id = item_id[:-4]
    image_dst = output_root / "images" / dataset / split / image_path.name
    target_dst = output_root / "masks_multilabel" / dataset / split / f"{item_id}.npz"
    valid_dst = output_root / "valid_masks" / dataset / split / f"{item_id}.npz"

    link_or_copy(image_path, image_dst, image_mode)
    save_npz(target_dst, target)
    save_npz(valid_dst, valid)

    rows_by_split.setdefault(split, []).append(
        {
            "dataset": dataset,
            "split": split,
            "id": item_id,
            "image_path": rel(image_dst, output_root),
            "target_path": rel(target_dst, output_root),
            "valid_mask_path": rel(valid_dst, output_root),
            "source_image": rel(image_path),
            "source_label": rel(label_path),
            "width": str(width),
            "height": str(height),
        }
    )


def convert_dacl(
    output_root: Path,
    rows_by_split: dict[str, list[dict[str, str]]],
    image_mode: str,
    limit: int | None,
) -> int:
    count = 0
    for source_split in ("train", "validation"):
        split = split_name(source_split)
        ann_dir = DACL_ROOT / "annotations" / source_split
        image_dir = DACL_ROOT / "images" / source_split
        for ann_path in sorted(ann_dir.glob("*.json")):
            image_name = json.loads(ann_path.read_text()).get("imageName")
            image_path = image_dir / image_name
            if not image_path.exists():
                raise FileNotFoundError(f"DACL image missing for {ann_path}: {image_path}")
            target, valid, width, height = dacl_target(ann_path)
            add_record(
                rows_by_split,
                output_root,
                "dacl",
                split,
                image_path,
                ann_path,
                target,
                valid,
                width,
                height,
                image_mode,
            )
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def convert_hrcds(
    output_root: Path,
    rows_by_split: dict[str, list[dict[str, str]]],
    image_mode: str,
    limit: int | None,
) -> int:
    count = 0
    for split in ("train", "val", "test"):
        mask_dir = HRCDS_ROOT / f"{split}_mask"
        image_dir = HRCDS_ROOT / f"{split}_image"
        ann_dir = HRCDS_ROOT / f"{split}_annotations"
        for mask_path in sorted(mask_dir.glob("*_mask.png")):
            item_id = mask_path.stem.removesuffix("_mask")
            image_path = image_dir / f"{item_id}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{item_id}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"HRCDS image missing for {mask_path}")
            ann_path = ann_dir / f"{item_id}.json"
            label_path = ann_path if ann_path.exists() else mask_path
            target, valid, width, height = hrcds_target(mask_path)
            add_record(
                rows_by_split,
                output_root,
                "hrcds",
                split,
                image_path,
                label_path,
                target,
                valid,
                width,
                height,
                image_mode,
            )
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def convert_s2ds(
    output_root: Path,
    rows_by_split: dict[str, list[dict[str, str]]],
    image_mode: str,
    limit: int | None,
    color_map: dict[tuple[int, int, int], str | None],
) -> int:
    count = 0
    for split in ("train", "val", "test"):
        split_dir = S2DS_ROOT / split
        for mask_path in sorted(split_dir.glob("*_lab.png")):
            item_id = mask_path.stem.removesuffix("_lab")
            image_path = split_dir / f"{item_id}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"S2DS image missing for {mask_path}: {image_path}")
            target, valid, width, height = s2ds_target(mask_path, color_map)
            add_record(
                rows_by_split,
                output_root,
                "s2ds",
                split,
                image_path,
                mask_path,
                target,
                valid,
                width,
                height,
                image_mode,
            )
            count += 1
            if limit is not None and count >= limit:
                return count
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DACL10K, S2DS, and HRCDS into unified multi-label segmentation arrays."
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "unified_damage_data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("dacl", "s2ds", "hrcds"),
        default=("dacl", "s2ds", "hrcds"),
    )
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize images under output-root/images.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=None,
        help="Debug option: convert at most this many samples from each selected dataset.",
    )
    parser.add_argument(
        "--s2ds-color-map-json",
        type=Path,
        default=None,
        help='Optional RGB map, e.g. {"255,0,0": "crack", "255,255,255": null}.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_classes(output_root / "classes.json")

    rows_by_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    counts: dict[str, int] = {}
    if "dacl" in args.datasets:
        counts["dacl"] = convert_dacl(output_root, rows_by_split, args.image_mode, args.limit_per_dataset)
    if "s2ds" in args.datasets:
        color_map = load_s2ds_color_map(args.s2ds_color_map_json)
        counts["s2ds"] = convert_s2ds(
            output_root,
            rows_by_split,
            args.image_mode,
            args.limit_per_dataset,
            color_map,
        )
    if "hrcds" in args.datasets:
        counts["hrcds"] = convert_hrcds(output_root, rows_by_split, args.image_mode, args.limit_per_dataset)

    for split, rows in rows_by_split.items():
        write_csv(output_root / f"{split}.csv", rows)
    write_csv(output_root / "all.csv", [row for rows in rows_by_split.values() for row in rows])

    summary = {
        "classes": len(UNIFIED_CLASSES),
        "counts": counts,
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
