from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(os.environ.get("DAMAGE_DATA_ROOT", Path(__file__).resolve().parents[2]))

DACL_ROOT = ROOT / "dacl10k_v2_devphase"
LCW_ROOT = ROOT / "LCW Concrete Crack Detection" / "LCW Concrete Crack Detection"
CONG_ROOT = ROOT / "Conglomerate Concrete Crack Detection" / "Conglomerate Concrete Crack Detection"
CSB_ROOT = ROOT / "Cracks in Steel Bridges (CSB) dataset" / "CSB_dataset"
SEGCODE_ROOT = ROOT / "SegCODEBRIM"

CRACK_LABELS = {"Crack", "ACrack"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: Iterable[dict[str, str]]) -> None:
    rows = list(rows)
    fieldnames = [
        "dataset",
        "split",
        "image_path",
        "label_path",
        "label_kind",
        "mask_strategy",
        "width",
        "height",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def binary_mask_from_image(path: Path, threshold: int = 127) -> Image.Image:
    arr = np.array(Image.open(path).convert("L"))
    return Image.fromarray((arr > threshold).astype(np.uint8), mode="L")


def dacl_crack_mask(annotation_path: Path) -> Image.Image:
    data = json.loads(annotation_path.read_text())
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in data.get("shapes", []):
        if shape.get("label") not in CRACK_LABELS:
            continue
        points = [tuple(p) for p in shape.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=1)
    return mask


def csb_json_mask(annotation_path: Path, image_path: Path) -> Image.Image:
    with Image.open(image_path) as im:
        width, height = im.size
    data = json.loads(annotation_path.read_text())
    arr = np.zeros((height, width), dtype=np.uint8)
    crack_pixels = data.get("crack pixels", [])
    if crack_pixels:
        pts = np.asarray(crack_pixels, dtype=np.int64)
        y = pts[:, 0]
        x = pts[:, 1]
        good = (0 <= x) & (x < width) & (0 <= y) & (y < height)
        arr[y[good], x[good]] = 1
    return Image.fromarray(arr, mode="L")


def save_binary_png(mask: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(mask.convert("L"))
    Image.fromarray((arr > 0).astype(np.uint8), mode="L").save(path)


def iter_files(root: Path, suffixes: set[str] = IMAGE_SUFFIXES) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path

