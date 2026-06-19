#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save SegFormer crack segmentation prediction overlays.")
    parser.add_argument("--model-dir", type=Path, default=Path("/tmp/crack_segformer_smoke"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/grads/z/zuoxu/data/damage_data/unified_crack"))
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/crack_segformer_smoke_predictions"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--prefer-nonempty", action="store_true", help="Visualize samples whose ground-truth mask has cracks.")
    parser.add_argument(
        "--crack-threshold",
        type=float,
        default=0.65,
        help="Minimum crack probability needed to mark a pixel as crack. Higher values reduce false positives.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mask_has_crack(mask_path: Path) -> bool:
    mask = np.array(Image.open(mask_path).convert("L"))
    return bool((mask == 1).any())


def load_rows(data_root: Path, split: str, num_samples: int, prefer_nonempty: bool) -> list[dict[str, str]]:
    rows = []
    with (data_root / "manifest.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                rows.append(row)

    if prefer_nonempty:
        nonempty = [row for row in rows if mask_has_crack(data_root / row["mask_path"])]
        if nonempty:
            rows = nonempty

    return rows[:num_samples]


def colorize(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[mask.astype(bool), :3] = color
    out[mask.astype(bool), 3] = 140
    return Image.fromarray(out, mode="RGBA")


def save_panel(image: Image.Image, gt: np.ndarray, pred: np.ndarray, output_path: Path) -> None:
    image = image.convert("RGB")
    gt_overlay = image.convert("RGBA")
    gt_overlay.alpha_composite(colorize(gt == 1, (0, 255, 0)))

    pred_overlay = image.convert("RGBA")
    pred_overlay.alpha_composite(colorize(pred == 1, (255, 0, 0)))

    panel = Image.new("RGB", (image.width * 3, image.height), "white")
    panel.paste(image, (0, 0))
    panel.paste(gt_overlay.convert("RGB"), (image.width, 0))
    panel.paste(pred_overlay.convert("RGB"), (image.width * 2, 0))
    panel.save(output_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_dir).to(args.device)
    model.eval()

    rows = load_rows(args.data_root, args.split, args.num_samples, args.prefer_nonempty)
    if not rows:
        raise ValueError(f"No rows found for split={args.split}")

    for idx, row in enumerate(rows):
        image = Image.open(args.data_root / row["image_path"]).convert("RGB")
        gt = np.array(Image.open(args.data_root / row["mask_path"]).convert("L"))

        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = F.interpolate(outputs.logits, size=gt.shape, mode="bilinear", align_corners=False)
        crack_prob = logits.softmax(dim=1)[0, 1]
        pred = (crack_prob >= args.crack_threshold).detach().cpu().numpy().astype(np.uint8)

        output_path = args.output_dir / f"{idx:02d}_{Path(row['image_path']).stem}.jpg"
        save_panel(image, gt, pred, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
