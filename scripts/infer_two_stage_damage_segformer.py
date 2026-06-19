#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from train_two_stage_damage_segformer import (
    expand_box,
    predicted_canvas_box_to_original,
    preprocess_roi_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage ROI-guided multi-label damage segmentation.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--stage1-model-dir", type=Path, required=True, help="Trained ROI model directory.")
    parser.add_argument("--stage2-model-dir", type=Path, required=True, help="Trained fine segmentation model directory.")
    parser.add_argument(
        "--classes-json",
        type=Path,
        default=Path("/home/grads/z/zuoxu/data/damage_data/unified_damage_data/classes.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"])
    parser.add_argument("--roi-threshold", type=float, default=0.5)
    parser.add_argument("--fine-threshold", type=float, default=0.5)
    parser.add_argument("--roi-crop-padding", type=float, default=0.15)
    parser.add_argument("--gate-with-roi", action="store_true", help="Multiply final class masks by the Stage 1 ROI mask.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    with path.open() as f:
        classes = json.load(f)
    return [item["name"] for item in sorted(classes, key=lambda item: item["id"])]


def predict_stage1_roi(
    image: Image.Image,
    model,
    image_processor,
    image_size: int,
    resize_mode: str,
    threshold: float,
    padding: float,
    device: str,
) -> tuple[torch.Tensor, list[int]]:
    encoded, meta = preprocess_roi_image(image, image_processor, image_size, resize_mode)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    logits = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=False)
    roi_canvas = (torch.sigmoid(logits)[0, 0].cpu() >= threshold).float()
    box = predicted_canvas_box_to_original(roi_canvas, image, resize_mode, meta, padding)
    return roi_canvas, box


def preprocess_crop(
    crop: Image.Image, image_processor, image_size: int, resize_mode: str
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    return preprocess_roi_image(crop, image_processor, image_size, resize_mode)


def unletterbox_probs(
    probs: torch.Tensor, crop: Image.Image, image_size: int, resize_mode: str, meta: dict[str, float | int]
) -> torch.Tensor:
    crop_width, crop_height = crop.size
    if resize_mode == "stretch":
        return F.interpolate(probs.unsqueeze(0), size=(crop_height, crop_width), mode="bilinear", align_corners=False).squeeze(0)

    left = int(meta["left"])
    top = int(meta["top"])
    new_width = int(meta["new_width"])
    new_height = int(meta["new_height"])
    content = probs[:, top : top + new_height, left : left + new_width]
    return F.interpolate(content.unsqueeze(0), size=(crop_height, crop_width), mode="bilinear", align_corners=False).squeeze(0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(args.classes_json)
    image = Image.open(args.image).convert("RGB")

    stage1_processor = AutoImageProcessor.from_pretrained(args.stage1_model_dir)
    stage1_model = AutoModelForSemanticSegmentation.from_pretrained(args.stage1_model_dir).to(args.device).eval()
    stage2_processor = AutoImageProcessor.from_pretrained(args.stage2_model_dir)
    stage2_model = AutoModelForSemanticSegmentation.from_pretrained(args.stage2_model_dir).to(args.device).eval()

    roi_canvas, box = predict_stage1_roi(
        image,
        stage1_model,
        stage1_processor,
        args.image_size,
        args.resize_mode,
        args.roi_threshold,
        args.roi_crop_padding,
        args.device,
    )

    left, top, right, bottom = expand_box(box, image.width, image.height, padding=0.0)
    crop = image.crop((left, top, right, bottom))
    encoded, crop_meta = preprocess_crop(crop, stage2_processor, args.image_size, args.resize_mode)
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.no_grad():
        fine_logits = stage2_model(**encoded).logits
    fine_logits = F.interpolate(fine_logits, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
    crop_probs = unletterbox_probs(torch.sigmoid(fine_logits)[0].cpu(), crop, args.image_size, args.resize_mode, crop_meta)

    full_probs = torch.zeros((len(class_names), image.height, image.width), dtype=crop_probs.dtype)
    full_probs[:, top:bottom, left:right] = crop_probs

    # This gates by the ROI crop box, not the raw low-resolution ROI shape. It is conservative and keeps output aligned.
    if args.gate_with_roi:
        gate = torch.zeros((image.height, image.width), dtype=torch.float32)
        gate[top:bottom, left:right] = 1.0
        full_probs = full_probs * gate.unsqueeze(0)

    binary = (full_probs >= args.fine_threshold).numpy().astype(np.uint8) * 255
    for class_id, class_name in enumerate(class_names):
        Image.fromarray(binary[class_id], mode="L").save(args.output_dir / f"{class_id:02d}_{class_name}.png")

    roi_box_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    roi_box_mask[top:bottom, left:right] = 255
    Image.fromarray(roi_box_mask, mode="L").save(args.output_dir / "roi_box.png")
    print(f"ROI box: {[left, top, right, bottom]}")
    print(f"Saved {len(class_names)} class masks to {args.output_dir}")


if __name__ == "__main__":
    main()
