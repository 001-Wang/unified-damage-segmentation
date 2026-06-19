#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

from train_two_stage_damage_segformer import (
    load_classes,
    load_npz_mask,
    make_roi_target_and_valid,
    mask_to_box,
    parse_class_ids,
    preprocess_roi_image,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage 1 ROI binary predictions against GT ROI masks.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trained Stage 1 ROI model directory.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/grads/z/zuoxu/data/damage_data/unified_damage_data"),
        help="Directory containing split CSVs, classes.json, images/, masks_multilabel/, valid_masks/.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to visualize, for example: dacl hrcds s2ds lcw.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--prefer-positive", action="store_true", help="Prefer samples with non-empty GT ROI masks.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--resize-mode", default="letterbox", choices=["letterbox", "stretch"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--roi-positive-class-ids",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,18",
        help="Comma-separated class ids that define ROI positives.",
    )
    parser.add_argument(
        "--roi-negative-valid-policy",
        default="all",
        choices=["all", "any"],
        help="Used only for optional per-sample valid-mask metrics in the summary CSV.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    normalized = device.lower()
    if normalized == "gpu":
        normalized = "cuda"
    elif normalized.startswith("gpu:"):
        normalized = f"cuda:{normalized.split(':', 1)[1]}"

    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")

    try:
        torch.device(normalized)
    except RuntimeError as error:
        raise ValueError(f"Invalid torch device '{device}'. Use values like cpu, cuda, cuda:0, or mps.") from error

    return normalized


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_rows(args.data_root, args.split, max_samples=0)
    if args.datasets is not None:
        datasets = set(args.datasets)
        rows = [row for row in rows if row["dataset"] in datasets]
    rows = rows[args.start_index :]
    if args.prefer_positive:
        positive = []
        non_positive = []
        id2label, _ = load_classes(args.data_root)
        roi_class_ids = parse_class_ids(args.roi_positive_class_ids, len(id2label))
        for row in rows:
            target = load_npz_mask(row["target_path"])
            valid_mask = load_npz_mask(row["valid_mask_path"])
            gt_roi, _ = make_roi_target_and_valid(
                target, valid_mask, roi_class_ids, args.roi_negative_valid_policy
            )
            if gt_roi.sum().item() > 0:
                positive.append(row)
            else:
                non_positive.append(row)
        rows = positive + non_positive
    return rows[: args.num_samples]


def probs_to_original(
    probs: torch.Tensor,
    image: Image.Image,
    image_size: int,
    resize_mode: str,
    meta: dict[str, float | int],
) -> torch.Tensor:
    width, height = image.size
    if resize_mode == "stretch":
        return F.interpolate(probs.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)

    left = int(meta["left"])
    top = int(meta["top"])
    new_width = int(meta["new_width"])
    new_height = int(meta["new_height"])
    content = probs[:, top : top + new_height, left : left + new_width]
    return F.interpolate(content.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False).squeeze(0)


def mask_image(mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[mask.astype(bool), :3] = color
    out[mask.astype(bool), 3] = 150
    return Image.fromarray(out, mode="RGBA")


def make_overlay(image: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = image.convert("RGBA")
    pred_only = pred & ~gt
    gt_only = gt & ~pred
    overlap = pred & gt
    base.alpha_composite(mask_image(gt_only, (0, 210, 80)))
    base.alpha_composite(mask_image(pred_only, (230, 40, 40)))
    base.alpha_composite(mask_image(overlap, (255, 220, 0)))
    return base.convert("RGB")


def binary_l_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


def draw_box(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    out = image.convert("RGB")
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return out
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    ImageDraw.Draw(out).rectangle(box, outline=color, width=4)
    return out


def make_panel(image: Image.Image, gt: np.ndarray, pred: np.ndarray, overlay: Image.Image) -> Image.Image:
    width, height = image.size
    gt_rgb = Image.merge("RGB", (binary_l_image(gt), binary_l_image(gt), binary_l_image(gt)))
    pred_rgb = Image.merge("RGB", (binary_l_image(pred), binary_l_image(pred), binary_l_image(pred)))
    panel = Image.new("RGB", (width * 4, height), (255, 255, 255))
    panel.paste(image.convert("RGB"), (0, 0))
    panel.paste(gt_rgb, (width, 0))
    panel.paste(pred_rgb, (width * 2, 0))
    panel.paste(overlay, (width * 3, 0))
    return panel


def metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    tp = float((gt_bool & pred_bool).sum())
    fp = float((~gt_bool & pred_bool).sum())
    fn = float((gt_bool & ~pred_bool).sum())
    tn = float((~gt_bool & ~pred_bool).sum())
    eps = 1e-6
    return {
        "iou_full": tp / (tp + fp + fn + eps),
        "precision_full": tp / (tp + fp + eps),
        "recall_full": tp / (tp + fn + eps),
        "f1_full": (2 * tp) / (2 * tp + fp + fn + eps),
        "pred_area_ratio": float(pred_bool.mean()),
        "gt_area_ratio": float(gt_bool.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id2label, _ = load_classes(args.data_root)
    roi_class_ids = parse_class_ids(args.roi_positive_class_ids, len(id2label))
    rows = selected_rows(args)
    if not rows:
        raise ValueError("No rows selected for visualization.")

    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_dir).to(args.device).eval()

    summary = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            image = Image.open(row["image_path"]).convert("RGB")
            target = load_npz_mask(row["target_path"])
            valid_mask = load_npz_mask(row["valid_mask_path"])
            gt_roi, _ = make_roi_target_and_valid(
                target, valid_mask, roi_class_ids, args.roi_negative_valid_policy
            )
            gt = gt_roi[0].numpy().astype(bool)

            encoded, meta = preprocess_roi_image(image, image_processor, args.image_size, args.resize_mode)
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            logits = F.interpolate(logits, size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
            probs = probs_to_original(torch.sigmoid(logits)[0].cpu(), image, args.image_size, args.resize_mode, meta)
            pred = (probs[0].numpy() >= args.threshold)

            stem = f"{index:04d}_{row['dataset']}_{row['split']}_{row['id']}"
            sample_dir = args.output_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            overlay = make_overlay(image, gt, pred)
            image.save(sample_dir / "image.jpg")
            binary_l_image(gt).save(sample_dir / "gt_roi.png")
            binary_l_image(pred).save(sample_dir / "pred_roi.png")
            overlay.save(sample_dir / "overlay_gt_green_pred_red_overlap_yellow.jpg")
            draw_box(image, pred, (230, 40, 40)).save(sample_dir / "pred_box.jpg")
            make_panel(image, gt, pred, overlay).save(sample_dir / "panel_image_gt_pred_overlay.jpg")

            row_metrics = metrics(gt, pred)
            pred_box = mask_to_box(torch.from_numpy(pred.astype(np.uint8)), image.width, image.height, padding=0.0)
            box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            row_summary = {
                "dataset": row["dataset"],
                "split": row["split"],
                "id": row["id"],
                "sample_dir": str(sample_dir),
                "pred_box_area_ratio": box_area / (image.width * image.height),
                **row_metrics,
            }
            summary.append(row_summary)
            print(
                f"{stem}: iou={row_metrics['iou_full']:.4f} "
                f"recall={row_metrics['recall_full']:.4f} pred_area={row_metrics['pred_area_ratio']:.4f}"
            )

    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
