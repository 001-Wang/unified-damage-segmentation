#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, default_data_collator

from train_two_stage_damage_segformer import (
    TwoStageDamageDataset,
    load_classes,
    parse_class_ids,
    read_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Stage 1 ROI model by source dataset.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Trained Stage 1 ROI model directory.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/grads/z/zuoxu/data/damage_data/unified_damage_data"),
        help="Directory containing split CSVs, classes.json, images/, masks_multilabel/, valid_masks/.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to evaluate, for example: dacl hrcds s2ds lcw.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all selected samples.")
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
        help="Must match the Stage 1 training policy for comparable metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
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


def read_selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    rows: list[dict[str, str]] = []
    for split in splits:
        rows.extend(read_rows(args.data_root, split, max_samples=0))

    if args.datasets is not None:
        selected = set(args.datasets)
        rows = [row for row in rows if row["dataset"] in selected]

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No rows selected for evaluation.")
    return rows


class DatasetAwareRoiDataset(TwoStageDamageDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = super().__getitem__(index)
        row = self.rows[index]
        item["dataset"] = row["dataset"]
        item["id"] = row["id"]
        return item


def collate_with_metadata(features: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    tensor_features = [
        {key: value for key, value in feature.items() if key not in {"dataset", "id"}} for feature in features
    ]
    batch = default_data_collator(tensor_features)
    batch["dataset"] = [str(feature["dataset"]) for feature in features]
    batch["id"] = [str(feature["id"]) for feature in features]
    return batch


def empty_stats() -> dict[str, float]:
    return {
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "valid_pixels": 0.0,
        "positive_pixels": 0.0,
        "total_pixels": 0.0,
        "pred_positive_pixels_full": 0.0,
        "gt_positive_pixels_full": 0.0,
        "intersection_full": 0.0,
        "union_full": 0.0,
        "samples": 0.0,
    }


def update_stats(stats: dict[str, float], pred: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor) -> None:
    stats["total_pixels"] += float(labels.numel())
    stats["pred_positive_pixels_full"] += pred.sum().item()
    stats["gt_positive_pixels_full"] += labels.sum().item()
    stats["intersection_full"] += (pred * labels).sum().item()
    stats["union_full"] += ((pred + labels) > 0).float().sum().item()

    pred = pred * valid_mask
    labels = labels * valid_mask
    stats["tp"] += (pred * labels).sum().item()
    stats["fp"] += (pred * (1 - labels) * valid_mask).sum().item()
    stats["fn"] += ((1 - pred) * labels).sum().item()
    stats["tn"] += ((1 - pred) * (1 - labels) * valid_mask).sum().item()
    stats["valid_pixels"] += valid_mask.sum().item()
    stats["positive_pixels"] += labels.sum().item()
    stats["samples"] += float(labels.shape[0])


def finalize_stats(name: str, stats: dict[str, float]) -> dict[str, float | str]:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]
    valid_pixels = stats["valid_pixels"]
    positive_pixels = stats["positive_pixels"]
    total_pixels = stats["total_pixels"]
    eps = 1e-6
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    accuracy = (tp + tn) / (valid_pixels + eps)
    pred_area_ratio_full = stats["pred_positive_pixels_full"] / (total_pixels + eps)
    gt_area_ratio_full = stats["gt_positive_pixels_full"] / (total_pixels + eps)
    roi_iou_full = stats["intersection_full"] / (stats["union_full"] + eps)
    return {
        "dataset": name,
        "samples": int(stats["samples"]),
        "valid_pixels": int(valid_pixels),
        "total_pixels": int(total_pixels),
        "valid_pixel_ratio": valid_pixels / (total_pixels + eps),
        "positive_pixels": int(positive_pixels),
        "miou": iou,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "pred_area_ratio_full": pred_area_ratio_full,
        "gt_area_ratio_full": gt_area_ratio_full,
        "roi_iou_full": roi_iou_full,
    }


def print_table(results: list[dict[str, float | str]]) -> None:
    fields = [
        "dataset",
        "samples",
        "miou",
        "precision",
        "recall",
        "f1",
        "valid_pixel_ratio",
        "pred_area_ratio_full",
        "gt_area_ratio_full",
        "roi_iou_full",
    ]
    print("\t".join(fields))
    for row in results:
        values = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        print("\t".join(values))


def write_csv(path: Path, results: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    id2label, _ = load_classes(args.data_root)
    roi_class_ids = parse_class_ids(args.roi_positive_class_ids, len(id2label))
    rows = read_selected_rows(args)

    image_processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_dir).to(args.device).eval()

    dataset = DatasetAwareRoiDataset(
        rows=rows,
        image_processor=image_processor,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        task="roi",
    )
    dataset.set_roi_policy(roi_class_ids, args.roi_negative_valid_policy)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_with_metadata,
    )

    stats_by_dataset: dict[str, dict[str, float]] = defaultdict(empty_stats)
    overall = empty_stats()

    with torch.no_grad():
        for batch in dataloader:
            dataset_names = batch.pop("dataset")
            batch.pop("id")
            labels = batch.pop("labels").to(args.device)
            valid_mask = batch.pop("valid_mask").to(args.device)
            pixel_values = batch.pop("pixel_values").to(args.device)

            logits = model(pixel_values=pixel_values).logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logits) >= args.threshold).float()

            update_stats(overall, pred.cpu(), labels.cpu(), valid_mask.cpu())
            for sample_index, dataset_name in enumerate(dataset_names):
                update_stats(
                    stats_by_dataset[dataset_name],
                    pred[sample_index : sample_index + 1].cpu(),
                    labels[sample_index : sample_index + 1].cpu(),
                    valid_mask[sample_index : sample_index + 1].cpu(),
                )

    results = [finalize_stats("overall", overall)]
    results.extend(finalize_stats(name, stats_by_dataset[name]) for name in sorted(stats_by_dataset))
    print_table(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)


if __name__ == "__main__":
    main()
