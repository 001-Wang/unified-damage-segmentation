#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage SegFormer training for unified damage segmentation.")
    parser.add_argument(
        "--stage",
        default="both",
        choices=["roi", "fine", "both"],
        help="Train the ROI model, the fine multi-label model, or both sequentially.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/grads/z/zuoxu/data/damage_data/unified_damage_data"),
        help="Directory containing train.csv, val.csv, classes.json, images/, masks_multilabel/, valid_masks/.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Fallback checkpoint for both stages when stage-specific checkpoint args are not set.",
    )
    parser.add_argument(
        "--stage1-model-name-or-path",
        default=None,
        help="Checkpoint for Stage 1 ROI segmentation. Defaults to --model-name-or-path.",
    )
    parser.add_argument(
        "--stage2-model-name-or-path",
        default=None,
        help="Checkpoint for Stage 2 fine multi-label segmentation. Defaults to --model-name-or-path.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/two_stage_damage_segformer"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--resize-mode",
        default="letterbox",
        choices=["letterbox", "stretch"],
        help="letterbox preserves aspect ratio and ignores padded pixels; stretch directly resizes to a square.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all training samples.")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 means use all validation samples.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means train by --num-train-epochs.")
    parser.add_argument("--num-train-epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size. Defaults to --batch-size if not set.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers for train/eval.")
    parser.add_argument(
        "--eval-accumulation-steps",
        type=int,
        default=1,
        help="Move eval predictions to CPU every N steps to reduce GPU memory pressure.",
    )
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--roi-tversky-weight", type=float, default=1.0)
    parser.add_argument("--roi-tversky-alpha", type=float, default=0.3, help="False-positive weight for ROI Tversky.")
    parser.add_argument("--roi-tversky-beta", type=float, default=0.7, help="False-negative weight for ROI Tversky.")
    parser.add_argument(
        "--roi-positive-class-ids",
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,18",
        help=(
            "Comma-separated class ids that define Stage 1 ROI positives. "
            "Default excludes object-part classes 13-17 and vegetation 19."
        ),
    )
    parser.add_argument(
        "--roi-negative-valid-policy",
        default="all",
        choices=["all", "any"],
        help=(
            "For ROI background pixels, all means supervise negatives only when every selected ROI class is annotated; "
            "any is less conservative and supervises negatives when at least one selected ROI class is annotated."
        ),
    )
    parser.add_argument("--fine-dice-weight", type=float, default=0.5)
    parser.add_argument("--roi-threshold", type=float, default=0.5)
    parser.add_argument("--fine-threshold", type=float, default=0.5)
    parser.add_argument(
        "--fine-roi-source",
        default="pred",
        choices=["pred", "gt", "none"],
        help="How Stage 2 chooses its crop. pred uses Stage 1 ROI predictions, gt uses target masks, none uses full images.",
    )
    parser.add_argument(
        "--stage1-model-dir",
        type=Path,
        default=None,
        help="Stage 1 ROI model directory to use when --stage fine --fine-roi-source pred.",
    )
    parser.add_argument(
        "--roi-crop-padding",
        type=float,
        default=0.15,
        help="Fractional padding added around the ROI box before Stage 2 cropping.",
    )
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or 'true' for latest. Only applies when --stage is not both.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report-to", default="none", choices=["none", "wandb", "tensorboard"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow model files to be downloaded if they are not already in the local Hugging Face cache.",
    )
    return parser.parse_args()


def load_classes(data_root: Path) -> tuple[dict[int, str], dict[str, int]]:
    with (data_root / "classes.json").open() as f:
        classes = json.load(f)
    id2label = {int(item["id"]): item["name"] for item in classes}
    label2id = {name: idx for idx, name in id2label.items()}
    return id2label, label2id


def read_rows(data_root: Path, split: str, max_samples: int) -> list[dict[str, str]]:
    csv_path = data_root / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            image_path = data_root / row["image_path"]
            target_path = data_root / row["target_path"]
            valid_mask_path = data_root / row["valid_mask_path"]
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")
            if not target_path.exists():
                raise FileNotFoundError(f"Missing target mask: {target_path}")
            if not valid_mask_path.exists():
                raise FileNotFoundError(f"Missing valid mask: {valid_mask_path}")
            rows.append(
                {
                    "image_path": str(image_path),
                    "target_path": str(target_path),
                    "valid_mask_path": str(valid_mask_path),
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "id": row["id"],
                }
            )

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def load_npz_mask(path: str) -> torch.Tensor:
    loaded = np.load(path)
    if "mask" in loaded.files:
        array = loaded["mask"]
    elif len(loaded.files) == 1:
        array = loaded[loaded.files[0]]
    else:
        raise ValueError(f"Expected one array or key 'mask' in {path}, got keys {loaded.files}")
    if array.ndim != 3:
        raise ValueError(f"Expected [C,H,W] mask in {path}, got shape {array.shape}")
    return torch.from_numpy(array).float()


def parse_class_ids(value: str, num_classes: int) -> list[int]:
    class_ids = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    invalid = [class_id for class_id in class_ids if class_id < 0 or class_id >= num_classes]
    if invalid:
        raise ValueError(f"Invalid class ids {invalid}; valid range is 0..{num_classes - 1}")
    if not class_ids:
        raise ValueError("--roi-positive-class-ids must contain at least one class id.")
    return class_ids


def make_roi_target_and_valid(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    class_ids: list[int],
    negative_valid_policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_target = target[class_ids]
    selected_valid = valid_mask[class_ids]
    roi_positive = (selected_target.sum(dim=0, keepdim=True) > 0).float()

    if negative_valid_policy == "all":
        reliable_negative = (selected_valid.prod(dim=0, keepdim=True) > 0).float()
    elif negative_valid_policy == "any":
        reliable_negative = (selected_valid.sum(dim=0, keepdim=True) > 0).float()
    else:
        raise ValueError(f"Unknown ROI negative valid policy: {negative_valid_policy}")

    roi_valid = torch.maximum(roi_positive, reliable_negative)
    return roi_positive, roi_valid


class TwoStageDamageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_processor: AutoImageProcessor,
        image_size: int,
        resize_mode: str,
        task: str,
        crop_boxes: dict[str, list[int]] | None = None,
    ) -> None:
        self.rows = rows
        self.image_processor = image_processor
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.task = task
        self.crop_boxes = crop_boxes or {}
        self.roi_class_ids: list[int] | None = None
        self.roi_negative_valid_policy = "all"

    def set_roi_policy(self, class_ids: list[int], negative_valid_policy: str) -> None:
        self.roi_class_ids = class_ids
        self.roi_negative_valid_policy = negative_valid_policy

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        target = load_npz_mask(row["target_path"])
        valid_mask = load_npz_mask(row["valid_mask_path"])

        if self.task == "roi":
            if self.roi_class_ids is None:
                raise ValueError("ROI dataset requires roi_class_ids to be configured.")
            target, valid_mask = make_roi_target_and_valid(
                target,
                valid_mask,
                self.roi_class_ids,
                self.roi_negative_valid_policy,
            )
        elif self.crop_boxes:
            box = self.crop_boxes.get(row_key(row))
            if box is not None:
                image, target, valid_mask = crop_sample(image, target, valid_mask, box)

        if self.resize_mode == "stretch":
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            target = resize_mask(target, self.image_size, self.image_size)
            valid_mask = resize_mask(valid_mask, self.image_size, self.image_size)
        else:
            image, target, valid_mask = self.letterbox(image, target, valid_mask)

        encoded = self.image_processor(images=image, do_resize=False, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": target,
            "valid_mask": valid_mask,
        }

    def letterbox(
        self, image: Image.Image, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        width, height = image.size
        scale = min(self.image_size / width, self.image_size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2

        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        image_canvas = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        image_canvas.paste(image, (left, top))

        target_resized = resize_mask(target, new_height, new_width)
        valid_resized = resize_mask(valid_mask, new_height, new_width)
        target_canvas = target.new_zeros((target.shape[0], self.image_size, self.image_size))
        valid_canvas = valid_mask.new_zeros((valid_mask.shape[0], self.image_size, self.image_size))
        target_canvas[:, top : top + new_height, left : left + new_width] = target_resized
        valid_canvas[:, top : top + new_height, left : left + new_width] = valid_resized
        return image_canvas, target_canvas, valid_canvas


def resize_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(mask.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)


def row_key(row: dict[str, str]) -> str:
    return f"{row['dataset']}:{row['split']}:{row['id']}"


def crop_sample(
    image: Image.Image, target: torch.Tensor, valid_mask: torch.Tensor, box: list[int]
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    left, top, right, bottom = box
    image = image.crop((left, top, right, bottom))
    target = target[:, top:bottom, left:right]
    valid_mask = valid_mask[:, top:bottom, left:right]
    return image, target, valid_mask


def mask_to_box(mask: torch.Tensor, width: int, height: int, padding: float) -> list[int]:
    ys, xs = torch.where(mask > 0)
    if ys.numel() == 0:
        return [0, 0, width, height]

    left = xs.min().item()
    right = xs.max().item() + 1
    top = ys.min().item()
    bottom = ys.max().item() + 1
    box_width = right - left
    box_height = bottom - top
    pad_x = round(box_width * padding)
    pad_y = round(box_height * padding)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    if right <= left or bottom <= top:
        return [0, 0, width, height]
    return [left, top, right, bottom]


def make_gt_roi_boxes(rows: list[dict[str, str]], class_ids: list[int], padding: float) -> dict[str, list[int]]:
    boxes = {}
    for row in rows:
        image = Image.open(row["image_path"])
        target = load_npz_mask(row["target_path"])
        roi = (target[class_ids].sum(dim=0) > 0).float()
        boxes[row_key(row)] = mask_to_box(roi, image.width, image.height, padding)
    return boxes


def preprocess_roi_image(
    image: Image.Image, image_processor: AutoImageProcessor, image_size: int, resize_mode: str
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    width, height = image.size
    if resize_mode == "stretch":
        resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        meta = {"scale_x": image_size / width, "scale_y": image_size / height, "left": 0, "top": 0}
    else:
        scale = min(image_size / width, image_size / height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        left = (image_size - new_width) // 2
        top = (image_size - new_height) // 2
        resized_content = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        resized = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        resized.paste(resized_content, (left, top))
        meta = {"scale": scale, "left": left, "top": top, "new_width": new_width, "new_height": new_height}

    encoded = image_processor(images=resized, do_resize=False, return_tensors="pt")
    return encoded, meta


def predicted_canvas_box_to_original(
    mask: torch.Tensor, image: Image.Image, resize_mode: str, meta: dict[str, float | int], padding: float
) -> list[int]:
    width, height = image.size
    if resize_mode == "stretch":
        canvas_box = mask_to_box(mask, mask.shape[1], mask.shape[0], padding=0.0)
        left, top, right, bottom = canvas_box
        scale_x = float(meta["scale_x"])
        scale_y = float(meta["scale_y"])
        box = [
            round(left / scale_x),
            round(top / scale_y),
            round(right / scale_x),
            round(bottom / scale_y),
        ]
    else:
        left_pad = int(meta["left"])
        top_pad = int(meta["top"])
        new_width = int(meta["new_width"])
        new_height = int(meta["new_height"])
        content = mask[top_pad : top_pad + new_height, left_pad : left_pad + new_width]
        content_box = mask_to_box(content, new_width, new_height, padding=0.0)
        scale = float(meta["scale"])
        left, top, right, bottom = content_box
        box = [
            round(left / scale),
            round(top / scale),
            round(right / scale),
            round(bottom / scale),
        ]
    return expand_box(box, width, height, padding)


def expand_box(box: list[int], width: int, height: int, padding: float) -> list[int]:
    left, top, right, bottom = box
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    pad_x = round(box_width * padding)
    pad_y = round(box_height * padding)
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    if right <= left or bottom <= top:
        return [0, 0, width, height]
    return [left, top, right, bottom]


def make_pred_roi_boxes(
    rows: list[dict[str, str]],
    stage1_model_dir: Path,
    image_size: int,
    resize_mode: str,
    threshold: float,
    padding: float,
    device: str,
) -> dict[str, list[int]]:
    image_processor = AutoImageProcessor.from_pretrained(stage1_model_dir)
    model = AutoModelForSemanticSegmentation.from_pretrained(stage1_model_dir).to(device)
    model.eval()
    boxes = {}
    for index, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        encoded, meta = preprocess_roi_image(image, image_processor, image_size, resize_mode)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        logits = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=False)
        roi = (torch.sigmoid(logits)[0, 0].cpu() >= threshold).float()
        boxes[row_key(row)] = predicted_canvas_box_to_original(roi, image, resize_mode, meta, padding)
        if (index + 1) % 100 == 0:
            print(f"Generated {index + 1}/{len(rows)} predicted ROI boxes from {stage1_model_dir}")
    return boxes


def masked_bce_loss(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * valid_mask
    return loss.sum() / (valid_mask.sum() + eps)


def masked_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask
    dims = (0, 2, 3)
    intersection = (prob * target).sum(dims)
    denominator = prob.sum(dims) + target.sum(dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    class_valid = (valid_mask.sum(dims) > 0).float()
    return ((1 - dice) * class_valid).sum() / (class_valid.sum() + eps)


def masked_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid_mask
    target = target * valid_mask
    dims = (0, 2, 3)
    true_positive = (prob * target).sum(dims)
    false_positive = (prob * (1 - target) * valid_mask).sum(dims)
    false_negative = ((1 - prob) * target).sum(dims)
    tversky = (true_positive + eps) / (true_positive + alpha * false_positive + beta * false_negative + eps)
    class_valid = (valid_mask.sum(dims) > 0).float()
    return ((1 - tversky) * class_valid).sum() / (class_valid.sum() + eps)


class RoiTrainer(Trainer):
    def __init__(self, *args, tversky_weight: float, tversky_alpha: float, tversky_beta: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        valid_mask = inputs.pop("valid_mask")
        outputs = model(**inputs)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = masked_bce_loss(logits, labels, valid_mask)
        if self.tversky_weight:
            loss = loss + self.tversky_weight * masked_tversky_loss(
                logits, labels, valid_mask, self.tversky_alpha, self.tversky_beta
            )
        return (loss, outputs) if return_outputs else loss


class FineTrainer(Trainer):
    def __init__(self, *args, dice_weight: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dice_weight = dice_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        valid_mask = inputs.pop("valid_mask")
        outputs = model(**inputs)
        logits = F.interpolate(outputs.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = masked_bce_loss(logits, labels, valid_mask)
        if self.dice_weight:
            loss = loss + self.dice_weight * masked_dice_loss(logits, labels, valid_mask)
        return (loss, outputs) if return_outputs else loss


def make_roi_metrics(threshold: float):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels_and_valid = eval_pred
        labels, valid_mask = labels_and_valid
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels).float()
        valid_mask = torch.from_numpy(valid_mask).float()
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logits) >= threshold).float() * valid_mask
        labels = labels * valid_mask
        true_positive = (pred * labels).sum()
        false_positive = (pred * (1 - labels) * valid_mask).sum()
        false_negative = ((1 - pred) * labels).sum()
        true_negative = ((1 - pred) * (1 - labels) * valid_mask).sum()
        iou = true_positive / (true_positive + false_positive + false_negative + 1e-6)
        f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-6)
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        accuracy = (true_positive + true_negative) / (valid_mask.sum() + 1e-6)
        return {
            "miou": iou.item(),
            "roi_iou": iou.item(),
            "roi_f1": f1.item(),
            "roi_precision": precision.item(),
            "roi_recall": recall.item(),
            "roi_accuracy": accuracy.item(),
        }

    return compute_metrics


def make_fine_metrics(threshold: float):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels_and_valid = eval_pred
        labels, valid_mask = labels_and_valid
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels).float()
        valid_mask = torch.from_numpy(valid_mask).float()
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logits) >= threshold).float() * valid_mask
        labels = labels * valid_mask
        dims = (0, 2, 3)
        true_positive = (pred * labels).sum(dims)
        false_positive = (pred * (1 - labels) * valid_mask).sum(dims)
        false_negative = ((1 - pred) * labels).sum(dims)
        valid_classes = valid_mask.sum(dims) > 0
        iou = true_positive / (true_positive + false_positive + false_negative + 1e-6)
        f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-6)
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        mean_iou = iou[valid_classes].mean().item() if valid_classes.any() else 0.0
        metrics = {
            "miou": mean_iou,
            "mean_iou": mean_iou,
            "mean_f1": f1[valid_classes].mean().item() if valid_classes.any() else 0.0,
            "mean_precision": precision[valid_classes].mean().item() if valid_classes.any() else 0.0,
            "mean_recall": recall[valid_classes].mean().item() if valid_classes.any() else 0.0,
        }
        for class_id in range(labels.shape[1]):
            if valid_classes[class_id]:
                metrics[f"class_{class_id:02d}_iou"] = iou[class_id].item()
                metrics[f"class_{class_id:02d}_f1"] = f1[class_id].item()
        return metrics

    return compute_metrics


def make_model(
    model_name_or_path: str,
    id2label: dict[int, str],
    label2id: dict[str, int],
    local_files_only: bool,
):
    config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )
    return model, image_processor


def make_training_args(args: argparse.Namespace, output_dir: Path, run_name: str | None) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=args.eval_strategy != "no",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="polynomial",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size or args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=run_name,
        fp16=args.fp16,
        seed=args.seed,
        label_names=["labels", "valid_mask"],
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        eval_accumulation_steps=args.eval_accumulation_steps,
        metric_for_best_model="eval_miou",
        greater_is_better=True,
    )


class BestMiouCheckpointCallback(TrainerCallback):
    def __init__(self, metric_name: str = "eval_miou") -> None:
        self.metric_name = metric_name
        self.best_metric: float | None = None

    def on_save(self, args, state, control, **kwargs):
        current_metric = None
        for record in reversed(state.log_history):
            if record.get("step") != state.global_step:
                continue
            if self.metric_name in record:
                current_metric = float(record[self.metric_name])
                break

        if current_metric is None:
            return control
        if self.best_metric is not None and current_metric <= self.best_metric:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return control

        best_dir = Path(args.output_dir) / "best-checkpoint"
        tmp_dir = Path(args.output_dir) / "best-checkpoint.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(checkpoint_dir, tmp_dir)
        if best_dir.exists():
            shutil.rmtree(best_dir)
        tmp_dir.rename(best_dir)
        self.best_metric = current_metric
        print(f"Updated best checkpoint by {self.metric_name}: {best_dir} ({current_metric:.6f})")
        return control


def get_best_checkpoint_from_log_history(trainer: Trainer, metric_name: str = "eval_miou") -> Path | None:
    best_checkpoint = Path(trainer.args.output_dir) / "best-checkpoint"
    if best_checkpoint.exists():
        print(f"Best checkpoint by {metric_name}: {best_checkpoint}")
        return best_checkpoint

    best_step = None
    best_metric = None
    for record in trainer.state.log_history:
        if metric_name not in record or "step" not in record:
            continue
        metric = float(record[metric_name])
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_step = int(record["step"])

    if best_step is None:
        return None
    checkpoint = Path(trainer.args.output_dir) / f"checkpoint-{best_step}"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint by {metric_name} was not found: {checkpoint}")
    print(f"Best checkpoint by {metric_name}: {checkpoint} ({best_metric:.6f})")
    return checkpoint


def save_best_or_current_model(trainer: Trainer, image_processor: AutoImageProcessor) -> None:
    best_checkpoint = get_best_checkpoint_from_log_history(trainer)
    if best_checkpoint is None:
        trainer.save_model()
        return

    model = AutoModelForSemanticSegmentation.from_pretrained(best_checkpoint)
    model.save_pretrained(trainer.args.output_dir)
    image_processor.save_pretrained(trainer.args.output_dir)


def train_roi(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    roi_class_ids: list[int],
) -> Path:
    local_files_only = not args.allow_download
    id2label = {0: "roi"}
    label2id = {"roi": 0}
    model_name_or_path = args.stage1_model_name_or_path or args.model_name_or_path
    print(f"Stage 1 model: {model_name_or_path}")
    model, image_processor = make_model(model_name_or_path, id2label, label2id, local_files_only)
    train_dataset = TwoStageDamageDataset(train_rows, image_processor, args.image_size, args.resize_mode, task="roi")
    eval_dataset = TwoStageDamageDataset(eval_rows, image_processor, args.image_size, args.resize_mode, task="roi")
    train_dataset.set_roi_policy(roi_class_ids, args.roi_negative_valid_policy)
    eval_dataset.set_roi_policy(roi_class_ids, args.roi_negative_valid_policy)
    output_dir = args.output_dir / "stage1_roi"
    trainer = RoiTrainer(
        model=model,
        args=make_training_args(args, output_dir, add_run_suffix(args.run_name, "roi")),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=default_data_collator,
        compute_metrics=make_roi_metrics(args.roi_threshold),
        callbacks=[BestMiouCheckpointCallback()],
        tversky_weight=args.roi_tversky_weight,
        tversky_alpha=args.roi_tversky_alpha,
        tversky_beta=args.roi_tversky_beta,
    )
    trainer.train(resume_from_checkpoint=resolve_resume(args))
    save_best_or_current_model(trainer, image_processor)
    if args.eval_strategy != "no":
        trainer.save_metrics("eval", trainer.evaluate())
    print(f"Saved stage 1 ROI model to {output_dir}")
    return output_dir


def train_fine(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    id2label: dict[int, str],
    label2id: dict[str, int],
    train_crop_boxes: dict[str, list[int]] | None,
    eval_crop_boxes: dict[str, list[int]] | None,
) -> None:
    local_files_only = not args.allow_download
    model_name_or_path = args.stage2_model_name_or_path or args.model_name_or_path
    print(f"Stage 2 model: {model_name_or_path}")
    model, image_processor = make_model(model_name_or_path, id2label, label2id, local_files_only)
    train_dataset = TwoStageDamageDataset(
        train_rows,
        image_processor,
        args.image_size,
        args.resize_mode,
        task="fine",
        crop_boxes=train_crop_boxes,
    )
    eval_dataset = TwoStageDamageDataset(
        eval_rows,
        image_processor,
        args.image_size,
        args.resize_mode,
        task="fine",
        crop_boxes=eval_crop_boxes,
    )
    output_dir = args.output_dir / "stage2_fine"
    trainer = FineTrainer(
        model=model,
        args=make_training_args(args, output_dir, add_run_suffix(args.run_name, "fine")),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=default_data_collator,
        compute_metrics=make_fine_metrics(args.fine_threshold),
        callbacks=[BestMiouCheckpointCallback()],
        dice_weight=args.fine_dice_weight,
    )
    trainer.train(resume_from_checkpoint=resolve_resume(args))
    save_best_or_current_model(trainer, image_processor)
    if args.eval_strategy != "no":
        trainer.save_metrics("eval", trainer.evaluate())
    print(f"Saved stage 2 fine segmentation model to {output_dir}")


def add_run_suffix(run_name: str | None, suffix: str) -> str | None:
    if run_name is None:
        return None
    return f"{run_name}-{suffix}"


def resolve_resume(args: argparse.Namespace):
    if args.stage == "both":
        return None
    if args.resume_from_checkpoint is not None and args.resume_from_checkpoint.lower() == "true":
        return True
    return args.resume_from_checkpoint


def make_stage2_crop_boxes(
    args: argparse.Namespace,
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    stage1_model_dir: Path | None,
    roi_class_ids: list[int],
) -> tuple[dict[str, list[int]] | None, dict[str, list[int]] | None]:
    if args.fine_roi_source == "none":
        print("Stage 2 ROI source is none: using full images for fine segmentation.")
        return None, None

    if args.fine_roi_source == "gt":
        print("Stage 2 ROI source is gt: using target-derived ROI crops.")
        return (
            make_gt_roi_boxes(train_rows, roi_class_ids, args.roi_crop_padding),
            make_gt_roi_boxes(eval_rows, roi_class_ids, args.roi_crop_padding),
        )

    model_dir = args.stage1_model_dir or stage1_model_dir
    if model_dir is None:
        raise ValueError("--fine-roi-source pred requires a Stage 1 model. Use --stage both or pass --stage1-model-dir.")
    print(f"Stage 2 ROI source is pred: generating crops from {model_dir}")
    return (
        make_pred_roi_boxes(
            train_rows,
            model_dir,
            args.image_size,
            args.resize_mode,
            args.roi_threshold,
            args.roi_crop_padding,
            args.device,
        ),
        make_pred_roi_boxes(
            eval_rows,
            model_dir,
            args.image_size,
            args.resize_mode,
            args.roi_threshold,
            args.roi_crop_padding,
            args.device,
        ),
    )


def validate_best_model_settings(args: argparse.Namespace) -> None:
    if args.eval_strategy == "no":
        raise ValueError("Best-model saving by eval_miou requires --eval-strategy to be 'epoch' or 'steps'.")
    if args.save_strategy == "no":
        raise ValueError("Best-model saving by eval_miou requires --save-strategy to be 'epoch' or 'steps'.")
    if args.eval_strategy != args.save_strategy:
        raise ValueError("Best-model saving requires --eval-strategy and --save-strategy to match.")
    if args.eval_strategy == "steps" and args.save_steps % args.eval_steps != 0:
        raise ValueError("Best-model saving with steps requires --save-steps to be a multiple of --eval-steps.")


def main() -> None:
    args = parse_args()
    validate_best_model_settings(args)
    id2label, label2id = load_classes(args.data_root)
    roi_class_ids = parse_class_ids(args.roi_positive_class_ids, len(id2label))
    train_rows = read_rows(args.data_root, "train", args.max_train_samples)
    eval_rows = read_rows(args.data_root, "val", args.max_eval_samples)
    print(f"Loaded {len(train_rows)} train rows and {len(eval_rows)} validation rows from {args.data_root}")
    print(f"Stage 1 ROI positive class ids: {roi_class_ids}")
    print(f"Stage 1 ROI negative valid policy: {args.roi_negative_valid_policy}")

    stage1_model_dir = None
    if args.stage in {"roi", "both"}:
        stage1_model_dir = train_roi(args, train_rows, eval_rows, roi_class_ids)
    if args.stage in {"fine", "both"}:
        train_crop_boxes, eval_crop_boxes = make_stage2_crop_boxes(
            args, train_rows, eval_rows, stage1_model_dir, roi_class_ids
        )
        train_fine(args, train_rows, eval_rows, id2label, label2id, train_crop_boxes, eval_crop_boxes)


if __name__ == "__main__":
    main()
