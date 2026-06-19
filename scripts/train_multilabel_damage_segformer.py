#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
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
    TrainingArguments,
    default_data_collator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer for unified multi-label damage segmentation.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/grads/z/zuoxu/data/damage_data/unified_damage_data"),
        help="Directory containing train.csv, val.csv, classes.json, images/, masks_multilabel/, valid_masks/.",
    )
    parser.add_argument("--model-name-or-path", default="nvidia/segformer-b3-finetuned-ade-512-512")
    parser.add_argument("--output-dir", type=Path, default=Path("./runs/multilabel_damage_segformer"))
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
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for validation metrics.")
    parser.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--resume-from-checkpoint", default=None, help="Checkpoint path, or 'true' for latest.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
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


class UnifiedDamageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        image_processor: AutoImageProcessor,
        image_size: int,
        resize_mode: str,
    ) -> None:
        self.rows = rows
        self.image_processor = image_processor
        self.image_size = image_size
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        target = load_npz_mask(row["target_path"])
        valid_mask = load_npz_mask(row["valid_mask_path"])

        if self.resize_mode == "stretch":
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            target = F.interpolate(
                target.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest"
            ).squeeze(0)
            valid_mask = F.interpolate(
                valid_mask.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest"
            ).squeeze(0)
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

        target_resized = F.interpolate(target.unsqueeze(0), size=(new_height, new_width), mode="nearest").squeeze(0)
        valid_resized = F.interpolate(valid_mask.unsqueeze(0), size=(new_height, new_width), mode="nearest").squeeze(0)
        target_canvas = target.new_zeros((target.shape[0], self.image_size, self.image_size))
        valid_canvas = valid_mask.new_zeros((valid_mask.shape[0], self.image_size, self.image_size))
        target_canvas[:, top : top + new_height, left : left + new_width] = target_resized
        valid_canvas[:, top : top + new_height, left : left + new_width] = valid_resized
        return image_canvas, target_canvas, valid_canvas


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


class MultilabelSegmentationTrainer(Trainer):
    def __init__(self, *args, dice_weight: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dice_weight = dice_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        valid_mask = inputs.pop("valid_mask")
        outputs = model(**inputs)
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = masked_bce_loss(logits, labels, valid_mask)
        if self.dice_weight:
            loss = loss + self.dice_weight * masked_dice_loss(logits, labels, valid_mask)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(threshold: float):
    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels_and_valid = eval_pred
        if not isinstance(labels_and_valid, (tuple, list)) or len(labels_and_valid) != 2:
            raise ValueError("Expected eval labels to contain both labels and valid_mask.")
        labels, valid_mask = labels_and_valid
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        valid_mask = torch.from_numpy(valid_mask)

        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logits) >= threshold).float()
        labels = labels.float()
        valid_mask = valid_mask.float()

        pred = pred * valid_mask
        labels = labels * valid_mask
        dims = (0, 2, 3)
        tp = (pred * labels).sum(dims)
        fp = (pred * (1 - labels) * valid_mask).sum(dims)
        fn = ((1 - pred) * labels).sum(dims)
        valid_classes = valid_mask.sum(dims) > 0

        iou = tp / (tp + fp + fn + 1e-6)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        metrics = {
            "mean_iou": iou[valid_classes].mean().item() if valid_classes.any() else 0.0,
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


def main() -> None:
    args = parse_args()
    local_files_only = not args.allow_download

    id2label, label2id = load_classes(args.data_root)
    train_rows = read_rows(args.data_root, "train", args.max_train_samples)
    eval_rows = read_rows(args.data_root, "val", args.max_eval_samples)
    print(f"Loaded {len(train_rows)} train rows and {len(eval_rows)} validation rows from {args.data_root}")
    print(f"Classes: {id2label}")

    config = AutoConfig.from_pretrained(args.model_name_or_path, local_files_only=local_files_only)
    config.num_labels = len(id2label)
    config.id2label = id2label
    config.label2id = label2id
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path, local_files_only=local_files_only)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )

    train_dataset = UnifiedDamageDataset(train_rows, image_processor, args.image_size, args.resize_mode)
    eval_dataset = UnifiedDamageDataset(eval_rows, image_processor, args.image_size, args.resize_mode)
    max_steps = args.max_steps if args.max_steps > 0 else -1

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=args.eval_strategy != "no",
        max_steps=max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="polynomial",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
        run_name=args.run_name,
        fp16=args.fp16,
        seed=args.seed,
        label_names=["labels", "valid_mask"],
    )

    trainer = MultilabelSegmentationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=default_data_collator,
        compute_metrics=make_compute_metrics(args.threshold),
        dice_weight=args.dice_weight,
    )

    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint is not None and resume_from_checkpoint.lower() == "true":
        resume_from_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    if args.eval_strategy != "no":
        trainer.save_metrics("eval", trainer.evaluate())
    print(f"Saved multi-label damage SegFormer model to {args.output_dir}")


if __name__ == "__main__":
    main()
