#!/usr/bin/env python
"""Command-line arguments for two-stage training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


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
    parser.add_argument(
        "--streaming-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use memory-safe streaming validation for segmentation metrics. "
            "This avoids Hugging Face Trainer accumulating all validation logits/labels."
        ),
    )
    parser.add_argument(
        "--streaming-eval-log-steps",
        type=int,
        default=100,
        help="Print progress every N validation batches during streaming eval. 0 disables progress prints.",
    )
    parser.add_argument(
        "--stage1-metric-for-best-model",
        default="eval_miou",
        help="Metric used to select the best Stage 1 checkpoint when streaming eval is enabled.",
    )
    parser.add_argument(
        "--stage2-metric-for-best-model",
        default="eval_miou",
        help="Metric used to select the best Stage 2 checkpoint when streaming eval is enabled.",
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
