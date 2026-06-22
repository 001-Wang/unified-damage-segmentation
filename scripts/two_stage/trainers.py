"""Masked segmentation losses and Hugging Face Trainer subclasses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


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
