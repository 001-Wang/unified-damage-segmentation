"""Memory-safe segmentation evaluation and metric callbacks."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback

from .trainers import masked_bce_loss, masked_dice_loss, masked_tversky_loss


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def streaming_evaluate_segmentation(
    model: torch.nn.Module,
    eval_dataloader,
    threshold: float,
    task: str,
    device: torch.device,
    dice_weight: float = 0.0,
    tversky_weight: float = 0.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    log_steps: int = 100,
) -> dict[str, float]:
    """Memory-safe full-validation metrics for dense segmentation.

    Hugging Face Trainer's default compute_metrics path stores full validation
    logits/labels before computing metrics. For [N,C,H,W] segmentation tensors,
    this can exhaust CPU RAM. This function evaluates batch-by-batch and keeps
    only per-class TP/FP/FN counters.
    """
    was_training = model.training
    model.eval()

    tp: torch.Tensor | None = None
    fp: torch.Tensor | None = None
    fn: torch.Tensor | None = None
    valid_sum: torch.Tensor | None = None

    loss_sum = 0.0
    num_batches = 0
    num_samples = 0
    start_time = time.time()

    use_cuda_amp = device.type == "cuda"
    autocast = torch.cuda.amp.autocast if use_cuda_amp else torch.cpu.amp.autocast

    for step, batch in enumerate(eval_dataloader, start=1):
        batch = _move_batch_to_device(batch, device)
        labels = batch.pop("labels").float()
        valid_mask = batch.pop("valid_mask").float()

        with autocast(enabled=use_cuda_amp):
            outputs = model(**batch)
            logits = F.interpolate(
                outputs.logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = masked_bce_loss(logits, labels, valid_mask)
            if task == "fine" and dice_weight:
                loss = loss + dice_weight * masked_dice_loss(logits, labels, valid_mask)
            elif task == "roi" and tversky_weight:
                loss = loss + tversky_weight * masked_tversky_loss(
                    logits,
                    labels,
                    valid_mask,
                    tversky_alpha,
                    tversky_beta,
                )

        logits = logits.float()
        pred = (torch.sigmoid(logits) >= threshold).float() * valid_mask
        labels_valid = labels * valid_mask
        dims = (0, 2, 3)

        batch_tp = (pred * labels_valid).sum(dims).detach().cpu().double()
        batch_fp = (pred * (1 - labels_valid) * valid_mask).sum(dims).detach().cpu().double()
        batch_fn = ((1 - pred) * labels_valid).sum(dims).detach().cpu().double()
        batch_valid = valid_mask.sum(dims).detach().cpu().double()

        if tp is None:
            tp = batch_tp
            fp = batch_fp
            fn = batch_fn
            valid_sum = batch_valid
        else:
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            valid_sum += batch_valid

        loss_sum += float(loss.detach().cpu().item())
        num_batches += 1
        num_samples += int(labels.shape[0])

        if log_steps > 0 and step % log_steps == 0:
            elapsed = time.time() - start_time
            print(
                f"Streaming eval {task}: {step}/{len(eval_dataloader)} batches, "
                f"{num_samples} samples, elapsed={elapsed:.1f}s",
                flush=True,
            )

        del batch, labels, valid_mask, logits, pred, labels_valid, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if was_training:
        model.train()

    if tp is None or fp is None or fn is None or valid_sum is None:
        raise RuntimeError("Streaming evaluation received no batches.")

    valid_classes = valid_sum > 0
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    mean_iou = iou[valid_classes].mean().item() if valid_classes.any() else 0.0
    mean_f1 = f1[valid_classes].mean().item() if valid_classes.any() else 0.0
    mean_precision = precision[valid_classes].mean().item() if valid_classes.any() else 0.0
    mean_recall = recall[valid_classes].mean().item() if valid_classes.any() else 0.0

    elapsed = time.time() - start_time
    metrics: dict[str, float] = {
        "eval_loss": loss_sum / max(1, num_batches),
        "eval_miou": mean_iou,
        "eval_mean_iou": mean_iou,
        "eval_mean_f1": mean_f1,
        "eval_mean_precision": mean_precision,
        "eval_mean_recall": mean_recall,
        "eval_runtime": elapsed,
        "eval_samples_per_second": num_samples / elapsed if elapsed > 0 else 0.0,
    }

    if task == "roi":
        metrics.update(
            {
                "eval_roi_iou": mean_iou,
                "eval_roi_f1": mean_f1,
                "eval_roi_precision": mean_precision,
                "eval_roi_recall": mean_recall,
            }
        )
    else:
        for class_id in range(len(iou)):
            if valid_classes[class_id]:
                metrics[f"eval_class_{class_id:02d}_iou"] = iou[class_id].item()
                metrics[f"eval_class_{class_id:02d}_f1"] = f1[class_id].item()

    return metrics


class StreamingEvalCheckpointCallback(TrainerCallback):
    """Run streaming full-validation evaluation when Trainer saves a checkpoint.

    This avoids the default Trainer compute_metrics accumulation path and copies
    the best checkpoint to <output_dir>/best-checkpoint.
    """

    def __init__(
        self,
        task: str,
        metric_name: str,
        threshold: float,
        dice_weight: float = 0.0,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        log_steps: int = 100,
    ) -> None:
        self.task = task
        self.metric_name = metric_name
        self.threshold = threshold
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.log_steps = log_steps
        self.best_metric: float | None = None
        self.last_metrics: dict[str, float] | None = None
        self.trainer: Trainer | None = None

    def set_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control

        # Only the main process should run validation/copy best checkpoints.
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            print(f"Streaming eval skipped because checkpoint is missing: {checkpoint_dir}")
            return control

        print(
            f"Running streaming {self.task} evaluation for checkpoint {checkpoint_dir}",
            flush=True,
        )
        eval_dataloader = self.trainer.get_eval_dataloader()
        metrics = streaming_evaluate_segmentation(
            model=self.trainer.model,
            eval_dataloader=eval_dataloader,
            threshold=self.threshold,
            task=self.task,
            device=args.device,
            dice_weight=self.dice_weight,
            tversky_weight=self.tversky_weight,
            tversky_alpha=self.tversky_alpha,
            tversky_beta=self.tversky_beta,
            log_steps=self.log_steps,
        )
        metrics["epoch"] = float(state.epoch) if state.epoch is not None else 0.0
        self.last_metrics = metrics

        self.trainer.log(metrics)

        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            print(
                f"Metric {self.metric_name} was not produced by streaming eval. "
                f"Available keys: {sorted(metrics.keys())}"
            )
            return control

        if self.best_metric is not None and float(current_metric) <= self.best_metric:
            print(
                f"Streaming eval {self.metric_name}={float(current_metric):.6f} "
                f"did not improve best={self.best_metric:.6f}"
            )
            return control

        best_dir = Path(args.output_dir) / "best-checkpoint"
        tmp_dir = Path(args.output_dir) / "best-checkpoint.tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(checkpoint_dir, tmp_dir)
        if best_dir.exists():
            shutil.rmtree(best_dir)
        tmp_dir.rename(best_dir)
        self.best_metric = float(current_metric)
        print(
            f"Updated best checkpoint by {self.metric_name}: "
            f"{best_dir} ({self.best_metric:.6f})",
            flush=True,
        )
        return control


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
