"""Model construction, training arguments, and checkpoint persistence."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


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
    # When streaming_eval is enabled, disable Hugging Face Trainer's built-in
    # evaluation loop. The built-in loop accumulates full segmentation logits,
    # labels, and valid masks before calling compute_metrics, which can exhaust
    # CPU RAM for dense segmentation. We still keep save_strategy active; a
    # custom callback runs memory-safe streaming validation on each saved
    # checkpoint.
    trainer_eval_strategy = "no" if args.streaming_eval else args.eval_strategy
    trainer_do_eval = (not args.streaming_eval) and args.eval_strategy != "no"

    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=trainer_do_eval,
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
        eval_strategy=trainer_eval_strategy,
        eval_steps=args.eval_steps if trainer_eval_strategy == "steps" else None,
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
