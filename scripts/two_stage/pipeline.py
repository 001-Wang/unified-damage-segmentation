"""Stage 1 and Stage 2 training orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import TrainerCallback, default_data_collator

from .cli import parse_args
from .data import (
    TwoStageDamageDataset,
    load_classes,
    make_gt_roi_boxes,
    make_pred_roi_boxes,
    parse_class_ids,
    read_rows,
)
from .evaluation import (
    StreamingEvalCheckpointCallback,
    make_fine_metrics,
    make_roi_metrics,
)
from .modeling import (
    BestMiouCheckpointCallback,
    make_model,
    make_training_args,
    save_best_or_current_model,
)
from .trainers import FineTrainer, RoiTrainer


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
    streaming_callback = None
    callbacks: list[TrainerCallback] = []
    if args.streaming_eval:
        streaming_callback = StreamingEvalCheckpointCallback(
            task="roi",
            metric_name=args.stage1_metric_for_best_model,
            threshold=args.roi_threshold,
            tversky_weight=args.roi_tversky_weight,
            tversky_alpha=args.roi_tversky_alpha,
            tversky_beta=args.roi_tversky_beta,
            log_steps=args.streaming_eval_log_steps,
        )
        callbacks.append(streaming_callback)
        compute_metrics = None
    else:
        callbacks.append(BestMiouCheckpointCallback(metric_name=args.stage1_metric_for_best_model))
        compute_metrics = make_roi_metrics(args.roi_threshold)

    trainer = RoiTrainer(
        model=model,
        args=make_training_args(args, output_dir, add_run_suffix(args.run_name, "roi")),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        tversky_weight=args.roi_tversky_weight,
        tversky_alpha=args.roi_tversky_alpha,
        tversky_beta=args.roi_tversky_beta,
    )
    if streaming_callback is not None:
        streaming_callback.set_trainer(trainer)

    trainer.train(resume_from_checkpoint=resolve_resume(args))
    save_best_or_current_model(trainer, image_processor)
    if args.streaming_eval and streaming_callback is not None and streaming_callback.last_metrics is not None:
        trainer.save_metrics("eval", streaming_callback.last_metrics)
    elif args.eval_strategy != "no":
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
    streaming_callback = None
    callbacks: list[TrainerCallback] = []
    if args.streaming_eval:
        streaming_callback = StreamingEvalCheckpointCallback(
            task="fine",
            metric_name=args.stage2_metric_for_best_model,
            threshold=args.fine_threshold,
            dice_weight=args.fine_dice_weight,
            log_steps=args.streaming_eval_log_steps,
        )
        callbacks.append(streaming_callback)
        compute_metrics = None
    else:
        callbacks.append(BestMiouCheckpointCallback(metric_name=args.stage2_metric_for_best_model))
        compute_metrics = make_fine_metrics(args.fine_threshold)

    trainer = FineTrainer(
        model=model,
        args=make_training_args(args, output_dir, add_run_suffix(args.run_name, "fine")),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        dice_weight=args.fine_dice_weight,
    )
    if streaming_callback is not None:
        streaming_callback.set_trainer(trainer)

    trainer.train(resume_from_checkpoint=resolve_resume(args))
    save_best_or_current_model(trainer, image_processor)
    if args.streaming_eval and streaming_callback is not None and streaming_callback.last_metrics is not None:
        trainer.save_metrics("eval", streaming_callback.last_metrics)
    elif args.eval_strategy != "no":
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
    if args.streaming_eval:
        if args.save_strategy == "no":
            raise ValueError("Streaming best-model saving requires --save-strategy to be 'epoch' or 'steps'.")
        if args.save_strategy == "steps" and args.save_steps <= 0:
            raise ValueError("Streaming best-model saving with steps requires --save-steps > 0.")
        print(
            "Streaming evaluation is enabled: Hugging Face Trainer built-in eval is disabled, "
            "and full validation metrics are computed batch-by-batch at each checkpoint save."
        )
        return

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
