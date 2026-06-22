#!/usr/bin/env python
"""Backward-compatible CLI for two-stage damage segmentation training."""

from two_stage.cli import parse_args
from two_stage.data import (
    TwoStageDamageDataset,
    crop_sample,
    expand_box,
    load_classes,
    load_npz_mask,
    make_gt_roi_boxes,
    make_pred_roi_boxes,
    make_roi_target_and_valid,
    mask_to_box,
    parse_class_ids,
    predicted_canvas_box_to_original,
    preprocess_roi_image,
    read_rows,
    resize_mask,
    row_key,
)
from two_stage.pipeline import main

__all__ = [
    "TwoStageDamageDataset",
    "crop_sample",
    "expand_box",
    "load_classes",
    "load_npz_mask",
    "main",
    "make_gt_roi_boxes",
    "make_pred_roi_boxes",
    "make_roi_target_and_valid",
    "mask_to_box",
    "parse_args",
    "parse_class_ids",
    "predicted_canvas_box_to_original",
    "preprocess_roi_image",
    "read_rows",
    "resize_mask",
    "row_key",
]


if __name__ == "__main__":
    main()
