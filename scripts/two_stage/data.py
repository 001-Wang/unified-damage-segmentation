"""Dataset loading, preprocessing, and ROI geometry."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


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

        try:
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

            if target.shape[-1] <= 0 or target.shape[-2] <= 0:
                raise RuntimeError(f"Invalid target shape before resize: {tuple(target.shape)}")
            if valid_mask.shape[-1] <= 0 or valid_mask.shape[-2] <= 0:
                raise RuntimeError(f"Invalid valid_mask shape before resize: {tuple(valid_mask.shape)}")

            if self.resize_mode == "stretch":
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                target = resize_mask(target, self.image_size, self.image_size)
                valid_mask = resize_mask(valid_mask, self.image_size, self.image_size)
            else:
                image, target, valid_mask = self.letterbox(image, target, valid_mask)
        except Exception as e:
            raise RuntimeError(
                f"Failed sample index={index}, key={row_key(row)}, "
                f"dataset={row['dataset']}, split={row['split']}, id={row['id']}, "
                f"image_path={row['image_path']}, target_path={row['target_path']}, "
                f"valid_mask_path={row['valid_mask_path']}, "
                f"image_size={image.size}, target_shape={tuple(target.shape)}, "
                f"valid_shape={tuple(valid_mask.shape)}, "
                f"box={self.crop_boxes.get(row_key(row)) if self.crop_boxes else None}"
            ) from e

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
    if mask.shape[-2] <= 0 or mask.shape[-1] <= 0:
        raise ValueError(f"Cannot resize empty mask with shape {tuple(mask.shape)} to {(height, width)}")
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid resize target size: {(height, width)}")
    return F.interpolate(mask.unsqueeze(0), size=(height, width), mode="nearest").squeeze(0)


def row_key(row: dict[str, str]) -> str:
    return f"{row['dataset']}:{row['split']}:{row['id']}"


def crop_sample(
    image: Image.Image, target: torch.Tensor, valid_mask: torch.Tensor, box: list[int]
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Crop image/masks safely.

    The ROI box is produced in original image coordinates. Some datasets may have
    masks whose spatial size differs from the image size, so directly slicing the
    mask with image coordinates can create zero-width/zero-height tensors. This
    function clamps the image crop and converts it into mask coordinates before
    slicing masks.
    """
    img_w, img_h = image.size
    mask_h, mask_w = target.shape[-2:]

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {(img_w, img_h)}")
    if mask_w <= 0 or mask_h <= 0:
        raise ValueError(f"Invalid target mask shape: {tuple(target.shape)}")
    if valid_mask.shape[-2:] != target.shape[-2:]:
        raise ValueError(
            f"Target/valid mask size mismatch: "
            f"target={tuple(target.shape)}, valid_mask={tuple(valid_mask.shape)}"
        )

    left, top, right, bottom = [float(v) for v in box]

    # Clamp crop in image coordinates. Keep at least one pixel.
    left = max(0.0, min(left, img_w - 1))
    top = max(0.0, min(top, img_h - 1))
    right = max(left + 1.0, min(right, img_w))
    bottom = max(top + 1.0, min(bottom, img_h))

    i_left = int(math.floor(left))
    i_top = int(math.floor(top))
    i_right = int(math.ceil(right))
    i_bottom = int(math.ceil(bottom))

    i_left = max(0, min(i_left, img_w - 1))
    i_top = max(0, min(i_top, img_h - 1))
    i_right = max(i_left + 1, min(i_right, img_w))
    i_bottom = max(i_top + 1, min(i_bottom, img_h))

    # Convert image-coordinate box to mask-coordinate box.
    m_left = int(math.floor(i_left * mask_w / img_w))
    m_top = int(math.floor(i_top * mask_h / img_h))
    m_right = int(math.ceil(i_right * mask_w / img_w))
    m_bottom = int(math.ceil(i_bottom * mask_h / img_h))

    # Clamp crop in mask coordinates. Keep at least one pixel.
    m_left = max(0, min(m_left, mask_w - 1))
    m_top = max(0, min(m_top, mask_h - 1))
    m_right = max(m_left + 1, min(m_right, mask_w))
    m_bottom = max(m_top + 1, min(m_bottom, mask_h))

    image_crop = image.crop((i_left, i_top, i_right, i_bottom))
    target_crop = target[:, m_top:m_bottom, m_left:m_right]
    valid_crop = valid_mask[:, m_top:m_bottom, m_left:m_right]

    if target_crop.shape[-1] <= 0 or target_crop.shape[-2] <= 0:
        raise RuntimeError(
            f"Invalid crop: image_size={(img_w, img_h)}, "
            f"mask_size={(mask_w, mask_h)}, original_box={box}, "
            f"image_box={(i_left, i_top, i_right, i_bottom)}, "
            f"mask_box={(m_left, m_top, m_right, m_bottom)}, "
            f"target_crop_shape={tuple(target_crop.shape)}"
        )

    return image_crop, target_crop, valid_crop


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
