# Unified Multi-Damage Segmentation

Tools for merging public structural-damage datasets into a shared multi-label
format and training SegFormer models for damage segmentation.

This repository provides:

- a 20-class unified label space for DACL10K, HRCDS, S2DS, and LCW;
- validity masks for datasets that annotate only a subset of the classes;
- a direct multi-label SegFormer baseline;
- a two-stage ROI-guided SegFormer pipeline;
- evaluation, inference, visualization, and SLURM scripts.

Datasets and model checkpoints are not included. Review the license and
citation requirements of every source dataset before using or redistributing
it.

## Network

### Multi-label baseline

The baseline uses `AutoModelForSemanticSegmentation` with a SegFormer backbone
and one sigmoid output channel per damage class. Its loss is masked binary
cross-entropy plus masked Dice loss. Per-class validity masks prevent missing
annotations in one source dataset from being treated as negative labels.

### Two-stage model

```text
input image
    |
    v
Stage 1: binary ROI SegFormer (BCE + Tversky loss)
    |
    +--> predicted damage region and padded crop
             |
             v
Stage 2: 20-channel SegFormer (BCE + Dice loss)
             |
             v
        multi-label damage masks
```

Stage 1 combines selected damage classes into a binary region of interest.
Stage 2 trains on the full image or on crops derived from Stage 1 predictions
or ground truth. Different SegFormer sizes can be used for each stage, such as
B0 for ROI detection and B3 for fine segmentation.

## Repository Layout

```text
scripts/datasets/                         dataset inventory and conversion
scripts/train_multilabel_damage_segformer.py
scripts/train_two_stage_damage_segformer.py  two-stage CLI and compatibility imports
scripts/two_stage/                           two-stage training implementation
scripts/eval_stage1_roi_by_dataset.py
scripts/infer_two_stage_damage_segformer.py
scripts/visualize_stage1_roi_predictions.py
slurm/                                    HPRC training launchers
```

The two-stage implementation is organized by responsibility: `cli.py` defines
arguments, `data.py` handles datasets and ROI geometry, `trainers.py` defines
losses and custom trainers, `evaluation.py` computes metrics, `modeling.py`
configures models and checkpoints, and `pipeline.py` coordinates both stages.

## Environment Setup

```bash
git clone https://github.com/001-Wang/unified-damage-segmentation.git
cd unified-damage-segmentation

conda create -n damage-seg python=3.11 -y
conda activate damage-seg
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the appropriate CUDA-enabled PyTorch build for your system if the build
installed by `pip` does not detect your GPU. Check the environment with:

```bash
python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('cuda:', torch.cuda.is_available())"
```

Optional experiment tracking:

```bash
python -m pip install wandb tensorboard
wandb login
```

## Model Downloads and Network Access

Training is cache-only by default. Add `--allow-download` the first time a
Hugging Face checkpoint is used on a machine with internet access. On an
offline compute node, download the model on a login node first and use a shared
cache:

```bash
export HF_HOME=/path/to/shared/huggingface-cache
```

Then omit `--allow-download` during the offline job.

## Dataset Preparation

Place the downloaded datasets under one data root. The conversion code
recognizes these directory names:

```text
damage_data/
├── dacl10k_v2_devphase/
├── HRCDS/
├── s2ds/
└── LCW Concrete Crack Detection/
```

Set the data root and inspect the available datasets:

```bash
export DAMAGE_DATA_ROOT=/path/to/damage_data
python scripts/datasets/00_inventory.py
```

Build a small dataset for validation:

```bash
python scripts/datasets/04_merge_multilabel_damage_dataset.py \
  --output-root "$DAMAGE_DATA_ROOT/unified_damage_data_smoke" \
  --datasets dacl s2ds hrcds \
  --image-mode symlink \
  --limit-per-dataset 20
```

Build the complete unified dataset and optionally add LCW:

```bash
python scripts/datasets/04_merge_multilabel_damage_dataset.py \
  --output-root "$DAMAGE_DATA_ROOT/unified_damage_data" \
  --datasets dacl s2ds hrcds \
  --image-mode symlink

python scripts/datasets/05_add_lcw_to_unified_damage_data.py \
  --output-root "$DAMAGE_DATA_ROOT/unified_damage_data"
```

The generated directory contains:

```text
unified_damage_data/
├── classes.json
├── train.csv
├── val.csv
├── test.csv
├── images/
├── masks_multilabel/
└── valid_masks/
```

The 20 unified classes are crack, alligator crack, wet spot, efflorescence,
rust/corrosion, rock pocket, hollow area, cavity, spalling, graffiti,
weathering, rest formwork, exposed rebar, bearing, expansion joint, drainage,
protective equipment, joint tape, washout concrete corrosion, and vegetation.

The separate binary crack workflow is documented in
[`docs/dataset_unification.md`](docs/dataset_unification.md).

## Training

### Direct multi-label baseline

```bash
python scripts/train_multilabel_damage_segformer.py \
  --data-root "$DAMAGE_DATA_ROOT/unified_damage_data" \
  --model-name-or-path nvidia/segformer-b3-finetuned-ade-512-512 \
  --output-dir runs/multilabel_b3 \
  --image-size 512 \
  --resize-mode letterbox \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 10 \
  --fp16 \
  --allow-download
```

### Two-stage training

```bash
python scripts/train_two_stage_damage_segformer.py \
  --stage both \
  --data-root "$DAMAGE_DATA_ROOT/unified_damage_data" \
  --stage1-model-name-or-path nvidia/segformer-b0-finetuned-ade-512-512 \
  --stage2-model-name-or-path nvidia/segformer-b3-finetuned-ade-512-512 \
  --output-dir runs/two_stage_b0_b3 \
  --fine-roi-source pred \
  --roi-crop-padding 0.25 \
  --image-size 512 \
  --resize-mode letterbox \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 10 \
  --fp16 \
  --allow-download
```

Models are saved under:

```text
runs/two_stage_b0_b3/stage1_roi/
runs/two_stage_b0_b3/stage2_fine/
```

Use `--report-to wandb --run-name NAME` for Weights & Biases or
`--report-to tensorboard` for TensorBoard. Run either training script with
`--help` for all options.

## Evaluation and Visualization

```bash
python scripts/eval_stage1_roi_by_dataset.py \
  --model-dir runs/two_stage_b0_b3/stage1_roi \
  --data-root "$DAMAGE_DATA_ROOT/unified_damage_data" \
  --split val \
  --output-json runs/stage1_eval.json \
  --output-csv runs/stage1_eval.csv

python scripts/visualize_stage1_roi_predictions.py \
  --model-dir runs/two_stage_b0_b3/stage1_roi \
  --data-root "$DAMAGE_DATA_ROOT/unified_damage_data" \
  --split val \
  --output-dir runs/stage1_visualizations \
  --num-samples 24 \
  --prefer-positive
```

## Inference

```bash
python scripts/infer_two_stage_damage_segformer.py \
  --image /path/to/image.jpg \
  --stage1-model-dir runs/two_stage_b0_b3/stage1_roi \
  --stage2-model-dir runs/two_stage_b0_b3/stage2_fine \
  --classes-json "$DAMAGE_DATA_ROOT/unified_damage_data/classes.json" \
  --output-dir runs/inference/example \
  --roi-crop-padding 0.25 \
  --gate-with-roi
```

The output directory contains one binary PNG per class and the predicted ROI
mask.

## HPRC / SLURM

The launchers in `slurm/` are site-specific templates. Update the `#SBATCH`
account, log paths, Conda environment, cache paths, and dataset paths before
submitting. Run Stage 1 before Stage 2:

```bash
sbatch slurm/train_damage_stage1_roi_hprc.slurm

STAGE1_MODEL_DIR=/path/to/stage1_roi \
DATA_ROOT=/path/to/unified_damage_data \
sbatch slurm/train_damage_stage2_fine_hprc.slurm
```

The launchers support overrides including `REPO_DIR`, `DATA_ROOT`,
`OUTPUT_DIR`, `STAGE1_MODEL_DIR`, `HF_HOME`, `WANDB_MODE`, and `USE_WEBPROXY`.

## Notes

- `letterbox` preserves aspect ratio and ignores padded pixels during loss and
  metrics; `stretch` resizes directly to a square.
- Validity masks are required because the merged datasets do not all annotate
  the same damage classes.
- Generated datasets, checkpoints, logs, and experiment runs are ignored by
  Git and should be stored outside the repository or published separately.
