#!/bin/bash
#SBATCH --job-name=damage-stage2-fine
#SBATCH --output=/scratch/user/zuoxu/transformers/scripts/log/%x-%j.out
#SBATCH --error=/scratch/user/zuoxu/transformers/scripts/log/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --account=132736432922

set -euo pipefail

echo "[info] job id: ${SLURM_JOB_ID:-local}"
echo "[info] node: $(hostname)"
echo "[info] start time: $(date)"

module purge
module load Anaconda3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/user/zuoxu/conda_envs/crackseg

export PATH="/scratch/user/zuoxu/conda_envs/crackseg/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/scratch/user/zuoxu/datasets/unified_damage_data}"
STAGE1_MODEL_DIR="${STAGE1_MODEL_DIR:-$REPO_DIR/runs/stage1_roi_b0_best/stage1_roi}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/runs/stage2_fine_b3_best}"
STAGE2_MODEL="${STAGE2_MODEL:-nvidia/segformer-b3-finetuned-ade-512-512}"

HF_HOME="${HF_HOME:-/scratch/user/zuoxu/hf_cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

WANDB_PROJECT="${WANDB_PROJECT:-damage-two-stage-segformer}"
WANDB_NAME="${WANDB_NAME:-stage2-fine-b3-${SLURM_JOB_ID:-local}}"
WANDB_DIR="${WANDB_DIR:-/scratch/user/zuoxu/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_DIR/config}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WEBPROXY="${USE_WEBPROXY:-1}"

mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
mkdir -p /scratch/user/zuoxu/transformers/scripts/log

if [[ "$USE_WEBPROXY" == "1" ]]; then
  if module load WebProxy 2>/dev/null; then
    echo "[info] loaded WebProxy module"
  elif module load webproxy 2>/dev/null; then
    echo "[info] loaded webproxy module"
  else
    echo "[error] USE_WEBPROXY=1 but no WebProxy/webproxy module could be loaded" >&2
    exit 1
  fi
fi

export HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT WANDB_NAME WANDB_DIR WANDB_CACHE_DIR WANDB_CONFIG_DIR WANDB_MODE

echo "[info] REPO_DIR: $REPO_DIR"
echo "[info] DATA_ROOT: $DATA_ROOT"
echo "[info] STAGE1_MODEL_DIR: $STAGE1_MODEL_DIR"
echo "[info] OUTPUT_DIR: $OUTPUT_DIR"
echo "[info] STAGE2_MODEL: $STAGE2_MODEL"
echo "[info] WANDB_MODE: $WANDB_MODE"

if [[ ! -f "$REPO_DIR/scripts/train_two_stage_damage_segformer.py" ]]; then
  echo "[error] missing training script: $REPO_DIR/scripts/train_two_stage_damage_segformer.py" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[error] DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -d "$STAGE1_MODEL_DIR" ]]; then
  echo "[error] STAGE1_MODEL_DIR does not exist: $STAGE1_MODEL_DIR" >&2
  echo "[hint] Run scripts/train_damage_stage1_roi_hprc.slurm first." >&2
  exit 1
fi

python - <<'PY'
import torch
print("[check] torch:", torch.__version__)
print("[check] cuda available:", torch.cuda.is_available())
print("[check] gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("[check] gpu:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("CUDA is not available inside this GPU job.")
PY

cd "$REPO_DIR"

python scripts/train_two_stage_damage_segformer.py \
  --stage fine \
  --fine-roi-source pred \
  --stage1-model-dir "$STAGE1_MODEL_DIR" \
  --data-root "$DATA_ROOT" \
  --stage2-model-name-or-path "$STAGE2_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --image-size 512 \
  --resize-mode letterbox \
  --roi-positive-class-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,18 \
  --roi-negative-valid-policy all \
  --roi-crop-padding 0.25 \
  --batch-size 2 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --num-train-epochs 20 \
  --eval-strategy epoch \
  --save-strategy epoch \
  --save-total-limit 2 \
  --num-workers 4 \
  --eval-accumulation-steps 1 \
  --report-to wandb \
  --run-name "$WANDB_NAME" \
  --fp16

echo "[info] finish time: $(date)"
