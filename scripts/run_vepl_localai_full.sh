#!/usr/bin/env bash
set -euo pipefail

LOCALAI_PYTHON="${LOCALAI_PYTHON:-/home/useradmin/miniconda3/envs/localai/bin/python}"
DATA_DIR="${DATA_DIR:-data/vepl/TESELLATED_WITHOUT_AUGMENTATION}"

PYTHONPATH=src "$LOCALAI_PYTHON" -m veg_multigpu.train \
  --dataset vepl \
  --data-dir "$DATA_DIR" \
  --vepl-target foreground \
  --epochs 15 \
  --batch-size 8 \
  --image-size 192 \
  --train-samples 426 \
  --val-samples 106 \
  --base-channels 32 \
  --num-workers 2 \
  --amp \
  --device cuda \
  --out-dir artifacts/runs/vepl_localai_full
