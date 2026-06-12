#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/vepl/TESELLATED_WITHOUT_AUGMENTATION}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m veg_multigpu.train \
  --dataset vepl \
  --data-dir "$DATA_DIR" \
  --vepl-target foreground \
  --epochs 24 \
  --batch-size 8 \
  --image-size 192 \
  --train-samples 426 \
  --val-samples 106 \
  --base-channels 32 \
  --num-workers 2 \
  --amp \
  --device cuda \
  --out-dir artifacts/runs/kaggle_2gpu
