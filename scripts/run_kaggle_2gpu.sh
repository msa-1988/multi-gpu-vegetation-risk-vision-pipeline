#!/usr/bin/env bash
set -euo pipefail

torchrun --standalone --nproc_per_node=2 -m veg_multigpu.train \
  --epochs 5 \
  --batch-size 16 \
  --image-size 128 \
  --train-samples 4096 \
  --val-samples 1024 \
  --base-channels 24 \
  --num-workers 2 \
  --amp \
  --out-dir artifacts/runs/kaggle_2gpu

