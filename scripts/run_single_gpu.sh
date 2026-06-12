#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m veg_multigpu.train \
  --epochs 5 \
  --batch-size 16 \
  --image-size 128 \
  --train-samples 2048 \
  --val-samples 512 \
  --base-channels 24 \
  --num-workers 2 \
  --amp \
  --device auto \
  --out-dir artifacts/runs/single_gpu
