#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m veg_multigpu.train \
  --epochs 1 \
  --batch-size 4 \
  --image-size 64 \
  --train-samples 32 \
  --val-samples 16 \
  --base-channels 8 \
  --num-workers 0 \
  --device auto \
  --out-dir artifacts/runs/smoke
