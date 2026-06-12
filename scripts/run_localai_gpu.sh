#!/usr/bin/env bash
set -euo pipefail

LOCALAI_PYTHON="${LOCALAI_PYTHON:-/home/useradmin/miniconda3/envs/localai/bin/python}"

PYTHONPATH=src "$LOCALAI_PYTHON" -m veg_multigpu.train \
  --epochs 5 \
  --batch-size 16 \
  --image-size 128 \
  --train-samples 2048 \
  --val-samples 512 \
  --base-channels 24 \
  --num-workers 2 \
  --amp \
  --device cuda \
  --out-dir artifacts/runs/localai_single_gpu

