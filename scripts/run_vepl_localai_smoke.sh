#!/usr/bin/env bash
set -euo pipefail

LOCALAI_PYTHON="${LOCALAI_PYTHON:-/home/useradmin/miniconda3/envs/localai/bin/python}"
DATA_DIR="${DATA_DIR:-data/vepl/TESELLATED_WITHOUT_AUGMENTATION}"

PYTHONPATH=src "$LOCALAI_PYTHON" -m veg_multigpu.train \
  --dataset vepl \
  --data-dir "$DATA_DIR" \
  --vepl-target foreground \
  --epochs 3 \
  --batch-size 8 \
  --image-size 128 \
  --train-samples 160 \
  --val-samples 40 \
  --base-channels 24 \
  --num-workers 2 \
  --amp \
  --device cuda \
  --out-dir artifacts/runs/vepl_localai_smoke
