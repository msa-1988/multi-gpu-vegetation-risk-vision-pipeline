# Verification Notes

Verified locally on 2026-06-12.

## Environment

- Local machine has one NVIDIA RTX 5070 Laptop GPU.
- Base Python's PyTorch build sees CUDA but does not include `sm_120` kernels, so it falls back to CPU.
- The `localai` conda environment has PyTorch `2.10.0+cu128` with `sm_120` support and runs successfully on the local RTX 5070 GPU.
- Multi-GPU CUDA verification should be run on a compatible cloud runtime such as Kaggle T4x2.

## Commands Run

```bash
/home/useradmin/miniconda3/envs/localai/bin/python scripts/check_localai_cuda.py
```

```bash
PYTHONPATH=src /home/useradmin/miniconda3/envs/localai/bin/python -m veg_multigpu.train \
  --epochs 1 \
  --batch-size 4 \
  --image-size 64 \
  --train-samples 32 \
  --val-samples 16 \
  --base-channels 8 \
  --num-workers 0 \
  --device cuda \
  --out-dir artifacts/runs/localai_cuda_smoke
```

```bash
bash scripts/run_localai_gpu.sh
```

```bash
PYTHONPATH=src /home/useradmin/miniconda3/envs/localai/bin/torchrun --standalone --nproc_per_node=2 -m veg_multigpu.train \
  --epochs 1 \
  --batch-size 2 \
  --image-size 48 \
  --train-samples 16 \
  --val-samples 8 \
  --base-channels 8 \
  --num-workers 0 \
  --device cpu \
  --out-dir artifacts/runs/localai_ddp_cpu_smoke
```

```bash
PYTHONPATH=src /home/useradmin/miniconda3/envs/localai/bin/python scripts/visualize_predictions.py \
  --checkpoint artifacts/runs/localai_single_gpu/model.pt \
  --output artifacts/figures/localai_predictions.png \
  --device cuda
```

```bash
PYTHONPATH=src /home/useradmin/miniconda3/envs/localai/bin/python -m compileall -q src
PYTHONPATH=src /home/useradmin/miniconda3/envs/localai/bin/python scripts/compare_runs.py
```

## Local Results

| run | world_size | last val_iou | last val_f1 | images/sec |
|---|---:|---:|---:|---:|
| localai_cuda_smoke | 1 | 0.0524 | 0.0994 | 68.4 |
| localai_single_gpu | 1 | 0.8382 | 0.9120 | 758.4 |
| localai_ddp_cpu_smoke | 2 | 0.0457 | 0.0868 | 117.5 |
| vepl_localai_smoke | 1 | 0.6468 | 0.7849 | 507.8 |
| vepl_localai_full | 1 | 0.7290 | 0.8392 | 249.1 |

The single-GPU run is the main local quality run. The DDP CPU run is a sanity test for process groups, distributed samplers, rank-aware metric reduction, artifact writing, and run comparison. True multi-GPU CUDA speedup must be verified on a cloud runtime with at least two GPUs.

## Visual Artifact

Generated:

`artifacts/figures/localai_predictions.png`

The figure displays vegetation intensity, powerline corridor signal, ground-truth risk, predicted probability, and thresholded prediction mask.

## Real UAV Dataset Smoke Test

Dataset:

```text
VEPL: https://doi.org/10.5281/zenodo.7800234
```

The compact `TESELLATED_WITHOUT_AUGMENTATION.zip` archive was downloaded and extracted under `data/vepl/`. The loader found 532 image/mask pairs. Masks are decoded by class color: black background, green vegetation, gray powerline/corridor pixels.

Command:

```bash
bash scripts/run_vepl_localai_smoke.sh
```

Verified result:

```text
device: cuda:0
world_size: 1
val_iou: 0.6468
val_f1: 0.7849
images_per_sec: 507.8
```

Visual outputs:

```text
artifacts/figures/vepl_real_samples.png
artifacts/figures/vepl_predictions.png
```

## Representative VEPL Run

Command:

```bash
bash scripts/run_vepl_localai_full.sh
```

Configuration:

```text
dataset: VEPL compact non-augmented tiles
samples: 426 train / 106 validation
image_size: 192
base_channels: 32
epochs: 15
device: cuda:0
```

Verified result:

```text
best_val_iou: 0.7299 at epoch 11
final_val_iou: 0.7290
final_val_f1: 0.8392
final_val_accuracy: 0.8270
images_per_sec: 249.1
total_train_seconds: 30.48
```

Commit-ready visual outputs:

```text
docs/assets/vepl_real_samples.png
docs/assets/vepl_predictions_full.png
docs/assets/vepl_training_curves.png
```
