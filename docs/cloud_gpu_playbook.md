# Cloud GPU Playbook

## What This Project Demonstrates

This project is designed to run in three modes:

1. **Local smoke test** on CPU or one GPU.
2. **Single-GPU training** on the local RTX GPU.
3. **DistributedDataParallel training** on a cloud notebook with 2 GPUs, for example Kaggle T4x2 when available.

The code uses standard PyTorch `torchrun` and `DistributedDataParallel`, so the same training entry point works locally and in cloud notebooks.

## Kaggle 2-GPU Run

Kaggle often provides GPU notebook options such as P100 or T4x2 depending on availability and quota.

In a Kaggle notebook:

```bash
git clone <your-repo-url>
cd "07 - Multi-GPU Vegetation Risk Vision Pipeline"
pip install -r requirements.txt
export PYTHONPATH=src
bash scripts/run_kaggle_2gpu.sh
python scripts/compare_runs.py
```

Expected difference vs local:

- `world_size` should be `2`.
- `device` should be `cuda:0` in `metrics.json` from rank 0.
- `images/sec` should be higher than the local 1-GPU run if data loading is not the bottleneck.
- Model quality metrics may differ slightly because DDP changes batch ordering and effective optimization dynamics.

## Google Colab

Colab free GPU access is useful for single-GPU experimentation but is not guaranteed and typically does not expose multiple GPUs in the free tier. Use it for:

- quick single-GPU checks,
- AMP behavior,
- notebook demos,
- profiling smaller runs.

Open the prepared notebook:

```text
https://colab.research.google.com/github/msa-1988/multi-gpu-vegetation-risk-vision-pipeline/blob/main/notebooks/colab_vepl_single_gpu.ipynb
```

The notebook clones the repo, checks CUDA, downloads VEPL, runs a fast 8-epoch real-data experiment, and can optionally run the 40-epoch README-style benchmark.

Raw Colab single-GPU command:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m veg_multigpu.train \
  --epochs 5 \
  --batch-size 16 \
  --image-size 128 \
  --train-samples 2048 \
  --val-samples 512 \
  --base-channels 24 \
  --num-workers 2 \
  --amp \
  --device cuda \
  --out-dir artifacts/runs/colab_single_gpu
```

## Lightning AI Studio

Lightning AI Studio is the most IDE-like option. Use it when you want a persistent cloud workspace with GPU hours and a familiar development flow.

## Why Not Combine Local + Cloud GPUs?

Combining your local GPU with a cloud GPU into one training job would require distributed networking, stable public/private routing, synchronized environments, security configuration, and high-bandwidth low-latency communication. For a portfolio project, it is cleaner and more credible to show:

- local 1-GPU training,
- cloud 2-GPU DDP training,
- reproducible commands,
- scaling metrics and limitations.
