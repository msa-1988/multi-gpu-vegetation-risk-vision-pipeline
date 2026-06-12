from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import random_split
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from veg_multigpu.data import SyntheticVegetationRiskDataset, VEPLDataset
from veg_multigpu.metrics import average_dicts, segmentation_stats
from veg_multigpu.model import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a vegetation-risk segmentation model with optional DDP.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--dataset", choices=["synthetic", "vepl"], default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=Path("data/vepl/TESELLATED_WITHOUT_AUGMENTATION"))
    parser.add_argument("--vepl-target", choices=["foreground", "vegetation", "powerline"], default="foreground")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/runs/local"))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def json_ready_args(args: argparse.Namespace) -> dict[str, object]:
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def cuda_build_supports_current_gpu(local_rank: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    if local_rank >= torch.cuda.device_count():
        return False
    capability = torch.cuda.get_device_capability(local_rank)
    sm = f"sm_{capability[0]}{capability[1]}"
    arch_list = torch.cuda.get_arch_list()
    return sm in arch_list or not arch_list


def select_device(requested: str, local_rank: int, is_main: bool) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        if cuda_build_supports_current_gpu(local_rank):
            return torch.device(f"cuda:{local_rank}")
        message = (
            "CUDA is visible, but this PyTorch build does not include kernels for "
            f"{torch.cuda.get_device_name(local_rank)} capability {torch.cuda.get_device_capability(local_rank)}. "
            "Falling back to CPU. Install a matching PyTorch CUDA build to use this local GPU."
        )
        if requested == "cuda":
            raise RuntimeError(message)
        if is_main:
            print(f"[device] {message}")
    return torch.device("cpu")


def setup_distributed(requested_device: str) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        use_cuda_backend = requested_device != "cpu" and torch.cuda.is_available() and cuda_build_supports_current_gpu(local_rank)
        backend = "nccl" if use_cuda_backend else "gloo"
        dist.init_process_group(backend=backend)
        if use_cuda_backend:
            torch.cuda.set_device(local_rank)

    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def make_loaders(
    args: argparse.Namespace,
    distributed: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    if args.dataset == "synthetic":
        train_dataset = SyntheticVegetationRiskDataset(args.train_samples, args.image_size, seed=args.seed)
        val_dataset = SyntheticVegetationRiskDataset(args.val_samples, args.image_size, seed=args.seed + 50_000)
    else:
        # VEPL is split deterministically so metrics are reproducible across
        # local, Kaggle, and Colab runs.
        total_requested = args.train_samples + args.val_samples
        dataset = VEPLDataset(args.data_dir, args.image_size, limit=total_requested, target=args.vepl_target)
        if len(dataset) < 2:
            raise RuntimeError(f"VEPL dataset at {args.data_dir} needs at least 2 image/mask pairs.")
        val_count = min(args.val_samples, max(1, len(dataset) // 5))
        train_count = len(dataset) - val_count
        train_dataset, val_dataset = random_split(
            dataset,
            [train_count, val_count],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, distributed: bool) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    losses: list[float] = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            losses.append(float(loss.item()))
            rows.append(segmentation_stats(logits, masks))

    metrics = average_dicts(rows)
    metrics["loss"] = sum(losses) / max(len(losses), 1)

    for key, value in list(metrics.items()):
        # Average validation metrics across DDP workers so rank 0 logs one
        # consistent view of the whole validation split.
        tensor = torch.tensor(value, device=device)
        metrics[key] = float(reduce_mean(tensor, distributed).item())
    return metrics


def train() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size = setup_distributed(args.device)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    device = select_device(args.device, local_rank, is_main)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(args, distributed, rank, world_size, device)
    model = TinyUNet(base_channels=args.base_channels).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    history: list[dict[str, float | int]] = []
    global_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            # DDP requires epoch-specific shuffling so every worker sees a new
            # shard order while still covering the full dataset.
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        seen = 0
        start = time.perf_counter()
        iterator = tqdm(train_loader, disable=not is_main, desc=f"epoch {epoch}/{args.epochs}")

        for images, masks in iterator:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            # Autocast keeps convolutions fast on CUDA while GradScaler protects
            # small gradients from underflow during mixed-precision training.
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch = images.shape[0]
            epoch_loss += float(loss.item()) * batch
            seen += batch
            if is_main:
                iterator.set_postfix(loss=epoch_loss / max(seen, 1))

        elapsed = time.perf_counter() - start
        local_images_per_sec = torch.tensor(seen / max(elapsed, 1e-9), device=device)
        # Throughput is summed across workers after reducing each local rate.
        images_per_sec = reduce_mean(local_images_per_sec, distributed).item() * world_size
        train_loss = torch.tensor(epoch_loss / max(seen, 1), device=device)
        train_loss = float(reduce_mean(train_loss, distributed).item())
        val_metrics = evaluate(model, val_loader, device, distributed)

        row = {
            "epoch": epoch,
            "world_size": world_size,
            "train_loss": train_loss,
            "images_per_sec": float(images_per_sec),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        if is_main:
            print(json.dumps(row, indent=2))

    if is_main:
        total_time = time.perf_counter() - global_start
        payload = {
            "args": json_ready_args(args),
            "device": str(device),
            "world_size": world_size,
            "total_train_seconds": total_time,
            "history": history,
        }
        (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
        torch.save(model_to_save.state_dict(), args.out_dir / "model.pt")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    train()
