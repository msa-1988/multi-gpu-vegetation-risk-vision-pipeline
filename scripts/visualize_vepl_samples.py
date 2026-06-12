from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from veg_multigpu.data import VEPLDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a grid of VEPL real UAV image/mask samples.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/vepl/TESELLATED_WITHOUT_AUGMENTATION"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/vepl_real_samples.png"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--vepl-target", choices=["foreground", "vegetation", "powerline"], default="foreground")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = VEPLDataset(args.data_dir, image_size=args.image_size, limit=args.samples, target=args.vepl_target)
    rows = min(args.samples, len(dataset))
    fig, axes = plt.subplots(rows, 2, figsize=(8, 3.5 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    for idx in range(rows):
        image, mask = dataset[idx]
        axes[idx, 0].imshow(image.permute(1, 2, 0))
        axes[idx, 0].set_title("real UAV RGB tile")
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(mask.squeeze(), cmap="viridis", vmin=0, vmax=1)
        axes[idx, 1].set_title("VEPL foreground mask")
        axes[idx, 1].axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
