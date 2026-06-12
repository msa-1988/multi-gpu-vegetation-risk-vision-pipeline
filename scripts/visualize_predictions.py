from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from veg_multigpu.data import SyntheticVegetationRiskDataset
from veg_multigpu.model import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a qualitative vegetation-risk prediction grid.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/runs/localai_single_gpu/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/localai_predictions.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    model = TinyUNet(base_channels=args.base_channels).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    dataset = SyntheticVegetationRiskDataset(args.samples, image_size=args.image_size, seed=args.seed)
    rows = args.samples
    fig, axes = plt.subplots(rows, 5, figsize=(14, 3.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for idx in range(rows):
            image, mask = dataset[idx]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).squeeze().cpu()
            pred = (probs >= 0.5).float()

            panels = [
                ("vegetation", image[0]),
                ("powerline corridor", image[1]),
                ("ground truth risk", mask.squeeze()),
                ("predicted probability", probs),
                ("predicted mask", pred),
            ]
            for col, (title, arr) in enumerate(panels):
                ax = axes[idx, col]
                ax.imshow(arr, cmap="viridis", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

