from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from veg_multigpu.data import VEPLDataset
from veg_multigpu.model import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save VEPL real-image prediction examples.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/vepl/TESELLATED_WITHOUT_AUGMENTATION"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/runs/vepl_localai_smoke/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/figures/vepl_predictions.png"))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--vepl-target", choices=["foreground", "vegetation", "powerline"], default="foreground")
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = VEPLDataset(args.data_dir, image_size=args.image_size, limit=args.samples, target=args.vepl_target)
    rows = min(args.samples, len(dataset))
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.0 * rows), constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for idx in range(rows):
            image, mask = dataset[idx]
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).squeeze().cpu()
            pred = (probs >= 0.5).float()

            panels = [
                ("real UAV RGB", image.permute(1, 2, 0)),
                ("ground truth", mask.squeeze()),
                ("predicted probability", probs),
                ("predicted mask", pred),
            ]
            for col, (title, arr) in enumerate(panels):
                axes[idx, col].imshow(arr, cmap=None if col == 0 else "viridis", vmin=None if col == 0 else 0, vmax=None if col == 0 else 1)
                axes[idx, col].set_title(title)
                axes[idx, col].axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
