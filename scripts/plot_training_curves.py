from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from a metrics.json artifact.")
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/runs/vepl_localai_full/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/vepl_training_curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    history = payload["history"]
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val loss")
    axes[0].set_title("Optimization")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["val_iou"] for row in history], marker="o", label="IoU")
    axes[1].plot(epochs, [row["val_f1"] for row in history], marker="o", label="F1")
    axes[1].set_title("Segmentation Quality")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
