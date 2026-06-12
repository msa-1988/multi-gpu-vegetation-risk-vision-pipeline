from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class SyntheticVegetationRiskDataset(Dataset):
    """Deterministic satellite-like vegetation risk segmentation dataset.

    Each sample contains three channels:
    1. vegetation intensity,
    2. distance-to-powerline signal,
    3. texture/noise signal.

    The target mask marks vegetation that is close enough to a synthetic
    powerline corridor to represent an operational trimming/fire-risk area.
    """

    def __init__(self, samples: int, image_size: int = 128, seed: int = 13) -> None:
        self.samples = samples
        self.image_size = image_size
        self.seed = seed

        axis = torch.linspace(-1.0, 1.0, image_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.xx = xx
        self.yy = yy

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + idx)
        image_size = self.image_size
        xx = self.xx
        yy = self.yy

        slope = (torch.rand((), generator=g) * 1.10 - 0.55).item()
        intercept = (torch.rand((), generator=g) * 0.50 - 0.25).item()
        line_distance = torch.abs(yy - slope * xx - intercept) / math.sqrt(1.0 + slope * slope)
        corridor = torch.exp(-line_distance * 9.0)

        vegetation = torch.zeros((image_size, image_size), dtype=torch.float32)
        blob_count = int(torch.randint(7, 16, (1,), generator=g).item())
        for _ in range(blob_count):
            cx = (torch.rand((), generator=g) * 1.90 - 0.95).item()
            cy = (torch.rand((), generator=g) * 1.90 - 0.95).item()
            radius = (torch.rand((), generator=g) * 0.13 + 0.05).item()
            strength = (torch.rand((), generator=g) * 0.70 + 0.45).item()
            blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius * radius))
            vegetation = torch.maximum(vegetation, strength * blob)

        noise = torch.rand((image_size, image_size), generator=g) * 0.18
        terrain = 0.5 + 0.25 * torch.sin(7 * xx + 3 * yy) + 0.25 * torch.cos(5 * yy)
        vegetation = torch.clamp(vegetation + noise + 0.12 * terrain, 0.0, 1.0)

        risk = ((vegetation > 0.48) & (line_distance < 0.16)).float()
        weak_risk = ((vegetation > 0.62) & (line_distance < 0.23)).float()
        mask = torch.maximum(risk, 0.65 * weak_risk).unsqueeze(0)

        image = torch.stack(
            [
                vegetation,
                corridor,
                torch.clamp(terrain + noise, 0.0, 1.0),
            ],
            dim=0,
        ).float()

        return image, mask.float()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _pair_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("mask", "masks", "label", "labels", "image", "images", "rgb"):
        stem = stem.replace(token, "")
    return re.sub(r"[^a-z0-9]+", "", stem)


def _is_mask_path(path: Path) -> bool:
    text = "/".join(part.lower() for part in path.parts)
    return any(token in text for token in ("mask", "masks", "label", "labels"))


def find_image_mask_pairs(root: Path) -> list[tuple[Path, Path]]:
    files = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    images = [path for path in files if not _is_mask_path(path)]
    masks = [path for path in files if _is_mask_path(path)]
    masks_by_key: dict[str, Path] = {_pair_key(path): path for path in masks}

    pairs: list[tuple[Path, Path]] = []
    for image in images:
        mask = masks_by_key.get(_pair_key(image))
        if mask is not None:
            pairs.append((image, mask))

    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under {root}. Expected folders or filenames containing 'image' and 'mask'."
        )
    return pairs


def vepl_mask_to_binary(mask_rgb: torch.Tensor, target: str) -> torch.Tensor:
    red = mask_rgb[0].to(torch.int16)
    green = mask_rgb[1].to(torch.int16)
    blue = mask_rgb[2].to(torch.int16)

    vegetation = (green > 180) & (red < 80) & (blue < 80)
    powerline = (
        (torch.abs(red - green) < 20)
        & (torch.abs(green - blue) < 20)
        & (red > 70)
        & (red < 160)
    )

    if target == "vegetation":
        mask = vegetation
    elif target == "powerline":
        mask = powerline
    elif target == "foreground":
        mask = vegetation | powerline
    else:
        raise ValueError(f"Unsupported VEPL target: {target}")
    return mask.float()


class VEPLDataset(Dataset):
    """VEPL UAV vegetation-encroachment segmentation dataset adapter.

    The public VEPL masks are multi-color semantic masks:
    black background, green vegetation, and gray powerline corridor pixels.
    This adapter maps those colors into a binary target.
    """

    def __init__(
        self,
        root: Path | str,
        image_size: int = 128,
        limit: int | None = None,
        target: str = "foreground",
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.target = target
        pairs = find_image_mask_pairs(self.root)
        self.pairs = pairs[:limit] if limit is not None else pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        image = read_image(str(image_path), mode=ImageReadMode.RGB).float() / 255.0
        mask_rgb = read_image(str(mask_path), mode=ImageReadMode.RGB)

        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_rgb = F.interpolate(
            mask_rgb.float().unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0).to(torch.uint8)

        mask = vepl_mask_to_binary(mask_rgb, self.target)
        return image.float(), mask.unsqueeze(0)
