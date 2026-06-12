from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two convolutional layers with normalization and smooth activations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Compact U-Net for vegetation-risk segmentation.

    The encoder compresses RGB/UAV context into increasingly semantic feature
    maps. The decoder upsamples those features and uses skip connections to
    recover fine spatial boundaries, which matters for thin powerline corridors
    and vegetation edges.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: preserve one high-resolution feature map for each decoder skip.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bottleneck = self.enc3(self.pool(e2))

        # Decoder: concatenate coarse semantic context with local edge detail.
        d2 = self.up2(bottleneck)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Raw logits are returned so BCEWithLogitsLoss can apply a stable sigmoid.
        return self.head(d1)
