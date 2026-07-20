"""Reimplemented 4-level U-Net (R4, Section 4).

A plain U-Net matching the architecture description in Buda et al. (2019):
  * 4 encoder levels, each two (conv 3x3 -> [BN] -> ReLU) blocks,
  * 2x2 max-pool downsampling,
  * a bottleneck, then a symmetric decoder with transpose-conv up-sampling and
    skip connections,
  * a 1x1 output conv to ``num_classes`` logits.

Batch norm is optional and ON by default, matching the public reference repo
``mateuszbuda/brain-segmentation-pytorch`` (the paper's own net omitted BN; set
``batchnorm=False`` for the strictly-paper-faithful variant). The model is always
trained from random init — no BraTS-derived weights (R4).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, batchnorm: bool = True):
        super().__init__()
        layers: List[nn.Module] = []
        for a, b in ((in_ch, out_ch), (out_ch, out_ch)):
            layers.append(nn.Conv2d(a, b, kernel_size=3, padding=1, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm2d(b))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """4-level U-Net. ``channels`` are the encoder widths (default 32/64/128/256).

    The public repo uses a base width of 32; the classic U-Net uses 64. Default
    to 32 to match the reference implementation while keeping the model small.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: List[int] = (32, 64, 128, 256),
        batchnorm: bool = True,
    ):
        super().__init__()
        c = list(channels)
        assert len(c) == 4, "this is a 4-level U-Net"
        self.in_channels = in_channels
        self.channels = c

        self.enc1 = ConvBlock(in_channels, c[0], batchnorm)
        self.enc2 = ConvBlock(c[0], c[1], batchnorm)
        self.enc3 = ConvBlock(c[1], c[2], batchnorm)
        self.enc4 = ConvBlock(c[2], c[3], batchnorm)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c[3], c[3] * 2, batchnorm)

        self.up4 = nn.ConvTranspose2d(c[3] * 2, c[3], 2, stride=2)
        self.dec4 = ConvBlock(c[3] * 2, c[3], batchnorm)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBlock(c[2] * 2, c[2], batchnorm)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBlock(c[1] * 2, c[1], batchnorm)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBlock(c[0] * 2, c[0], batchnorm)
        self.out = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


# Named width presets used by the distillation / lightweight study (Section 6).
CHANNEL_PRESETS = {
    "full": [64, 128, 256, 512],
    "reference": [32, 64, 128, 256],  # default / public-repo width
    "slim": [24, 48, 96, 192],
    "micro": [16, 32, 64, 128],
}


def build_unet(name_or_channels="reference", in_channels=3, batchnorm=True, num_classes=1) -> UNet:
    if isinstance(name_or_channels, str):
        channels = CHANNEL_PRESETS[name_or_channels]
    else:
        channels = list(name_or_channels)
    return UNet(in_channels=in_channels, num_classes=num_classes, channels=channels, batchnorm=batchnorm)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
