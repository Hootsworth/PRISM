"""Frozen metadata-free Bayer-to-sRGB B1 network."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, bias=True),
                         nn.ReLU(inplace=True))


class B1PackedUNet(nn.Module):
    """Two-level U-Net operating on fixed-position packed Bayer planes.

    Input is N×4×H/2×W/2 ordered [TL,TR,BL,BR]. Output is N×3×H×W
    display-encoded sRGB in [0,1]. No CFA or camera metadata is accepted.
    """
    expected_parameters = 357276

    def __init__(self):
        super().__init__()
        self.stem = ConvRelu(4, 24)
        self.enc1a = ConvRelu(24, 24)
        self.enc1b = ConvRelu(24, 24)
        self.down1 = ConvRelu(24, 48, stride=2)
        self.enc2a = ConvRelu(48, 48)
        self.enc2b = ConvRelu(48, 48)
        self.down2 = ConvRelu(48, 96, stride=2)
        self.mid1 = ConvRelu(96, 96)
        self.mid2 = ConvRelu(96, 96)
        self.up2_reduce = nn.Conv2d(96, 48, kernel_size=1, bias=True)
        self.dec2a = ConvRelu(96, 48)
        self.dec2b = ConvRelu(48, 48)
        self.up1_reduce = nn.Conv2d(48, 24, kernel_size=1, bias=True)
        self.dec1a = ConvRelu(48, 24)
        self.dec1b = ConvRelu(24, 24)
        self.head = nn.Conv2d(24, 12, kernel_size=3, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.output_activation = nn.Sigmoid()
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != self.expected_parameters:
            raise RuntimeError("B1 parameter count mismatch: expected {}, got {}".format(
                self.expected_parameters, actual))

    @staticmethod
    def _pad_to_stride(value: torch.Tensor):
        h, w = value.shape[-2:]
        pad_h, pad_w = (-h) % 4, (-w) % 4
        if pad_h == 0 and pad_w == 0:
            return value, h, w
        mode = "reflect" if h > pad_h and w > pad_w else "replicate"
        return F.pad(value, (0, pad_w, 0, pad_h), mode=mode), h, w

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 4 or packed.shape[1] != 4:
            raise ValueError("B1 expects N×4×H/2×W/2 packed Bayer input; got {}".format(tuple(packed.shape)))
        if packed.dtype != torch.float32:
            raise TypeError("B1 frozen training/inference precision is float32; got {}".format(packed.dtype))
        x, original_h, original_w = self._pad_to_stride(packed)
        x = self.stem(x)
        skip1 = self.enc1b(self.enc1a(x))
        x = self.down1(skip1)
        skip2 = self.enc2b(self.enc2a(x))
        x = self.down2(skip2)
        x = self.mid2(self.mid1(x))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.up2_reduce(x), inplace=True)
        if x.shape[-2:] != skip2.shape[-2:]:
            x = x[..., :skip2.shape[-2], :skip2.shape[-1]]
        x = self.dec2b(self.dec2a(torch.cat((x, skip2), dim=1)))

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = F.relu(self.up1_reduce(x), inplace=True)
        if x.shape[-2:] != skip1.shape[-2:]:
            x = x[..., :skip1.shape[-2], :skip1.shape[-1]]
        x = self.dec1b(self.dec1a(torch.cat((x, skip1), dim=1)))
        x = self.pixel_shuffle(self.head(x))
        x = self.output_activation(x)
        return x[..., :original_h * 2, :original_w * 2]
