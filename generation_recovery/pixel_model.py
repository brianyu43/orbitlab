"""Small 64x64 conditional pixel-flow U-Net for the conditional P2 comparison."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

import torch
from torch import nn
from torch.nn import functional as F
import orbitlab as o


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, embedding: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.condition = nn.Linear(embedding, 2 * outgoing)
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.condition(embedding).chunk(2, dim=1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]))
        return (h + self.skip(x)) / math.sqrt(2)


class PixelVelocity(nn.Module):
    def __init__(self, equivariant: bool = False):
        super().__init__()
        self.equivariant = equivariant
        self.embed1 = nn.Linear(32 + 10, 128)
        self.embed2 = nn.Linear(128, 128)
        self.input = nn.Conv2d(3, 32, 3, padding=1)
        self.down0 = nn.ModuleList((ResidualBlock(32, 32, 128), ResidualBlock(32, 32, 128)))
        self.downsample0 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.down1 = nn.ModuleList((ResidualBlock(64, 64, 128), ResidualBlock(64, 64, 128)))
        self.downsample1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.middle = nn.ModuleList((ResidualBlock(64, 64, 128), ResidualBlock(64, 64, 128)))
        self.up1 = nn.ModuleList((ResidualBlock(128, 64, 128), ResidualBlock(64, 64, 128)))
        self.up0 = nn.ModuleList((ResidualBlock(96, 32, 128), ResidualBlock(32, 32, 128)))
        self.output_norm = nn.GroupNorm(8, 32)
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    def embedding(self, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if label.shape != (len(t), 2):
            raise ValueError("Expected one shape and one color per image")
        if bool(((label[:, 0] < 0) | (label[:, 0] >= 4) |
                 (label[:, 1] < 0) | (label[:, 1] >= 6)).any()):
            raise ValueError("P2 does not train null conditions")
        frequencies = torch.exp(torch.linspace(math.log(1), math.log(1000), 16, device=t.device, dtype=t.dtype))
        angles = t[:, None] * frequencies[None, :] * (2 * math.pi)
        time_features = torch.cat((angles.sin(), angles.cos()), 1)
        categories = torch.cat((F.one_hot(label[:, 0], 4), F.one_hot(label[:, 1], 6)), 1).to(t.dtype)
        return self.embed2(F.silu(self.embed1(torch.cat((time_features, categories), 1))))

    def raw(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(t, label)
        h = self.input(x)
        for block in self.down0:
            h = block(h, emb)
        skip0 = h
        h = self.downsample0(h)
        for block in self.down1:
            h = block(h, emb)
        skip1 = h
        h = self.downsample1(h)
        for block in self.middle:
            h = block(h, emb)
        h = torch.cat((F.interpolate(h, scale_factor=2, mode="nearest"), skip1), 1)
        for block in self.up1:
            h = block(h, emb)
        h = torch.cat((F.interpolate(h, scale_factor=2, mode="nearest"), skip0), 1)
        for block in self.up0:
            h = block(h, emb)
        return self.output(F.silu(self.output_norm(h)))

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if not self.equivariant:
            return self.raw(x, t, label)
        return torch.stack([o.rotate(self.raw(o.rotate(x, k), t, label), -k) for k in range(4)]).mean(0)
