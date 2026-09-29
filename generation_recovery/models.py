"""Conditioning ablation models for the 4x16 C4 latent velocity field."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))

import torch
from torch import nn
from torch.nn import functional as F
import orbitlab as o


def condition(label: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Both factors use -1 for the jointly dropped, learned null condition."""
    if label.shape[-1] != 2:
        raise ValueError("Expected shape and color indices")
    if not torch.equal(label[:, 0] < 0, label[:, 1] < 0):
        raise ValueError("Partial condition dropout is outside this ablation")
    active = (label[:, 0] >= 0).to(dtype)[:, None]
    a = F.one_hot(label[:, 0].clamp_min(0), 4).to(dtype)
    b = F.one_hot(label[:, 1].clamp_min(0), 6).to(dtype)
    return torch.cat((a, b), dim=1) * active


class ConditionalVelocity(nn.Module):
    def __init__(self, kind: str, width: int = 256):
        super().__init__()
        if kind not in {"concat", "film"}:
            raise ValueError(kind)
        self.kind = kind
        self.width = width
        if kind == "concat":
            self.net = nn.Sequential(nn.Linear(64 + 1 + 10, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 64))
        else:
            self.fc1 = nn.Linear(65, width)
            self.fc2 = nn.Linear(width, width)
            self.out = nn.Linear(width, 64)
            self.film1 = nn.Linear(10, 2 * width)
            self.film2 = nn.Linear(10, 2 * width)
            # Identity modulation at initialization; learning the condition is required.
            nn.init.zeros_(self.film1.weight)
            nn.init.zeros_(self.film1.bias)
            nn.init.zeros_(self.film2.weight)
            nn.init.zeros_(self.film2.bias)

    def raw(self, z: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        base = torch.cat((z.flatten(1), t[:, None]), dim=1)
        cond = condition(label, z.dtype)
        if self.kind == "concat":
            value = self.net(torch.cat((base, cond), dim=1))
        else:
            a1, b1 = self.film1(cond).chunk(2, dim=1)
            a2, b2 = self.film2(cond).chunk(2, dim=1)
            h = F.silu(self.fc1(base)) * (1 + a1) + b1
            h = F.silu(self.fc2(h)) * (1 + a2) + b2
            value = self.out(h)
        return value.reshape_as(z)

    def forward(self, z: torch.Tensor, t: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.stack([o.rho(self.raw(o.rho(z, k), t, label), -k) for k in range(4)]).mean(0)


def from_variant(variant: str) -> ConditionalVelocity:
    if variant not in {"F0", "F1", "F2", "F3"}:
        raise ValueError(variant)
    return ConditionalVelocity("film" if variant in {"F1", "F3"} else "concat")


def guided(flow: ConditionalVelocity, z: torch.Tensor, t: torch.Tensor,
           label: torch.Tensor, scale: float) -> torch.Tensor:
    conditional = flow(z, t, label)
    if scale == 1:
        return conditional
    blank = torch.full_like(label, -1)
    unconditional = flow(z, t, blank)
    return unconditional + scale * (conditional - unconditional)
