"""C4 decoder interventions; all new state stays under research_v3.

The spatial arm changes encoder, decoder and reconstruction objective. The
coordinate arm adds true-center supervision. Neither is an architecture-only
matched-information comparison. Standard versus edge is the controlled pair.
"""
import hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from v3_common import ROOT
import orbitlab as o
import research as r
from decoder_study import ControlledDecoder, rot_vector
from models import from_variant

CANDIDATES = ('standard', 'edge', 'spatial', 'coordinate')

def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def edge_loss(pred, target):
    mask = (target.amax(1, keepdim=True) > .05).float()
    error = (pred - target).square().mean(1, keepdim=True)
    area = mask.sum((1, 2, 3)).clamp_min(1)
    fg = (error * mask).sum((1, 2, 3)) / area
    bg = (error * (1-mask)).sum((1, 2, 3)) / (1-mask).sum((1, 2, 3)).clamp_min(1)
    gradients = sum((torch.diff(pred, dim=d)-torch.diff(target, dim=d)).square().mean(1).sum((1, 2)) / area for d in (-1, -2))
    return (fg + .1 * bg + .25 * gradients).mean()

class SpatialDecoder(nn.Module):
    """Each of four orientation blocks retains one 4x4 feature map."""
    def __init__(self):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1))
    def forward(self, z):
        logits = self.branch(z.reshape(-1, 1, 4, 4)).reshape(len(z), 4, 3, 64, 64)
        return torch.stack([o.rotate(logits[:, k], k) for k in range(4)]).mean(0).sigmoid()

class SpatialAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 1, 4, 2, 1))
        self.decoder = SpatialDecoder()
    def encode(self, x):
        z = self.encoder(torch.cat([o.rotate(x, -k) for k in range(4)]))
        return z.reshape(4, len(x), 16).transpose(0, 1).contiguous()
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        return self.decode(self.encode(x))

class CoordinateDecoder(nn.Module):
    """Learn an RGBA patch and its location; explicit bilinear placement.

No true center at inference, no template library, no true shape mask. Center
labels supervise the location head in addition to the image loss in training.
"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 32*4*4)
        self.patch = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 4, 4, 2, 1))
        self.xy = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 2), nn.Tanh())
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        self.register_buffer('pixels', torch.stack((xx, yy), -1).float(), persistent=False)
    def centers(self, z):
        return self.xy(z)
    def forward(self, z):
        flat = z.reshape(-1, 16)
        rgba = F.interpolate(self.patch(self.fc(flat).reshape(-1, 32, 4, 4)),
                             size=(33, 33), mode='bilinear', align_corners=True).sigmoid()
        center_px = (self.xy(flat)+1)*31.5
        grid = (self.pixels[None]-center_px[:, None, None, :])/16
        premultiplied = rgba[:, :3]*rgba[:, 3:4]
        placed = F.grid_sample(premultiplied, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        branches = placed.reshape(len(z), 4, 3, 64, 64)
        return torch.stack([o.rotate(branches[:, k], k) for k in range(4)]).mean(0)

class FactorBridge(nn.Module):
    """Diagnostic readout with true factor supervision, never a primary arm."""
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 16))
    def forward(self, z):
        return self.net(z)

def ae_model(representation):
    return SpatialAE() if representation == 'spatial' else r.ResearchAE('equivariant', 16)

def decoder_model(candidate):
    if candidate in ('standard', 'edge'): return ControlledDecoder('logit_mean')
    if candidate == 'spatial': return SpatialDecoder()
    if candidate == 'coordinate': return CoordinateDecoder()
    raise ValueError(candidate)

def representation(candidate):
    return 'spatial' if candidate == 'spatial' else 'base'

def oracle_codes(labels, metadata, rotation):
    """True factors. Pixel-center-correct coordinates for exact quarter turns."""
    fixed = torch.cat([F.one_hot(labels[:, 0], 4), F.one_hot(labels[:, 1], 6)], 1).float()
    center = torch.tensor([[(64*m['cx']-31.5)/31.5, (64*m['cy']-31.5)/31.5] for m in metadata])
    radius = torch.tensor([[m['radius']/.24] for m in metadata])
    direction = torch.tensor([[1., 0.]]).expand(len(labels), 2)
    return torch.stack([torch.cat([fixed, rot_vector(center, rotation-k), radius,
        rot_vector(direction, rotation-k), torch.ones(len(labels), 1)], 1) for k in range(4)], 1)

def decoder_loss(candidate, model, z, target, center):
    pred = model(z)
    image = o.reconstruction_loss(pred, target) if candidate == 'standard' else edge_loss(pred, target)
    coordinate = F.mse_loss((model.centers(z)-center)*31.5/8, torch.zeros_like(center)) if candidate == 'coordinate' else image.new_zeros(())
    return image + .1*coordinate, {'image': float(image.detach()), 'center_8px_mse': float(coordinate.detach())}
