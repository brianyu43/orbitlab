"""Matched supervised disk components, with explicit information boundaries."""
import itertools
import torch
from torch import nn
import torch.nn.functional as F
from r3_core import coarse
from dynamics_models import rotate_state, rotate_vec, rotate_context, rotate_motion

ARMS = ('reader', 'force', 'decoder', 'bounded', 'contact')
STEPS = {'reader': 4000, 'force': 3000, 'decoder': 3000, 'bounded': 2000, 'contact': 2000}


def encode(raw, rgb):
    raw = torch.as_tensor(raw, dtype=torch.float32)
    rgb = torch.as_tensor(rgb, dtype=torch.float32)
    while rgb.ndim < raw.ndim:
        rgb = rgb.unsqueeze(1)
    rgb = rgb.expand(*raw.shape[:-1], 3)
    scale = raw.new_tensor([31.5, 31.5, 3, 3, 8])
    return torch.cat([raw[..., :5] / scale, rgb, torch.ones_like(raw[..., :1])], -1)


def context(raw):
    v = torch.as_tensor(raw, dtype=torch.float32)
    return torch.cat([(v[..., :2] + v[..., 2:4]) / .1, v[..., 4:5] / .01], -1)


def rgb_window(images, ep, frame):
    # Input images E,T,H,W,C, chronological t-3..t. No future observations.
    offsets = torch.arange(-3, 1)
    return images[ep[:, None], frame[:, None] + offsets].permute(0, 1, 4, 2, 3).float() / 255


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(12, 32, 5, 2, 2), nn.SiLU(), nn.Conv2d(32, 48, 5, 2, 2), nn.SiLU(), nn.Conv2d(48, 48, 3, 1, 1), nn.SiLU())
        self.heatmap = nn.Conv2d(48, 1, 1)
        self.crop = nn.Sequential(nn.Conv2d(12, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(800, 96), nn.SiLU(), nn.Linear(96, 8))
        yy, xx = torch.meshgrid(torch.arange(33) - 16, torch.arange(33) - 16, indexing='ij')
        self.register_buffer('crop_grid', torch.stack([xx, yy], -1).float())
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        self.register_buffer('full_grid', torch.stack([xx, yy], -1).float())

    def forward(self, window):
        b = len(window)
        image = window.flatten(1, 2)
        logits = F.interpolate(self.heatmap(self.backbone(image)), size=64, mode='bilinear', align_corners=False)[:, 0]
        peaks = logits.detach().clone()
        positions, scores = [], []
        for _ in range(6):
            index = peaks.flatten(1).argmax(-1)
            point = torch.stack([index % 64, index // 64], -1).float()
            positions.append(point)
            scores.append(logits.flatten(1).gather(1, index[:, None])[:, 0])
            peaks = peaks.masked_fill((self.full_grid[None] - point[:, None, None]).square().sum(-1) <= 36, -1000)
        xy = torch.stack(positions, 1)
        logits_at_peaks = torch.stack(scores, 1)
        grid = (self.crop_grid[None, None] + xy[:, :, None, None]) / 31.5 - 1
        patches = F.grid_sample(image[:, None].expand(-1, 6, -1, -1, -1).reshape(b * 6, 12, 64, 64), grid.reshape(b * 6, 33, 33, 2), align_corners=True)
        v = self.crop(patches).reshape(b, 6, 8)
        pos = ((xy + 4 * v[..., :2].tanh()).clamp(0, 63) - 31.5) / 31.5
        velocity = 2 * v[..., 2:4].tanh()  # Fixed +-6px/frame support, not test fitted.
        radius = (3 + 4 * v[..., 4:5].sigmoid()) / 8
        state = torch.cat([pos, velocity, radius, v[..., 5:8].sigmoid(), logits_at_peaks.sigmoid()[..., None]], -1)
        return {'state': state, 'presence_logits': logits_at_peaks, 'heatmap_logits': logits}


def reader_loss(output, target):
    s = output['state']
    d = s[:, None, :, :8] - target[:, :, None, :8]
    cost = 4 * (d[..., :2] * 31.5 / 8).square().mean(-1) + d[..., 2:4].square().mean(-1) + 4 * (d[..., 4] * 8 / 2).square() + 4 * d[..., 5:8].square().mean(-1)
    match_cost = cost - (2 / 6) * output['presence_logits'][:, None]
    combos = torch.tensor(list(itertools.permutations(range(6), 2)), device=s.device)
    assignment = combos[(match_cost[:, 0, combos[:, 0]] + match_cost[:, 1, combos[:, 1]]).detach().argmin(-1)]
    present = torch.zeros_like(output['presence_logits']).scatter(1, assignment, 1)
    loss = cost.gather(2, assignment[..., None]).mean() + F.binary_cross_entropy_with_logits(output['presence_logits'], present)
    yy, xx = torch.meshgrid(torch.arange(64, device=s.device), torch.arange(64, device=s.device), indexing='ij')
    grid = torch.stack([xx, yy], -1)
    delta = grid[None, None] - (target[:, :, None, None, :2] * 31.5 + 31.5)
    heat = (-delta.square().sum(-1) / (2 * 1.5 ** 2)).exp().amax(1)
    return loss + 10 * ((output['heatmap_logits'].sigmoid() - heat).square() * (1 + 50 * heat)).mean()


class ForceReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(12, 24, 5, 2, 2), nn.SiLU(), nn.Conv2d(24, 48, 5, 2, 2), nn.SiLU(), nn.Conv2d(48, 64, 3, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 96), nn.SiLU(), nn.Linear(96, 3))

    def forward(self, window):
        x = self.net(window.flatten(1, 2))
        return torch.cat([1.5 * x[:, :2].tanh(), 1.5 * x[:, 2:].sigmoid()], -1)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 4 * 17 * 17))
        yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
        self.register_buffer('grid', torch.stack([xx, yy], -1).float() - 31.5)

    def forward(self, state):
        b, n, _ = state.shape
        if n == 0:
            return state.new_zeros(b, 3, 64, 64)
        # Only radius/RGB enter patches. True/learned XY places them. No sourceRGB bypass.
        p = self.net(state[..., 4:8]).reshape(b * n, 4, 17, 17).sigmoid()
        p = torch.cat([p[:, :3] * p[:, 3:], p[:, 3:]], 1)
        grid = (self.grid[None, None] - state[:, :, None, None, :2] * 31.5) / 8
        layers = F.grid_sample(p, grid.reshape(b * n, 64, 64, 2), align_corners=True).reshape(b, n, 4, 64, 64)
        layers = layers * state[:, :, 8, None, None, None]
        out = state.new_zeros(b, 3, 64, 64)
        for j in range(n):
            out = out * (1 - layers[:, j, 3:]) + layers[:, j, :3]
        return out


class Interaction(nn.Module):
    def __init__(self, contact=False):
        super().__init__()
        self.contact = contact
        self.edge = nn.Sequential(nn.Linear(25, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
        self.net = nn.Sequential(nn.Linear(62, 96), nn.SiLU(), nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def raw(self, s, a, c):
        b, n, _ = s.shape
        sa = torch.cat([s, a], -1)
        edges = self.edge(torch.cat([sa[:, :, None].expand(-1, -1, n, -1), sa[:, None].expand(-1, n, -1, -1), c[:, None, None].expand(-1, n, n, -1)], -1))
        mask = 1 - torch.eye(n, device=s.device)[None]
        if self.contact:
            distance = (s[:, :, None, :2] - s[:, None, :, :2]).norm(dim=-1) * 31.5
            mask = mask * (distance <= (s[:, :, None, 4] + s[:, None, :, 4]) * 8 + 6)
        message = (edges * mask[..., None]).sum(2)
        if self.contact:
            message = message / mask.sum(2, keepdim=True).clamp_min(1)
        return self.net(torch.cat([sa, c[:, None].expand(-1, n, -1), message], -1))

    def forward(self, s, a, c, velocity_limit):
        correction = torch.stack([rotate_motion(self.raw(rotate_state(s, -k), rotate_vec(a, -k), rotate_context(c, -k)), k) for k in range(4)]).mean(0)
        if self.contact:
            motion = coarse(s, a, c, .15 * correction[..., 2:].tanh())
        else:
            v = s[..., 2:4] + a
            motion = torch.cat([s[..., :2] + v * 3 / 31.5, v], -1) + correction
            wall = 1 - s[..., 4:5] * 8 / 31.5
            xy = torch.maximum(torch.minimum(motion[..., :2], wall), -wall)
            v = motion[..., 2:]
            v = v * (velocity_limit / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
            motion = torch.cat([xy, v], -1)
        return torch.cat([motion, s[..., 4:]], -1)


def make(arm):
    if arm == 'reader':
        return Reader()
    if arm == 'force':
        return ForceReader()
    if arm == 'decoder':
        return Decoder()
    return Interaction(contact=arm == 'contact')
