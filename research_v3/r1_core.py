"""R1: matched crop readers and renderers, with explicit geometric equivalence."""
from v3_common import ROOT, HERE, sha, dump, lock
import dataclasses, hashlib, json
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
import orbitlab as o
import object_world as w

BASE = HERE / 'r1'
RENDERERS = ['pil4_lanczos', 'symmetric4_lanczos', 'pil8_box']
ARMS = [f'{kind}_{loss}' for kind in ['binary', 'alpha', 'rgb'] for loss in ['full', 'quotient']]
DEVELOPMENT = 886101
CONFIRMATION = [886201, 886202, 886203]

def freeze():
    return lock(BASE / 'protocol.json', {
        'sources': {p: sha(HERE / p) for p in ['v3_common.py', 'r1_core.py', 'r1_data.py', 'r1_train.py', 'r1_evaluate.py']},
        'dependencies': {p: sha(ROOT / p) for p in ['work/orbitlab.py', 'followup/object_world.py']},
        'development_seed': DEVELOPMENT, 'confirmation_seeds': CONFIRMATION, 'initializations': [0, 1, 2],
        'arms': ARMS, 'renderers': RENDERERS, 'training_renderer': RENDERERS[0],
        'train_scenes_per_seed': 1024, 'evaluation_scenes_per_split': 128, 'splits': ['train', 'val', 'test', 'ood'],
        'training_steps': 6000, 'batch': 64, 'optimizer': 'AdamW lr=.001 weight_decay=.0001 gradient_clip=1',
        'loss': 'shape CE + pose CE or half-turn quotient log-sum-probability for shape 3 + radius CE + 4*MSE(center_offset/8)',
        'matching': 'All arms have 3 input channels and exactly the same architecture and parameter count; binary and intensity are repeated across channels. Same batches, initial weights, train scenes and optimizer updates.',
        'priors': 'Known six-color palette and distinct object colors; RGB-derived masks and bounding-box crop; supervised shape/pose/size/center labels. Analytic edit controller and known renderer. Not unsupervised perception.',
        'renderers_detail': 'Original PIL supersampling4 Lanczos; same raster explicitly averaged with its half-turn for geometrically symmetric shape3; independent supersampling8 BOX raster. All poses are exact rot90 of their canonical raster.',
        'split_unit': 'Whole source vector scene and all edit/rotation/renderer variants; reject canonical geometric source/target family overlap across old and new scenes.',
        'training_scope': 'New train data per confirmation seed. Perceptual readers trained on seen factor pairs. OOD places one object in a held-out shape-color pair.',
        'metrics': 'Raw strict label accuracy is descriptive; primary editing respects geometric half-turn equivalence for shape3. Exact raster equivalence, identifiable pose, shape/color/position/size, selection and non-target preservation are separate.',
        'template_reference': 'Exhaustive known-renderer crop template matching; renderer identity supplied only to this diagnostic nonlearned reference. Not an equal-information learned competitor.',
        'development_selection': 'All six arms reported. Relative to alpha_full, select maximum mean quotient-edit success on both unseen renderers, then minimum worst drop, then arm order. Advance only if gain>=.05 and drop<=.05 on BOTH unseen renderers for both val and OOD. No development test use.',
        'confirmation_rule': 'If development gate passes, baseline and selected candidate are retrained on all 3 fresh datasets x 3 initializations, reporting both unseen renderers and seen renderer. If it fails, no confirmatory robustness claim and no automatic repeat expansion; retain full six-arm development diagnosis.',
        'primary_claim': 'Renderer generalization, not semantic pose identifiability from identical pixels.',
        'resource_cap': {'local_seconds': 43200, 'disk_bytes': 30_000_000_000}})

@lru_cache(maxsize=2048)
def alpha(shape, radius, pose, renderer):
    if renderer in RENDERERS[:2]:
        a = w.local_alpha(shape, radius, 0).copy()
        if renderer == RENDERERS[1] and shape == 3:
            a = (a + np.rot90(a, 2)) / 2
    else:
        assert renderer == RENDERERS[2]
        side, ss = 33, 8
        im = Image.new('L', (side * ss, side * ss))
        vertices = [((side / 2 + x * radius) * ss, (side / 2 + y * radius) * ss) for x, y in o.POLYGONS[shape]]
        ImageDraw.Draw(im).polygon(vertices, fill=255)
        a = np.asarray(im.resize((side, side), Image.Resampling.BOX), dtype=np.float32) / 255
    return np.ascontiguousarray(np.rot90(a, pose % 4))

def render(scene, renderer):
    rgb = np.zeros((64, 64, 3), np.float32)
    layers = []
    for v in scene:
        x, y = int(round(float(v['x']))), int(round(float(v['y'])))
        patch = alpha(int(v['shape']), int(v['radius']), int(v['pose']), renderer)
        mask = np.zeros((64, 64), np.float32)
        x0, y0 = x - 16, y - 16
        l, t, r, b = max(0, x0), max(0, y0), min(64, x0 + 33), min(64, y0 + 33)
        if r > l and b > t:
            mask[t:b, l:r] = patch[t-y0:b-y0, l-x0:r-x0]
        rgb = rgb * (1-mask[..., None]) + np.asarray(o.PALETTE[int(v['color'])]) * mask[..., None]
        layers.append(mask)
    return np.rint(rgb).clip(0, 255).astype(np.uint8), np.stack(layers)

def canonical(scene):
    variants = []
    current = [dict(v) for v in scene]
    for _ in range(4):
        variants.append(tuple(sorted((v['shape'], v['color'], v['x'], v['y'], v['radius'], v['pose'] % (2 if v['shape'] == 3 else 4)) for v in current)))
        current = [{**v, 'x': v['y'], 'y': 63-v['x'], 'pose': (v['pose']+1) % 4} for v in current]
    return hashlib.sha256(repr(min(variants)).encode()).hexdigest()

def tasks(scene, index):
    tid = index % len(scene)
    obj = scene[tid]
    colors = [c for c in range(6) if c not in [v['color'] for v in scene]]
    color = colors[index % len(colors)]
    dx = 5 if obj['x'] <= 47 else -5
    output = []
    for kind in ['color', 'rotate', 'translate', 'color_rotate']:
        target = [dict(v) for v in scene]
        if 'color' in kind: target[tid]['color'] = color
        if 'rotate' in kind: target[tid]['pose'] = (target[tid]['pose'] + 1) % 4
        if kind == 'translate': target[tid]['x'] += dx
        output.append({'kind': kind, 'target_id': tid, 'color': color, 'dx': dx, 'target': target})
    return output

def inputs(image, kind):
    rgb = image.astype(np.float32)
    palette = np.asarray(o.PALETTE, dtype=np.float32)
    unit = rgb / np.linalg.norm(rgb, axis=-1, keepdims=True).clip(1e-6)
    labels = (unit @ (palette / np.linalg.norm(palette, axis=-1, keepdims=True)).T).argmax(-1)
    foreground = rgb.max(-1) > 13
    xs, records = [], []
    for color in range(6):
        mask = (labels == color) & foreground
        if mask.sum() < 3: continue
        yy, xx = np.nonzero(mask)
        cx, cy = int(np.rint((xx.min()+xx.max())/2)), int(np.rint((yy.min()+yy.max())/2))
        if kind == 'rgb': value = rgb / 255 * mask[..., None]
        else:
            value = mask.astype(np.float32) if kind == 'binary' else np.clip((rgb @ palette[color]) / np.square(palette[color]).sum(), 0, 1) * mask
            value = np.repeat(value[..., None], 3, -1)
        padded = np.pad(value, ((16, 16), (16, 16), (0, 0)))
        xs.append(torch.from_numpy(padded[cy:cy+33, cx:cx+33].copy()).permute(2, 0, 1))
        records.append({'color': color, 'cx': cx, 'cy': cy})
    return torch.stack(xs) if xs else torch.empty(0, 3, 33, 33), records

class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.SiLU(), nn.Flatten(), nn.Linear(800, 96), nn.SiLU(), nn.Linear(96, 14))
    def forward(self, x): return self.net(x)

def loss(pred, target, kind):
    shape, pose = target[:, 0].long(), target[:, 1].long()
    lp = pred[:, 4:8].log_softmax(-1)
    pl = -lp.gather(1, pose[:, None])[:, 0]
    if kind == 'quotient':
        eq = -torch.logsumexp(torch.stack([lp.gather(1, (pose % 2)[:, None])[:, 0], lp.gather(1, (pose % 2+2)[:, None])[:, 0]], -1), -1)
        pl = torch.where(shape == 3, eq, pl)
    return F.cross_entropy(pred[:, :4], shape) + pl.mean() + F.cross_entropy(pred[:, 8:12], target[:, 2].long()) + 4 * F.mse_loss(pred[:, 12:14], target[:, 3:5])

def decode(raw, records):
    return [{'id': i, 'color': r['color'], 'shape': int(p[:4].argmax()), 'pose': int(p[4:8].argmax()), 'radius': int(p[8:12].argmax())+5, 'x': float(p[12])*8+r['cx'], 'y': float(p[13])*8+r['cy']} for i, (p, r) in enumerate(zip(raw, records))]

@torch.no_grad()
def infer(images, model, kind):
    all_x, detections, counts = [], [], []
    for image in images:
        x, d = inputs(image, kind)
        all_x.append(x); detections.append(d); counts.append(len(d))
    x = torch.cat(all_x)
    raw = torch.cat([model(x[i:i+128]) for i in range(0, len(x), 128)])
    outputs, offset = [], 0
    for d, count in zip(detections, counts):
        outputs.append(decode(raw[offset:offset+count], d)); offset += count
    return outputs
