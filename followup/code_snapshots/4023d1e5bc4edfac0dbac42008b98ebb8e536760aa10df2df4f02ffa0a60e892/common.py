"""Shared utilities; all new artifacts stay under followup/."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'work'))
import numpy as np
import torch
import orbitlab as o
import research as r


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def digest_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fresh_dir(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def original_orbits():
    return {v['orbit_sha256'] for split in ['train', 'val', 'test', 'ood']
            for v in json.loads((ROOT / 'data' / f'{split}_metadata.json').read_text())}


def render_fresh(n, seed, split, forbidden=None):
    seen = set(original_orbits() if forbidden is None else forbidden)
    arrays, labels, metadata = [], [], []
    candidate = 0
    while len(arrays) < n:
        a, label, meta = o.render_base(candidate, seed, 64, split)
        candidate += 1
        h = r.orbit_hash(a)
        if h in seen:
            continue
        seen.add(h)
        arrays.append(a)
        labels.append(label)
        metadata.append({**meta, 'orbit_sha256': h})
    return (torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255,
            torch.tensor(labels), metadata)


def update_status(task, status, evidence=None, note=None):
    p = HERE / 'status.json'
    d = json.loads(p.read_text())
    item = d['tasks'][task]
    item['status'] = status
    if evidence is not None:
        item['evidence'] = evidence
    if note is not None:
        item['note'] = note
    dump(p, d)
