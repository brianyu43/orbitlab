"""Whole-family split construction; preserve all renderer variants together."""
import argparse, dataclasses, json, time
import numpy as np
import torch
import r1_core as c
from v3_common import ROOT, HERE, sha, dump, lock, event

def prior_families():
    found = set()
    paths = list((ROOT / 'followup/data/object_world_v1').glob('*/scenes.json'))
    paths += list((ROOT / 'followup/data/object_image_edit_v1').glob('*/scenes.json'))
    for name in ['perception', 'perception_detail']:
        paths += list((ROOT / f'completion_v2/{name}/data').glob('d*/*/scenes.json'))
    for path in paths:
        for row in json.loads(path.read_text()):
            found.add(c.canonical(row['objects']))
        for name in ['tasks.json', 'edits.json']:
            p = path.with_name(name)
            if p.exists():
                for row in json.loads(p.read_text()):
                    if 'target_objects' in row: found.add(c.canonical(row['target_objects']))
    for p in (c.BASE / 'data').glob('d*/*/labels.json'):
        for row in json.loads(p.read_text()):
            found.add(c.canonical(row['objects']))
            found.update(c.canonical(t['target']) for t in c.tasks(row['objects'], row['source_index']))
    return found

def audit_renderers():
    rows = []
    for shape in range(4):
        vertices = {tuple(np.round(v, 6)) for v in c.o.POLYGONS[shape]}
        half_turn = vertices == {tuple(np.round([-v[0], -v[1]], 6)) for v in c.o.POLYGONS[shape]}
        assert half_turn == (shape == 3), (shape, half_turn)
        for radius in range(5, 9):
            for renderer in c.RENDERERS:
                a = c.alpha(shape, radius, 0, renderer)
                b = c.alpha(shape, radius, 2, renderer)
                if renderer == 'symmetric4_lanczos' and shape == 3: assert np.array_equal(a, b)
                rows.append({'shape': shape, 'radius': radius, 'renderer': renderer, 'geometric_half_turn_symmetry': half_turn, 'raster_half_turn_equal': bool(np.array_equal(a, b)), 'max_half_turn_difference': float(np.abs(a-b).max())})
    dump(c.BASE / 'renderer_audit.json', {'rows': rows, 'source_sha256': sha(__file__), 'rule': 'Do not require identifying a pose label when the rendering is identical; raw full-label accuracy is diagnostic only.'})

def prepare(seed):
    c.freeze(); audit_renderers()
    root = c.BASE / f'data/d{seed}'
    if (root / 'manifest.json').exists():
        manifest = json.loads((root / 'manifest.json').read_text())
        for rel, digest in manifest['files'].items(): assert sha(root / rel) == digest
        return
    seen = prior_families(); old_count = len(seen)
    files, splits = {}, {}
    for split, n in [('train', 1024), ('val', 128), ('test', 128), ('ood', 128)]:
        folder = root / split; folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'labels.json').exists():
            raise RuntimeError('Partial data build: retain the partial directory and investigate before resuming.')
        images = {k: [] for k in c.RENDERERS}; records = []; candidate = 0
        rejected = {'family_overlap': 0, 'edited_overlap': 0}
        while len(records) < n:
            objects = [dataclasses.asdict(v) for v in c.w.sample_scene(candidate, seed, split)]
            candidate += 1
            planned = c.tasks(objects, len(records))
            family = {c.canonical(objects), *(c.canonical(t['target']) for t in planned)}
            if family & seen:
                rejected['family_overlap'] += 1; continue
            # Prevent renderer-specific occlusion confounding R1.
            members = [objects] + [t['target'] for t in planned]
            if any((((c.render(s, r)[1] > .05).sum(0)) > 1).any() for s in members for r in c.RENDERERS):
                rejected['edited_overlap'] += 1; continue
            source_index = len(records)
            records.append({'source_index': source_index, 'candidate': candidate-1, 'family_keys': sorted(family), 'objects': objects})
            seen.update(family)
            for renderer in c.RENDERERS: images[renderer].append(c.render(objects, renderer)[0])
        dump(folder / 'labels.json', records)
        for renderer, rows in images.items():
            np.savez_compressed(folder / f'{renderer}.npz', images=np.stack(rows))
        for path in folder.iterdir(): files[str(path.relative_to(root))] = sha(path)
        splits[split] = {'scenes': n, 'objects': 2*n, 'candidate_count': candidate, 'rejections': rejected}
        print('R1 data', seed, split, splits[split], flush=True)
    dump(root / 'manifest.json', {'seed': seed, 'splits': splits, 'files': files, 'prior_families': old_count, 'new_family_keys': len(seen)-old_count, 'source_sha256': sha(__file__), 'protocol_sha256': sha(c.BASE/'protocol.json')})
    event('R1_data_complete', seed=seed)

def training_data(seed, kind):
    root = c.BASE / f'data/d{seed}/train'
    images = np.load(root / 'pil4_lanczos.npz')['images']
    rows = json.loads((root / 'labels.json').read_text())
    xs, ys = [], []
    for image, row in zip(images, rows):
        x, records = c.inputs(image, kind); bycolor = {v['color']: v for v in row['objects']}
        assert len(x) == 2
        for a, r in zip(x, records):
            v = bycolor[r['color']]; xs.append(a)
            ys.append([v['shape'], v['pose'], v['radius']-5, (v['x']-r['cx'])/8, (v['y']-r['cy'])/8])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=c.DEVELOPMENT)
    args = parser.parse_args(); prepare(args.seed)
