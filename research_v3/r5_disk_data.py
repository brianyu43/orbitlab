"""Fresh, shared moving-disk families for perception, dynamics and rendering.

All numerical physics is the preserved float64 R3 reference. Continuous colors
are a separate sidecar; the old palette index never enters learned models.
"""
import argparse
import hashlib
import json
import time
import numpy as np
from v3_common import HERE, ROOT, dump, lock, sha
import dynamics_world as world
import r3_physics as physics

BASE = HERE / 'r5' / 'disks'
SEED = 891101
COUNTS = [2, 3, 4, 6]
HORIZONS = [1, 16, 61, 128]


def freeze():
    return lock(BASE / 'data_protocol.json', {
        'version': 1, 'seed': SEED,
        'sources': {'r5_disk_data.py': sha(__file__), 'r3_physics.py': sha(HERE / 'r3_physics.py'),
                    'followup/dynamics_world.py': sha(ROOT / 'followup/dynamics_world.py')},
        'train': {'episodes': 960, 'count': 2, 'transitions': 20},
        'calibration': {'episodes': 96, 'count': 2, 'transitions': 131},
        'test': {'episodes_per_count': 96, 'counts': COUNTS, 'transitions': 131},
        'input': 'Four consecutive RGB frames0..3; first intervention occurs at frame3. 64px black background, equal-mass disks, radii4/5/6. No occlusion panel or textured background; tangent disks can overlap only antialiased boundaries.',
        'colors': 'Continuous independent RGB in [40,240]/255, half of scene families have identical colors for all disks. Palette indices affect only the preserved proposal sampler; never reader/transition/decoder input.',
        'shared_training': 'Every reader, force head, transition and neural decoder is trained only on the same960 two-disk train families and frames0..20. No old learned weights reused. Independent supervised components; no end-to-end fine-tuning.',
        'strata': 'Equal quotas for no contact, wall-only, pair contact during forecast steps1..16. R3 controlled proposals reused: free velocities/forces/actions scaled .1/.01/.1; wall .05/.1/.1 with one outward1px/frame disk7px from wall; pair unscaled. Different intervention/force distributions are reported, not population representative.',
        'counterfactuals': 'From observed frame3: negate clicked disk impulse, or increment x net force by .04. Original observed frames identical. Known intervention delta also supplied to inferred baseline force.',
        'split': 'Fresh train/val/test numerical seeds through original sampler. All time windows, counterfactuals and C4 relatives stay in one scene family. Initial physical orbit hashes checked across all fresh groups and existing R3 groups, independent of RGB differences.',
        'physics': 'Eight substeps,32 maximum contact projection sweeps, float64. Equal masses, radius-dependent reflective walls. This simulator generates labels, not a learned model.',
        'renderer': 'Original radial alpha clip(r+.5-distance,0,1), stable persistent-ID painter order, continuous RGB sidecar, round to uint8. No inferred object reorder using true IDs.',
    })


def render(states, colors):
    """Batch states B,N,7 and colors B,N,3 in [0,1] -> B,64,64,3 uint8."""
    s = np.asarray(states, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    assert s.ndim == 3 and colors.shape == (*s.shape[:2], 3)
    yy, xx = np.mgrid[:64, :64].astype(np.float64)
    xx -= 31.5
    yy -= 31.5
    out = np.zeros((len(s), 64, 64, 3), np.float64)
    for j in range(s.shape[1]):
        distance = np.sqrt((xx - s[:, j, 0, None, None]) ** 2 + (yy - s[:, j, 1, None, None]) ** 2)
        alpha = np.clip(s[:, j, 4, None, None] + .5 - distance, 0, 1)[..., None]
        out = out * (1 - alpha) + colors[:, j, None, None] * 255 * alpha
    return np.rint(out).clip(0, 255).astype(np.uint8)


def category(events):
    return np.where(events[:, :, 0].sum(1) > 0, 2, np.where(events[:, :, 1].sum(1) > 0, 1, 0))


def path(split, count):
    return BASE / 'data' / split / f'n{count}'


def prepare_group(split, count):
    cfg = freeze()
    root = path(split, count)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        rec = json.loads((root / 'manifest.json').read_text())
        assert rec['protocol_sha256'] == sha(BASE / 'data_protocol.json')
        for name, digest in rec['files'].items():
            assert sha(root / name) == digest
        return
    n = 960 if split == 'train' else 96
    frames = 20 if split == 'train' else 131
    quota = n // 3
    buckets = [[], [], []]
    candidate = 0
    start = time.perf_counter()
    while min(map(len, buckets)) < quota:
        samples = []
        for _ in range(128):
            requested = candidate % 3
            mode = ['isotropic', 'fixed_gravity', 'variable_force'][(candidate // 3) % 3]
            s, ctx, actions, target = world.sample(candidate, SEED, split, mode, count)
            if requested == 0:
                s[:, 2:4] *= .1
                ctx[:4] *= .01
                actions *= .1
            elif requested == 1:
                s[:, 2:4] *= .05
                ctx[:4] *= .1
                actions *= .1
                side = 1 if (candidate // 9) % 2 else -1
                s[0, 0] = side * (31.5 - s[0, 4] - 7)
                s[0, 2] = side
                if physics.violation(s[None])[0] > 1e-8:
                    candidate += 1
                    continue
            samples.append((s, ctx, actions[:20], target, candidate, mode, requested))
            candidate += 1
        if samples:
            _, ev = physics.rollout(np.stack([v[0] for v in samples]), np.stack([v[1] for v in samples]), np.stack([v[2] for v in samples]))
            for sample, cat in zip(samples, category(ev[:, 3:19])):
                if cat == sample[6] and len(buckets[cat]) < quota:
                    buckets[cat].append(sample)
        if candidate > 50000:
            raise RuntimeError('Sampling exhausted; criteria must not silently change')
    selected = [buckets[k][i] for i in range(quota) for k in range(3)]
    initial = np.stack([v[0] for v in selected])
    contexts = np.stack([v[1] for v in selected])
    actions = np.zeros((n, frames, count, 2), np.float64)
    actions[:, :20] = np.stack([v[2] for v in selected])
    states, events = physics.rollout(initial, contexts, actions)
    colors = []
    same = []
    for i, v in enumerate(selected):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, ['train', 'val', 'test'].index(split), count, v[4], 891]))
        rgb = rng.uniform(40 / 255, 240 / 255, (count, 3))
        identical = (i // 3) % 2 == 0
        if identical:
            rgb[:] = rgb[0]
        colors.append(rgb)
        same.append(identical)
    colors = np.asarray(colors)
    keys = [world.initial_orbit_hash(s, ctx, a[3]) for s, ctx, a in zip(initial, contexts, actions)]
    assert len(keys) == len(set(keys))
    stored = {'states': states, 'events': events, 'contexts': contexts, 'actions': actions,
              'colors': colors, 'same_color': np.asarray(same), 'stratum': np.tile(np.arange(3), quota),
              'target_ids': np.asarray([v[3] for v in selected])}
    if split != 'train':
        rev, rev_events = physics.rollout(states[:, 3], contexts, -actions[:, 3:])
        force_contexts = contexts.copy()
        force_contexts[:, 2] += .04
        force, force_events = physics.rollout(states[:, 3], force_contexts, actions[:, 3:])
        stored.update(reversed_states=rev, reversed_events=rev_events, force_states=force, force_events=force_events, force_contexts=force_contexts)
    np.savez_compressed(root / 'trajectories.npz', **stored)
    times = range(21) if split == 'train' else range(4)
    images = np.stack([render(states[:, t], colors) for t in times], 1)
    np.save(root / 'images.npy', images)
    dump(root / 'metadata.json', [{'candidate': v[4], 'mode': v[5], 'initial_orbit_sha256': key,
                                 'stratum': i % 3, 'same_color': bool(same[i])} for i, (v, key) in enumerate(zip(selected, keys))])
    max_violation = max(float(physics.violation(states[:, t]).max()) for t in range(states.shape[1]))
    assert max_violation <= 1.01e-7
    # Renderer C4 and physical rollout checks use all four rotations on a fixed
    #16-scene prefix, including the three contact strata and both color regimes.
    checks = []
    for k in range(4):
        rs = np.stack([world.rotate_state(s, k) for s in states[:16, 3]])
        rc = np.stack([world.rotate_context(c, k) for c in contexts[:16]])
        ra = world.rotate_vectors(actions[:16, 3:19], k)
        rr, _ = physics.rollout(rs, rc, ra)
        expected = np.stack([[world.rotate_state(s, k) for s in seq] for seq in states[:16, 3:20]])
        delta = float(np.max(np.abs(rr - expected)))
        assert delta < 1e-7, delta
        assert np.array_equal(render(rs, colors[:16]), np.rot90(images[:16, 3], k, axes=(1, 2)))
        checks.append({'quarterturn': k, 'max_abs_state_error': delta, 'renderer_bitexact': True})
    distributions = {str(k): {'episodes': int((stored['stratum'] == k).sum()),
                             'net_force_mean': float(np.linalg.norm(contexts[stored['stratum'] == k, :2] + contexts[stored['stratum'] == k, 2:4], axis=-1).mean()),
                             'impulse_mean': float(np.linalg.norm(actions[stored['stratum'] == k, 3], axis=-1).max(1).mean())} for k in range(3)}
    files = {name: sha(root / name) for name in ['trajectories.npz', 'images.npy', 'metadata.json']}
    dump(root / 'manifest.json', {'protocol_sha256': sha(BASE / 'data_protocol.json'), 'files': files,
                               'episodes': n, 'count': count, 'transitions': frames, 'same_color_episodes': sum(same),
                               'candidates': candidate, 'stratum_distributions': distributions, 'max_geometry_violation': max_violation,
                               'C4_checks': checks, 'seconds': time.perf_counter() - start})
    print('disk data', split, count, n, 'candidates', candidate, flush=True)


def verify():
    freeze()
    seen = set()
    groups = []
    for split, count in [('train', 2), ('val', 2)] + [('test', n) for n in COUNTS]:
        root = path(split, count)
        manifest = json.loads((root / 'manifest.json').read_text())
        for name, digest in manifest['files'].items():
            assert sha(root / name) == digest
        keys = {v['initial_orbit_sha256'] for v in json.loads((root / 'metadata.json').read_text())}
        assert not seen & keys
        seen |= keys
        a = np.load(root / 'trajectories.npz')
        replay, events = physics.rollout(a['states'][:, 0], a['contexts'], a['actions'])
        assert np.array_equal(replay, a['states']) and np.array_equal(events, a['events'])
        images = np.load(root / 'images.npy', mmap_mode='r')
        for t in range(images.shape[1]):
            assert np.array_equal(render(a['states'][:, t], a['colors']), images[:, t])
        if split != 'train':
            for key, ctx, acts in [('reversed', a['contexts'], -a['actions'][:, 3:]), ('force', a['force_contexts'], a['actions'][:, 3:])]:
                rep, ev = physics.rollout(a['states'][:, 3], ctx, acts)
                assert np.array_equal(rep, a[key + '_states']) and np.array_equal(ev, a[key + '_events'])
        groups.append({'split': split, 'count': count, 'episodes': len(images), 'images_replayed': int(len(images) * images.shape[1]),
                       'all_physics_and_counterfactual_arrays_bitexact': True, 'manifest_sha256': sha(root / 'manifest.json')})
    historical = set()
    prior_paths = sorted((HERE / 'r3' / 'data').glob('d*/*/n*/metadata.json'))
    for p in prior_paths:
        historical.update(v['initial_orbit_sha256'] for v in json.loads(p.read_text()))
    assert not historical & seen
    dump(BASE / 'data_verification.json', {'protocol_sha256': sha(BASE / 'data_protocol.json'), 'groups': groups,
                                        'new_unique_families': len(seen), 'prior_R3_families': len(historical),
                                        'new_orbit_overlap': 0, 'cross_R3_orbit_overlap': 0,
                                        'prior_scope': 'Existing research_v3/R3 family hashes; original historical datasets are preserved but not all independently enumerated here.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'verify'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        for split, count in [('train', 2), ('val', 2)] + [('test', n) for n in COUNTS]:
            prepare_group(split, count)
    else:
        verify()
