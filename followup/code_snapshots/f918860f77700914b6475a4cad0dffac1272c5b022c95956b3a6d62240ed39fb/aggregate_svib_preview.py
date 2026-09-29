"""Scene-paired public-preview comparisons, preserving all fixed initializations."""
import argparse
import csv
import json
import time
import numpy as np
from common import HERE, dump, r
from svib_preview_data import NAME, ALPHAS, alpha_name

CONFIG = 'svib_preview_aggregate_v1'
METRICS = ['pixel_mse', 'image_squared_error_sum', 'identity_pixel_mse', 'foreground_pixel_mse',
           'changed_pixel_mse', 'unchanged_pixel_mse', 'pixel_mse_minus_identity', 'lower_pixel_mse_than_identity']
METHODS = ['identity', 'train_target_mean', 'plain', 'c4']
SPLITS = ['train', 'val', 'test_id', 'heldout']
CATEGORIES = ['all', 'changed', 'unchanged']
COMPARISONS = [('c4', 'plain'), ('plain', 'identity'), ('c4', 'identity'),
               ('plain', 'train_target_mean'), ('c4', 'train_target_mean')]


def summarize(values, mask):
    """Average initializations per scene, then bootstrap unique scenes only."""
    x = values[:, mask]
    finite = np.isfinite(x)
    assert (finite == finite[:1]).all(), 'Different finite cohorts require explicit handling'
    x = x[:, finite[0]]
    if not x.shape[1]:
        return {'mean': None, 'scene_ci95': None, 'per_seed_mean': [None] * len(x),
                'seed_sd': None, 'seed_min': None, 'seed_max': None,
                'finite_scenes': 0, 'total_scenes': int(mask.sum()), 'initializations': len(x)}
    scene = x.mean(0); seed = x.mean(1)
    rng = np.random.default_rng(9123)
    boot = scene[rng.integers(len(scene), size=(1000, len(scene)))].mean(1)
    return {'mean': float(scene.mean()), 'scene_ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'per_seed_mean': seed.tolist(), 'seed_sd': float(seed.std(ddof=1)) if len(seed) > 1 else None,
            'seed_min': float(seed.min()), 'seed_max': float(seed.max()),
            'finite_scenes': len(scene), 'total_scenes': int(mask.sum()), 'initializations': len(x)}


def preflight():
    x = np.array([[1., 2., np.nan, 4.], [3., 4., np.nan, 8.]])
    s = summarize(x, np.ones(4, bool))
    assert s['finite_scenes'] == 3 and s['total_scenes'] == 4 and s['initializations'] == 2
    np.testing.assert_allclose(s['mean'], 11/3)
    np.testing.assert_allclose(s['per_seed_mean'], [7/3, 5.])
    assert summarize(x, np.array([False, False, True, False]))['mean'] is None
    paired = summarize(x - np.array([[0., 1., np.nan, 2.]]), np.ones(4, bool))
    np.testing.assert_allclose(paired['mean'], 8/3)
    return {'all_passed': True, 'scene_then_seed_weighting_checked': True,
            'empty_group_and_paired_baseline_broadcast_checked': True, 'source_sha256': r.sha(__file__)}


def freeze():
    p = HERE / f'configs/{CONFIG}.json'
    if not p.exists():
        dump(p, {'alphas': ALPHAS, 'methods': METHODS, 'splits': SPLITS, 'categories': CATEGORIES,
                 'metrics': METRICS, 'comparisons_left_minus_right': COMPARISONS,
                 'aggregation': 'Equal weight across three fixed initializations per scene, then mean across unique scenes. Baselines have one deterministic result.',
                 'interval': '1000 bootstrap resamples of scene averages, RNG 9123. Conditional on this shared preview/test and fitted models; not independent-dataset uncertainty.',
                 'pairing': 'Subtract on identical scene and initialization; deterministic baselines broadcast. Empty change regions remain missing.',
                 'selection': 'All four alpha levels, all three initializations, all four partitions and both learned models; no result-based selection.',
                 'compute': 'Report actual recorded training seconds and all-100-image CPU forward times; no equal-compute or isolated timing claim.',
                 'scope': 'Small published preview only. No full SVIB, LPIPS or semantic Shape-Swap accuracy claim.',
                 'source_sha256': r.sha(__file__), 'model_config_sha256': r.sha(HERE / 'configs/svib_preview_models_v1.json'),
                 'preflight': preflight(), 'frozen_unix': time.time()})
    c = json.loads(p.read_text())
    assert c['source_sha256'] == r.sha(__file__)
    assert c['model_config_sha256'] == r.sha(HERE / 'configs/svib_preview_models_v1.json')
    return c


def gates():
    root = HERE / f'reports/{NAME}'; v = json.loads((root / 'model_verification.json').read_text())
    assert v['all_passed'] and v['models_verified'] == 24
    assert v['verifier_sha256'] == r.sha(HERE / 'verify_svib_preview_models_v2.py')
    assert v['summary_sha256'] == r.sha(root / 'model_summary.json')
    assert {(x['alpha'], x['kind'], x['seed']) for x in v['results']} == {(a, k, s) for a in ALPHAS for k in ['plain', 'c4'] for s in [0, 1, 2]}
    for item in v['results']:
        for file, sha in item['artifacts'].items(): assert r.sha(HERE / file) == sha
    b = json.loads((root / 'baselines/verification.json').read_text())
    assert b['all_passed'] and b['conditions_verified'] == 32
    assert b['summary_sha256'] == r.sha(root / 'baselines/summary.json')
    return {str(p.relative_to(HERE)): r.sha(p) for p in [root/'model_verification.json', root/'baselines/verification.json']}


def load_unit(alpha, method, split):
    seeds = [0, 1, 2] if method in ['plain', 'c4'] else [-1]
    values = []; sources = {}; first = None; cost = []
    for seed in seeds:
        if seed < 0:
            folder = HERE / f'reports/{NAME}/baselines/{alpha_name(alpha)}/{split}/{method}'
        else:
            base = HERE / f'runs/{NAME}/{alpha_name(alpha)}/{method}_s{seed}'; folder = base / split
            run = json.loads((base / 'run.json').read_text())
            sources[str((base / 'run.json').relative_to(HERE))] = r.sha(base / 'run.json')
        entry = json.loads((folder / 'metrics.json').read_text())
        for name, sha in entry['files'].items(): assert r.sha(folder / name) == sha
        rows = list(csv.DictReader((folder / 'rows.csv').open()))
        ids = np.array([int(x['episode']) for x in rows]); change = np.array([x['has_change'] == 'True' for x in rows])
        assert ids.tolist() == list(range(len(rows)))
        if first is None: first = (ids, change, entry['input_sha256'], entry['target_sha256'])
        else:
            np.testing.assert_array_equal(ids, first[0]); np.testing.assert_array_equal(change, first[1])
            assert (entry['input_sha256'], entry['target_sha256']) == first[2:]
        values.append(np.array([[np.nan if x[m] == '' else float(x[m] == 'True') if x[m] in ['True', 'False'] else float(x[m]) for m in METRICS] for x in rows]))
        for p in [folder/'rows.csv', folder/'metrics.json']: sources[str(p.relative_to(HERE))] = r.sha(p)
        if seed >= 0 and split == 'heldout':
            cost.append({'alpha': alpha, 'method': method, 'seed': seed, 'training_seconds': run['training_seconds'],
                         'cpu_forward_seconds_100_images': entry['cpu_forward_seconds'],
                         'parameters': run['parameters'], 'steps': run['steps'],
                         'rotation_mean_absolute_gap': entry['equivariance']['mean_absolute_gap'],
                         'rotation_maximum_absolute_gap': entry['equivariance']['maximum_absolute_gap']})
    return np.stack(values), first, seeds, sources, cost


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true'); args = parser.parse_args()
    c = freeze()
    if args.prepare_only:
        print('SVIB aggregation preflight passed and config frozen', flush=True); return
    evidence = gates(); root = HERE / f'reports/{NAME}/aggregate_v1'
    if root.exists(): raise RuntimeError(f'Preserve existing aggregate: {root}')
    root.mkdir(); entries = []; paired = []; manifests = []; costs = []
    for alpha in ALPHAS:
        for split in SPLITS:
            units = {}
            for method in METHODS:
                values, identity, seeds, sources, cost = load_unit(alpha, method, split); costs.extend(cost)
                units[method] = (values, identity)
                path = root / alpha_name(alpha) / split / method; path.mkdir(parents=True)
                np.savez_compressed(path / 'scenes.npz', values=values, episode=identity[0], has_change=identity[1])
                item = {'alpha': alpha, 'method': method, 'split': split, 'seeds': seeds, 'metrics': METRICS,
                        'values_axes': ['initialization', 'scene', 'metric'], 'input_sha256': identity[2], 'target_sha256': identity[3],
                        'sources': sources, 'files': {'scenes.npz': r.sha(path/'scenes.npz')}}
                dump(path/'manifest.json', item); manifests.append({'path': str((path/'manifest.json').relative_to(HERE)), 'sha256': r.sha(path/'manifest.json')})
                for category in CATEGORIES:
                    mask = np.ones(len(identity[0]), bool) if category == 'all' else identity[1] if category == 'changed' else ~identity[1]
                    for j, metric in enumerate(METRICS):
                        entries.append({'alpha': alpha, 'method': method, 'split': split, 'category': category, 'metric': metric,
                                        **summarize(values[..., j], mask)})
            for left, right in COMPARISONS:
                a, aid = units[left]; b, bid = units[right]
                assert aid[2:] == bid[2:]; np.testing.assert_array_equal(aid[0], bid[0]); np.testing.assert_array_equal(aid[1], bid[1])
                difference = a - b
                for category in CATEGORIES:
                    mask = np.ones(len(aid[0]), bool) if category == 'all' else aid[1] if category == 'changed' else ~aid[1]
                    for j, metric in enumerate(METRICS):
                        paired.append({'alpha': alpha, 'left': left, 'right': right, 'split': split, 'category': category, 'metric': metric,
                                       **summarize(difference[..., j], mask)})
    assert len(entries) == 1536 and len(paired) == 1920 and len(costs) == 24 and len(manifests) == 64
    dump(root/'summary.json', {'conditions': entries, 'paired_conditions': paired, 'costs': costs, 'scene_blocks': manifests,
         'condition_count': len(entries), 'paired_condition_count': len(paired), 'verification_evidence': evidence,
         'config_sha256': r.sha(HERE/f'configs/{CONFIG}.json'), 'scope': c['scope'], 'interval_scope': c['interval']})
    print('SVIB aggregate complete', len(entries), len(paired), flush=True)


if __name__ == '__main__': main()
