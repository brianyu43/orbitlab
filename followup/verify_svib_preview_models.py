"""Replay every preview prediction and independently audit training records and pixels."""
import argparse
import csv
import hashlib
import json
import time
import numpy as np
import torch
from common import HERE, dump, r
from svib_preview_data import NAME, alpha_name
from svib_preview_models import PreviewPredictor
from train_svib_preview import CONFIG, freeze


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def raw_images(address):
    folder = HERE / f'data/{NAME}/{address}'
    with np.load(folder / 'inputs.npz') as a:
        assert a.files == ['source_rgb']; source = a['source_rgb'].copy()
    with np.load(folder / 'labels.npz') as a:
        target = a['target_rgb'].copy()
        changed = np.any(source != target, axis=-1)
        foreground = np.any(source > 0, axis=-1) | np.any(target > 0, axis=-1)
        np.testing.assert_array_equal(changed, a['changed_mask'])
        np.testing.assert_array_equal(foreground, a['foreground_mask'])
    return (torch.from_numpy(source).permute(0, 3, 1, 2).float() / 255,
            torch.from_numpy(target).permute(0, 3, 1, 2).float() / 255,
            changed, foreground)


def check_rows_and_summary(folder, entry, prediction, source, target, changed, foreground):
    error = (prediction.astype(np.float64) - target.numpy().astype(np.float64)) ** 2
    identity = (source.numpy().astype(np.float64) - target.numpy().astype(np.float64)) ** 2
    rows = list(csv.DictReader((folder / 'rows.csv').open()))
    assert len(rows) == len(source) == entry['episodes']
    assert prediction.shape == source.shape and np.isfinite(prediction).all()
    assert prediction.min() >= 0 and prediction.max() <= 1
    reconstructed = []
    for i, row in enumerate(rows):
        values = {'episode': i, 'has_change': bool(changed[i].any()),
                  'changed_pixels': int(changed[i].sum()), 'foreground_pixels': int(foreground[i].sum()),
                  'pixel_mse': float(error[i].sum() / 49152),
                  'image_squared_error_sum': float(error[i].sum()),
                  'identity_pixel_mse': float(identity[i].sum() / 49152)}
        for region, mask in [('foreground', foreground[i]), ('changed', changed[i]), ('unchanged', ~changed[i])]:
            values[region + '_pixel_mse'] = float((error[i] * mask[None]).sum() / (3 * mask.sum())) if mask.any() else None
        values['pixel_mse_minus_identity'] = values['pixel_mse'] - values['identity_pixel_mse']
        values['lower_pixel_mse_than_identity'] = values['pixel_mse_minus_identity'] < -1e-12
        assert set(row) == set(values)
        for key, value in values.items():
            if value is None:
                assert row[key] == ''
            elif isinstance(value, bool):
                assert row[key] == str(value)
            else:
                np.testing.assert_allclose(float(row[key]), value, atol=1e-10, rtol=1e-12)
        reconstructed.append(values)
    checks = 0
    excluded = {'episode', 'has_change', 'changed_pixels', 'foreground_pixels'}
    assert set(entry['metrics']) == {'all', 'changed', 'unchanged'}
    for category, result in entry['metrics'].items():
        selected = [x for x in reconstructed if category == 'all' or x['has_change'] == (category == 'changed')]
        assert len(selected) == result['episodes']
        assert set(result['metrics']) == set(reconstructed[0]) - excluded
        for metric, stat in result['metrics'].items():
            values = np.array([x[metric] for x in selected if x[metric] is not None], dtype=float)
            assert stat['finite_episodes'] == len(values) and stat['total_episodes'] == len(selected)
            if not len(values):
                assert stat['mean'] is None and stat['base_scene_ci95'] is None
                continue
            generator = np.random.default_rng(9123)
            samples = values[generator.integers(len(values), size=(1000, len(values)))].mean(1)
            np.testing.assert_allclose(stat['mean'], values.mean(), atol=1e-10, rtol=1e-12)
            np.testing.assert_allclose(stat['base_scene_ci95'], np.percentile(samples, [2.5, 97.5]), atol=1e-10, rtol=1e-12)
            checks += 1
    return len(rows), checks


@torch.no_grad()
def verify_model(c, alpha, kind, seed):
    folder = HERE / f'runs/{NAME}/{alpha_name(alpha)}/{kind}_s{seed}'
    run = json.loads((folder / 'run.json').read_text())
    assert (run['alpha'], run['kind'], run['seed']) == (alpha, kind, seed)
    assert run['steps'] == c['steps'] == 2000
    assert run['config_sha256'] == r.sha(HERE / f'configs/{CONFIG}.json')
    assert run['checkpoint_sha256'] == r.sha(folder / 'model.pt')
    assert run['device'] in ['mps', 'cpu'] and run['training_seconds'] > 0
    assert [x['step'] for x in run['loss_log']] == [500, 1000, 1500, 2000]
    assert all(np.isfinite(x['loss']) and x['loss'] >= 0 and x['seconds'] > 0 for x in run['loss_log'])
    source, target, _, _ = raw_images(f'{alpha_name(alpha)}/train')
    assert len(source) == 70
    assert run['training_tensors_sha256'] == {'source': tensor_sha(source), 'target': tensor_sha(target)}
    torch.manual_seed(c['model_seed'] + seed)
    model = PreviewPredictor(kind)
    assert model_sha(model) == run['initial_weights_sha256']
    parameters = sum(x.numel() for x in model.parameters())
    assert parameters == run['parameters'] == 1266467
    generator = torch.Generator().manual_seed(c['sampling_seed'] + seed)
    sampling = hashlib.sha256()
    for _ in range(2000):
        indices = torch.randint(70, (8,), generator=generator)
        sampling.update(indices.numpy().tobytes())
    assert sampling.hexdigest() == run['sampling_sha256']
    saved = torch.load(folder / 'optimizer.pt', map_location='cpu', weights_only=True)
    assert torch.equal(saved['sampling_rng'], generator.get_state())
    optimizer = saved['optimizer']; assert len(optimizer['param_groups']) == 1
    group = optimizer['param_groups'][0]
    assert group['lr'] == .0005 and tuple(group['betas']) == (.9, .999)
    assert group['weight_decay'] == 0 and not group['amsgrad']
    assert len(optimizer['state']) == len(group['params']) == len(list(model.parameters()))
    for parameter, index in zip(model.parameters(), group['params']):
        state = optimizer['state'][index]
        assert int(state['step']) == 2000
        for key in ['exp_avg', 'exp_avg_sq']:
            assert state[key].shape == parameter.shape and torch.isfinite(state[key]).all()
        assert (state['exp_avg_sq'] >= 0).all()
    checkpoint = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
    assert checkpoint['kind'] == kind and all(torch.isfinite(x).all() for x in checkpoint['state_dict'].values())
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    evaluation = json.loads((folder / 'evaluation.json').read_text())
    assert evaluation['checkpoint_sha256'] == run['checkpoint_sha256'] and evaluation['config_sha256'] == run['config_sha256']
    assert [x['split'] for x in evaluation['results']] == ['train', 'val', 'test_id', 'heldout']
    instances = checks = rotations = 0
    gaps = []
    for split, n in [('train', 70), ('val', 15), ('test_id', 15), ('heldout', 100)]:
        out = folder / split; entry = json.loads((out / 'metrics.json').read_text())
        assert entry == next(x for x in evaluation['results'] if x['split'] == split)
        assert (entry['alpha'], entry['kind'], entry['seed'], entry['episodes']) == (alpha, kind, seed, n)
        address = 'heldout' if split == 'heldout' else f'{alpha_name(alpha)}/{split}'
        assert entry['address'] == address and entry['checkpoint_sha256'] == run['checkpoint_sha256']
        assert entry['config_sha256'] == run['config_sha256']
        assert entry['input_sha256'] == r.sha(HERE / f'data/{NAME}/{address}/inputs.npz')
        assert entry['target_sha256'] == r.sha(HERE / f'data/{NAME}/{address}/labels.npz')
        for name, sha in entry['files'].items():
            assert r.sha(out / name) == sha
        source, target, changed, foreground = raw_images(address)
        pred = torch.cat([model(source[i:i+16]) for i in range(0, n, 16)])
        with np.load(out / 'predictions.npz') as a:
            assert a.files == ['predictions']; stored = a['predictions'].copy()
        np.testing.assert_allclose(stored, pred.numpy(), atol=2e-7, rtol=1e-6)
        nrows, nchecks = check_rows_and_summary(out, entry, stored, source, target, changed, foreground)
        instances += nrows; checks += nchecks
        if kind == 'c4':
            # Evaluate the group average with separate rotations, independently of forward's concatenation.
            ref = sum(torch.rot90(model.direct(torch.rot90(source[:7], k, (-2, -1))), -k, (-2, -1)) for k in range(4)) / 4
            torch.testing.assert_close(ref, pred[:7], atol=2e-6, rtol=1e-6)
        if split == 'heldout':
            assert changed.any(axis=(1, 2)).sum() == 74
            ma, mx = [], []
            for k in [1, 2, 3]:
                rotated = torch.cat([model(torch.rot90(source[i:i+16], k, (-2, -1))) for i in range(0, n, 16)])
                error = (rotated - torch.rot90(pred, k, (-2, -1))).abs().flatten(1)
                ma.append(error.mean(1).numpy()); mx.append(error.max(1).values.numpy())
            ma, mx = np.stack(ma, axis=1), np.stack(mx, axis=1)
            with np.load(out / 'equivariance.npz') as a:
                np.testing.assert_allclose(a['mae'], ma, atol=1e-8, rtol=1e-5)
                np.testing.assert_allclose(a['max_abs'], mx, atol=2e-7, rtol=1e-5)
            assert entry['equivariance']['episodes'] == 100 and entry['equivariance']['rotations'] == [1, 2, 3]
            np.testing.assert_allclose(entry['equivariance']['mean_absolute_gap'], ma.mean(), atol=1e-8, rtol=1e-5)
            np.testing.assert_allclose(entry['equivariance']['maximum_absolute_gap'], mx.max(), atol=2e-7, rtol=1e-5)
            if kind == 'c4': assert mx.max() <= 2e-6
            rotations += 300; gaps.append(float(mx.max()))
        else:
            assert entry['equivariance'] is None
    assert instances == 200 and rotations == 300
    artifacts = {str(p.relative_to(HERE)): r.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    return {'all_passed': True, 'alpha': alpha, 'kind': kind, 'seed': seed,
            'predictions_replayed': instances, 'rotation_predictions_replayed': rotations, 'aggregate_interval_checks': checks,
            'initial_weights_sha256': run['initial_weights_sha256'], 'sampling_sha256': run['sampling_sha256'],
            'checkpoint_sha256': run['checkpoint_sha256'], 'parameters': parameters, 'maximum_rotation_gap': max(gaps),
            'artifacts': artifacts, 'verifier_sha256': r.sha(__file__), 'config_sha256': run['config_sha256']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--available', action='store_true'); args = parser.parse_args()
    c = freeze(); torch.set_num_threads(2); root = HERE / f'reports/{NAME}'
    results = []; started = time.time()
    for alpha in c['alphas']:
        for seed in c['seeds']:
            for kind in c['kinds']:
                folder = HERE / f'runs/{NAME}/{alpha_name(alpha)}/{kind}_s{seed}'
                if args.available and not (folder / 'evaluation.json').exists(): continue
                receipt = root / f'model_receipts/{alpha_name(alpha)}_{kind}_s{seed}.json'
                if receipt.exists():
                    result = json.loads(receipt.read_text())
                    assert result['all_passed'] and result['verifier_sha256'] == r.sha(__file__)
                    assert result['config_sha256'] == r.sha(HERE / f'configs/{CONFIG}.json')
                    assert (result['alpha'], result['kind'], result['seed']) == (alpha, kind, seed)
                    for file, sha in result['artifacts'].items(): assert r.sha(HERE / file) == sha
                else:
                    result = verify_model(c, alpha, kind, seed); dump(receipt, result)
                results.append(result)
                dump(root / 'model_verification_progress.json', {'models_verified': len(results), 'expected_models': 24, 'seconds': time.time()-started})
                print('verified SVIB model', alpha, kind, seed, len(results), '/24', flush=True)
    comparisons = []
    for alpha in c['alphas']:
        for seed in c['seeds']:
            pair = [x for x in results if x['alpha'] == alpha and x['seed'] == seed]
            if len(pair) == 2:
                for key in ['initial_weights_sha256', 'sampling_sha256', 'parameters']: assert pair[0][key] == pair[1][key]
                comparisons.append([alpha, seed])
    full = len(results) == 24; summary_sha = None
    if full:
        path = root / 'model_summary.json'; summary = json.loads(path.read_text())
        assert summary['models'] == len(summary['results']) == 24 and summary['evaluated_prediction_instances'] == 4800
        assert summary['config_sha256'] == r.sha(HERE / f'configs/{CONFIG}.json')
        for item in summary['results']:
            folder = HERE / f"runs/{NAME}/{alpha_name(item['alpha'])}/{item['kind']}_s{item['seed']}"
            assert item['run'] == json.loads((folder / 'run.json').read_text())
            assert item['evaluation'] == json.loads((folder / 'evaluation.json').read_text())['results']
        assert len(comparisons) == 12; summary_sha = r.sha(path)
    dump(root / 'model_verification.json', {'all_passed': full, 'all_available_checked_passed': True,
        'models_verified': len(results), 'expected_models': 24,
        'predictions_replayed': sum(x['predictions_replayed'] for x in results),
        'rotation_predictions_replayed': sum(x['rotation_predictions_replayed'] for x in results),
        'aggregate_interval_checks': sum(x['aggregate_interval_checks'] for x in results),
        'paired_initialization_sampling_checks': comparisons, 'summary_sha256': summary_sha,
        'verifier_sha256': r.sha(__file__), 'results': results,
        'scope': 'Checkpoint inference replay, raw-pixel scoring, optimizer final state/step and deterministic sampling audits. Not a full rerun of MPS training; one shared public test, not independent datasets.'})
    print('SVIB model audit', len(results), 'full', full, flush=True)


if __name__ == '__main__': main()
