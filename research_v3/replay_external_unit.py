"""Explicit fresh forward replay; original results and receipts stay unchanged.

Unlike resumable experiment drivers, --execute never skips inference because an
old verification receipt exists. Default mode prints a plan without GPU work.
This reuses the original model/evaluator implementation, not independent training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from v3_common import HERE, dump, sha
from r5_intake import TASKS
from r5_attribute_evaluate import backend
from r5_attributes import store_for
from r5_models import pixel_metrics


def checked_json(path, checked):
    checked[str(path.relative_to(HERE))] = sha(path)
    return json.loads(path.read_text())


@torch.no_grad()
def replay(family, task, kind, mode, seed, init, output, execute):
    assert family == 'dsprites' or task == 'Single_Atomic'
    assert kind != 'copy' or mode == 'validation_selected'
    mod = backend(family)
    mod.training.SPLIT_SEED = seed
    root = mod.unit(task, kind, init, mode)
    output = output.resolve()
    assert output.is_relative_to(HERE / 'review/replays'), 'Keep new replay receipts in research_v3/review/replays/'
    assert output.suffix == '.json' and not output.exists(), 'Choose a new receipt filename'
    checked = {}
    summary = checked_json(root / 'summary.json', checked)
    verification = checked_json(root / 'verification.json', checked)
    assert verification['summary_sha256'] == sha(root / 'summary.json')
    assert verification['all_RGBoutputs_replayed'] == 8000
    for name, key in [('rows.npy', 'rows_sha256'), ('batch_hashes.json', 'batch_hashes_sha256')]:
        assert sha(root / name) == summary[key]
        checked[str((root / name).relative_to(HERE))] = summary[key]
    # Existing protocol/partition paths must be present before provenance calls,
    # so the original freeze helpers cannot silently initialize a new experiment.
    protocol = HERE / 'r5' / f'{family}_training_protocol.json'
    assert protocol.exists()
    partition = (HERE / 'r5/audits' / f'{family}_hard' / task / 'partition.npz' if seed == 890101
                 else HERE / 'r5/partitions' / f'{family}_hard' / task / f's{seed}' / 'partition.npz')
    assert partition.exists()
    if seed != 890101:
        assert (partition.parent / 'receipt.json').exists()
    checked[str(protocol.relative_to(HERE))] = sha(protocol)
    checked[str(partition.relative_to(HERE))] = sha(partition)
    mod.training.freeze()
    provenance = mod.provenance(task, kind, init, mode)
    assert provenance == summary['provenance']
    if kind != 'copy':
        checkpoint = mod.training.location(task, kind, init) / f'epoch{provenance["epoch"]}.pt'
        checked[str(checkpoint.relative_to(HERE))] = provenance['checkpoint_sha256']
    spec = {'source_sha256': sha(__file__), 'family': family, 'task': task, 'kind': kind,
            'mode': mode, 'partition_seed': seed, 'initialization': init,
            'official_test_images': 8000, 'batch_size': 32, 'device': 'mps', 'precision': 'float32',
            'metric_row_tolerance_atol': 1e-6, 'prediction_batch_hash_tolerance': 0,
            'output': str(output.relative_to(HERE)), 'original_evidence': checked,
            'scope': 'Fresh full test forward pass from frozen weights; exact original float32 prediction hashes and all metric rows. Reuses original implementation; no retraining, selector changes, or independent human validation.'}
    if not execute:
        print(json.dumps(spec, indent=2))
        return spec
    assert torch.backends.mps.is_available(), 'Original test outputs require MPS; do not silently switch precision/device'
    torch.set_num_threads(2)
    expected_hashes = json.loads((root / 'batch_hashes.json').read_text())
    saved_rows = np.load(root / 'rows.npy')
    assert len(expected_hashes) == 250 and saved_rows.shape == (8000, 6)
    device = torch.device('mps')
    model = mod.load(task, kind, init, mode, device)
    store = store_for(family, task)
    rows = []
    started = time.perf_counter()
    try:
        for first in range(0, 8000, 32):
            ids = np.arange(first, first + 32)
            source, target = store.batch('test', ids, device)
            prediction = source if kind == 'copy' else model(source)
            digest = hashlib.sha256(prediction.cpu().numpy().tobytes()).hexdigest()
            assert digest == expected_hashes[first // 32], f'Prediction mismatch at test batch {first // 32}; keep original device/batch/precision'
            values = pixel_metrics(prediction, source, target).cpu().numpy()
            np.testing.assert_allclose(values, saved_rows[first:first + 32], rtol=0, atol=1e-6)
            rows.append(values)
    finally:
        store.close()
    actual_rows = np.concatenate(rows)
    metrics = mod.training.aggregate_rows(actual_rows)
    for key, expected in summary['metrics'].items():
        assert metrics[key] is None if expected is None else abs(metrics[key] - expected) <= 1e-6, key
    for path, digest in checked.items():
        assert sha(HERE / path) == digest, f'Original evidence changed while replaying: {path}'
    result = {**spec, 'executed': True, 'all_250_prediction_batches_bitexact': True,
              'metric_rows_recomputed': 8000, 'max_abs_metric_difference': float(np.max(np.abs(actual_rows - saved_rows))),
              'metrics': metrics, 'original_checked_evidence_unchanged': True,
              'seconds': time.perf_counter() - started, 'completed_unix': time.time()}
    dump(output, result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('dsprites', 'clevr', 'clevrtex'), required=True)
    parser.add_argument('--task', choices=TASKS, default='Single_Atomic')
    parser.add_argument('--kind', choices=('copy', 'plain', 'c4', 'object'), required=True)
    parser.add_argument('--mode', choices=('validation_selected', 'fixed20epoch'), default='validation_selected')
    parser.add_argument('--split-seed', type=int, choices=(890101, 890201, 890202, 890203), default=890101)
    parser.add_argument('--init', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    replay(args.family, args.task, args.kind, args.mode, args.split_seed, args.init, args.output, args.execute)
