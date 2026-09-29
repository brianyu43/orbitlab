"""Recompute complete external-result accounting without hiding pending cells.

This consumes frozen runs and verified inference artifacts. It never changes
model/checkpoint selection, thresholds, or official test membership.
"""
import argparse
import json
import time
import numpy as np
import torch
from v3_common import HERE, dump, lock, sha
from r5_intake import BASE, TASKS
from r5_attribute_evaluate import backend, measurements
from r5_attributes import (folder as reader_folder, schema, load_factors, store_for,
                           summary as attribute_summary, freeze as reader_freeze,
                           Reader, FAMILIES, audit_folder)
from r4_core import state_hash
from r5_models import pixel_metrics
from r5_train import aggregate_rows
from r5_attribute_calibration_diagnostic import evaluate as calibration_diagnostic

TASK_PAIRS = [('dsprites', task) for task in TASKS] + [('clevr', 'Single_Atomic'), ('clevrtex', 'Single_Atomic')]


def same(a, b, tolerance=1e-6):
    assert a.keys() == b.keys()
    for key, value in a.items():
        if value is None:
            assert b[key] is None
        else:
            assert abs(value - b[key]) <= tolerance, (key, value, b[key])


def check_file(path, expected):
    assert sha(path) == expected, f'Artifact hash changed: {path}'


def run_record(family, task, kind, init, seed, pending):
    mod = backend(family)
    tr = mod.training
    tr.SPLIT_SEED = seed
    path = tr.location(task, kind, init)
    key = f'{family}/{task}/p{seed}/{kind}_s{init}'
    metadata = {'key': key, 'family': family, 'task': task, 'kind': kind,
                'initialization': init, 'partition_seed': seed}
    required = ['run.json', 'verification.json', 'verification_protocol.json']
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        pending.append({'item': key, 'required': missing, 'type': 'training_or_full_validation_replay'})
        progress = path / 'progress.json'
        return {**metadata, 'complete': False, 'progress': json.loads(progress.read_text()) if progress.exists() else None}
    run = json.loads((path / 'run.json').read_text())
    verification = json.loads((path / 'verification.json').read_text())
    protocol = json.loads((path / 'verification_protocol.json').read_text())
    check_file(path / 'run.json', protocol['run_sha256'])
    check_file(path / 'verification_protocol.json', verification['protocol_sha256'])
    assert run['context'] == tr.context(task, kind, init) == protocol['context']
    assert run['epochs'] == 20 and run['steps'] == 36000 and run['official_test_used'] is False
    assert verification['selected_epoch_recomputed'] == run['selected_epoch']
    assert verification['optimizer_parameter_step_counts']['min'] == verification['optimizer_parameter_step_counts']['max'] == 36000
    assert verification['initial_weights_recomputed'] and verification['all20epoch_exampleorders_recomputed'] and verification['final_weights_match_progress_bitexact']
    replayed = {v['epoch']: v for v in verification['full_validation_replays']}
    validation = {}
    for epoch in (5, 10, 20):
        check_file(path / f'epoch{epoch}.pt', run['checkpoints'][str(epoch)])
        p = path / f'val_epoch{epoch}.json'
        check_file(p, run['validation'][str(epoch)])
        v = json.loads(p.read_text())
        check_file(path / f'val_epoch{epoch}.npy', v['rows_sha256'])
        rows = np.load(path / f'val_epoch{epoch}.npy')
        assert rows.shape == (6400, 6) and np.isfinite(rows).all()
        actual = aggregate_rows(rows)
        same(actual, v['metrics'])
        assert replayed[epoch]['all_validation_forwards_repeated'] == replayed[epoch]['all_metricrows_compared'] == 6400
        validation[str(epoch)] = actual
    selected = min((5, 10, 20), key=lambda ep: (validation[str(ep)]['image_mse'], ep))
    assert selected == run['selected_epoch']
    return {**metadata,
            'complete': True, 'run_sha256': sha(path / 'run.json'), 'verification_sha256': sha(path / 'verification.json'),
            'parameters': run['parameters'], 'full_updates': run['steps'], 'selected_epoch': selected, 'validation': validation,
            'training_seconds': run['seconds'], 'validation_seconds': run['validation_seconds'],
            'scheduling_amendment': str((path / 'gpu_schedule_amendment.json').relative_to(HERE)) if (path / 'gpu_schedule_amendment.json').exists() else None}


def check_selection(family, task, runs, pending):
    path = BASE / 'external' / f'{family}_hard' / 'evaluation' / task / 'development_selection.json'
    if not path.exists() or not all(v['complete'] for v in runs):
        pending.append({'item': f'{family}/{task}', 'type': 'validation_only_selection', 'required': 'Allthreeverifieddevelopmentarms pluslockedselection'})
        return None
    selection = json.loads(path.read_text())
    assert selection['official_test_used'] is False
    bykind = {v['kind']: v for v in runs}
    for kind, run in bykind.items():
        assert selection['runs'][kind] == run['run_sha256'] and selection['selected_epochs'][kind] == run['selected_epoch']
        same(selection['validation_metrics'][kind], run['validation'][str(run['selected_epoch'])])
    # Recompute the complete input-copy validation control on CPU from original
    #PNGbytes. This checksselection independentlyofstoredmean values, without
    #readingofficialtest orchangingtheoriginalGPUselectionrecord.
    dest = BASE / 'reporting_copy_validation' / family / task
    mod = backend(family)
    mod.training.SPLIT_SEED = 890101
    ids = mod.training.partition(task)['val']
    context = {'aggregate_source_sha256': sha(__file__), 'selection_sha256': sha(path), 'partition_sha256': sha(mod.training.prepare_partition(task, 890101))}
    lock(dest / 'protocol.json', context)
    if (dest / 'verification.json').exists():
        rec = json.loads((dest / 'verification.json').read_text())
        check_file(dest / 'protocol.json', rec['protocol_sha256'])
        check_file(dest / 'rows.npy', rec['rows_sha256'])
        rows = np.load(dest / 'rows.npy')
    else:
        torch.set_num_threads(2)
        store = store_for(family, task)
        output = []
        for first in range(0, 6400, 32):
            x, y = store.batch('train', ids[first:first + 32], torch.device('cpu'))
            output.append(pixel_metrics(x, x, y).numpy())
        store.close()
        rows = np.concatenate(output)
        np.save(dest / 'rows.npy', rows)
        dump(dest / 'verification.json', {'protocol_sha256': sha(dest / 'protocol.json'), 'rows_sha256': sha(dest / 'rows.npy'), 'all_validation_copy_forwards': 6400})
    copy = aggregate_rows(rows)
    same(copy, selection['copy_validation'])
    baseline = selection['validation_metrics']['plain']
    eligible, gates = [], []
    for kind in ('c4', 'object'):
        v = selection['validation_metrics'][kind]
        tests = {'image_better_than_plain_and_copy': v['image_mse'] < min(baseline['image_mse'], copy['image_mse']),
                 'changed_better_than_plain_and_copy': v['changed_scene_changed_pixel_mse'] < min(baseline['changed_scene_changed_pixel_mse'], copy['changed_scene_changed_pixel_mse']),
                 'preserved_damage_not_over5percent_worse': v['preserved_pixel_mse'] <= 1.05 * baseline['preserved_pixel_mse']}
        passed = all(tests.values())
        gates.append({'candidate': kind, 'gates': tests, 'passed': passed})
        if passed:
            eligible.append(kind)
    chosen = min(eligible, key=lambda kind: selection['validation_metrics'][kind]['image_mse']) if eligible else None
    assert gates == selection['gates'] and chosen == selection['selected_candidate']
    return {'selection_sha256': sha(path), 'selected_candidate': chosen, 'selected_epochs': selection['selected_epochs'],
            'validation_metrics': selection['validation_metrics'], 'copy_validation': copy, 'gates': gates,
            'copy_validation_recomputed_from_all6400images': True, 'test_used': False}


def reader_record(family, task, pending):
    root = reader_folder(family, task)
    needed = ['run.json', 'reader.pt', 'progress.pt', 'protocol.json'] + [f'calibration_{split}{suffix}' for split in ('val', 'test') for suffix in ('.json', '_verification.json', '.npz')]
    if any(not (root / name).exists() for name in needed):
        pending.append({'item': f'{family}/{task}/attribute_reader', 'type': 'full_clean_calibration', 'required': [name for name in needed if not (root / name).exists()]})
        return None
    reader_freeze(family, task)
    names, counts = schema(family)
    run = json.loads((root / 'run.json').read_text())
    check_file(root / 'reader.pt', run['checkpoint_sha256'])
    check_file(root / 'protocol.json', run['context']['protocol_sha256'])
    assert run['steps'] == 6000 and run['batch'] == 64 and run['no_model_predictions_used']
    seed = 960000 + FAMILIES.index(family) * 100 + list(TASKS).index(task)
    torch.manual_seed(seed)
    model = Reader(counts)
    assert state_hash(model) == run['context']['initial_state_sha256']
    assert sum(v.numel() for v in model.parameters()) == run['parameters']
    final = torch.load(root / 'reader.pt', map_location='cpu', weights_only=True)
    progress = torch.load(root / 'progress.pt', map_location='cpu', weights_only=True)
    assert final['context'] == progress['context'] == run['context']
    assert final['steps'] == progress['step'] == 6000
    step_counts = [int(v['step']) for v in progress['optimizer']['state'].values() if 'step' in v]
    assert step_counts and min(step_counts) == max(step_counts) == 6000
    for name, tensor in final['state_dict'].items():
        torch.testing.assert_close(tensor, progress['state_dict'][name], rtol=0, atol=0)
    partition = np.load(audit_folder(family, task) / 'partition.npz')
    train_ids = partition['train']
    rng = np.random.default_rng(seed + 900)
    for _ in range(6000):
        rng.choice(train_ids, 64)
        rng.integers(0, 2, 64)
        rng.integers(0, 2, 64)
    assert rng.bit_generator.state == progress['numpy_rng']
    factors, _ = load_factors(family, task)
    values = {}
    for split in ('val', 'test'):
        p = root / f'calibration_{split}.json'
        cal = json.loads(p.read_text())
        ver = json.loads((root / f'calibration_{split}_verification.json').read_text())
        check_file(p, ver['summary_sha256'])
        check_file(root / 'reader.pt', cal['checkpoint_sha256'])
        check_file(root / f'calibration_{split}.npz', cal['predictions_sha256'])
        z = np.load(root / f'calibration_{split}.npz')
        n = 6400 if split == 'val' else 8000
        assert z['prediction'].shape == z['truth'].shape == (n * 4, len(names))
        assert ver['all_rows_bitexact'] and ver['all_object_predictions_replayed'] == n * 4
        assert attribute_summary(z['prediction'], z['truth'], names) == cal['all']
        ids = partition['val'] if split == 'val' else np.arange(64000, 72000)
        items = np.array([(i, r, o) for i in ids for r in (0, 1) for o in (0, 1)])
        np.testing.assert_array_equal(items, z['items'])
        np.testing.assert_array_equal(z['truth'], factors[items[:, 0], items[:, 1], items[:, 2]])
        for role, label in enumerate(('source', 'target')):
            mask = items[:, 1] == role
            assert attribute_summary(z['prediction'][mask], z['truth'][mask], names) == cal[label]
        eq = (z['prediction'] == z['truth']).reshape(n, 2, 2, len(names))[:, 1]
        assert float(eq.all((1, 2)).mean()) == cal['target_both_objects_all_attributes_accuracy']
        values[split] = {'target': cal['target'], 'source': cal['source'], 'target_both_objects_all_attributes_accuracy': cal['target_both_objects_all_attributes_accuracy'],
                         'calibration_sha256': sha(p), 'verification_sha256': sha(root / f'calibration_{split}_verification.json')}
    return {'attribute_names': names, 'validation': values['val'], 'official_test': values['test'], 'perfect_semantic_oracle': False,
            'run_sha256': sha(root / 'run.json'), 'parameters': run['parameters'], 'full_updates': run['steps'],
            'training_seconds': run['seconds'], 'optimizer_steps_verified': 6000,
            'initialization_and_sampling_RNG_recomputed': True, 'final_weights_match_progress_bitexact': True,
            'privileged_source_center': True, 'all_generated_scores_require_calibration_caveat': True}


def unit_record(family, task, kind, init, mode, seed, pending):
    mod = backend(family)
    mod.training.SPLIT_SEED = seed
    root = mod.unit(task, kind, init, mode)
    key = f'{family}/{task}/p{seed}/{kind}_s{init}/{mode}'
    metadata = {'key': key, 'family': family, 'task': task, 'kind': kind,
                'initialization': init, 'partition_seed': seed, 'mode': mode}
    if not (root / 'summary.json').exists() or not (root / 'verification.json').exists():
        pending.append({'item': key, 'type': 'all_test_pixels_and_replay', 'required': '8000predictionsandfullverification'})
        return {**metadata, 'complete': False, 'pixels_complete': False, 'attributes_complete': False}
    rec = json.loads((root / 'summary.json').read_text())
    ver = json.loads((root / 'verification.json').read_text())
    check_file(root / 'summary.json', ver['summary_sha256'])
    assert rec['provenance'] == mod.provenance(task, kind, init, mode)
    for name, keyname in [('rows.npy', 'rows_sha256'), ('batch_hashes.json', 'batch_hashes_sha256'), ('examples.pt', 'examples_sha256')]:
        check_file(root / name, rec[keyname])
    rows = np.load(root / 'rows.npy')
    assert rows.shape == (8000, 6) and np.isfinite(rows).all()
    metrics = aggregate_rows(rows)
    same(metrics, rec['metrics'])
    assert ver['all_RGBoutputs_replayed'] == ver['all_metricrows_recomputed'] == 8000
    assert ver['max_abs_pixel_error'] <= 1e-6 and ver['total_batches'] == 250
    assert ver['original_temporary_cache_sha256'] == rec['cache_sha256'] and ver['temporary_cache_removed_after_pass']
    assert not (root / 'prediction_verification_cache.npy').exists()
    value = {**metadata,
             'selected_epoch': rec['provenance']['epoch'], 'pixels_complete': True, 'attributes_complete': False, 'complete': False,
             'pixel_metrics': metrics, 'pixel_summary_sha256': sha(root / 'summary.json'), 'pixel_verification_sha256': sha(root / 'verification.json')}
    attrs = root / 'attributes'
    if not (attrs / 'summary.json').exists() or not (attrs / 'verification.json').exists():
        pending.append({'item': key, 'type': 'all_object_readouts_and_replay', 'required': '16000objectreadouts and calibrationdiagnostic'})
        return value
    av = json.loads((attrs / 'summary.json').read_text())
    vv = json.loads((attrs / 'verification.json').read_text())
    check_file(attrs / 'summary.json', vv['summary_sha256'])
    check_file(attrs / 'protocol.json', av['protocol_sha256'])
    check_file(attrs / 'rows.npz', av['rows_sha256'])
    assert vv['all_generated_images_replayed'] == 8000 and vv['all_object_readouts_replayed'] == 16000 and vv['input_batches_bitexact'] == 250
    z = np.load(attrs / 'rows.npz')
    names, _ = schema(family)
    factors, _ = load_factors(family, task)
    assert z['prediction'].shape == (8000, 2, len(names))
    np.testing.assert_array_equal(z['truth'], factors[64000:, 1])
    np.testing.assert_array_equal(z['source'], factors[64000:, 0])
    computed = measurements(z['prediction'], z['truth'], z['source'], z['clean_readout'], names)
    assert computed == av['metrics']
    diagnostic = calibration_diagnostic(family, task, kind, mode, init, seed)
    semantic_unchanged = np.all(z['source'] == z['truth'], axis=(1, 2))
    semantic_groups = {name: aggregate_rows(rows[mask]) if mask.any() else None for name, mask in [('unchanged', semantic_unchanged), ('changed', ~semantic_unchanged)]}
    value.update(attributes_complete=True, complete=True, attribute_metrics=computed, clean_calibration_diagnostic=diagnostic['groups'],
                 semantic_scene_groups=semantic_groups, attribute_summary_sha256=sha(attrs / 'summary.json'), attribute_verification_sha256=sha(attrs / 'verification.json'),
                 diagnostic_sha256=sha(attrs / 'clean_calibration_diagnostic.json'))
    return value


def collect(require_complete=False):
    torch.set_num_threads(2)
    pending, tasks, confirmation = [], [], []
    for family, task in TASK_PAIRS:
        runs = [run_record(family, task, kind, 0, 890101, pending) for kind in ('plain', 'c4', 'object')]
        selection = check_selection(family, task, runs, pending)
        reader = reader_record(family, task, pending)
        units = []
        for kind in ('copy', 'plain', 'c4', 'object'):
            for mode in (['validation_selected'] if kind == 'copy' else ['validation_selected', 'fixed20epoch']):
                units.append(unit_record(family, task, kind, 0, mode, 890101, pending))
        tasks.append({'family': family, 'task': task, 'runs': runs, 'selection': selection, 'reader': reader, 'test_units': units})
        if selection is not None and selection['selected_candidate'] is not None:
            candidate = selection['selected_candidate']
            for seed in (890201, 890202, 890203):
                for init in (0, 1, 2):
                    for kind in ('plain', candidate):
                        run = run_record(family, task, kind, init, seed, pending)
                        units = [unit_record(family, task, kind, init, mode, seed, pending) for mode in ('validation_selected', 'fixed20epoch')]
                        confirmation.append({'family': family, 'task': task, 'run': run, 'test_units': units})
    result = {'aggregate_source_sha256': sha(__file__), 'snapshot_unix': time.time(), 'tasks': tasks, 'conditional_confirmation': confirmation,
              'pending': pending, 'external_scope_complete': not pending, 'independent_human_evaluation': False,
              'primary_development_runs_required': 18, 'primary_development_runs_verified': sum(r['complete'] for t in tasks for r in t['runs']),
              'primary_development_pixel_units_required': 42, 'primary_development_pixel_units_verified': sum(v['pixels_complete'] for t in tasks for v in t['test_units']),
              'primary_development_attribute_units_verified': sum(v['attributes_complete'] for t in tasks for v in t['test_units']),
              'confirmation_runs_required_from_available_selections': len(confirmation),
              'confirmation_runs_verified': sum(v['run']['complete'] for v in confirmation),
              'scope': 'Everyrequiredtask/arm/mode/resplit/init retained;missingcells arenotaveragedaway. Confirmationsarereshuffledpartitionsofsameofficialdata/test,notnewindependentdatasets. ScopecompletehereisexternalR5only,notallresearchorhumanevaluation.'}
    destination = BASE / 'external_aggregate.json' if require_complete else BASE / 'external_scope_snapshot.json'
    if require_complete and pending:
        dump(BASE / 'external_incomplete_scope.json', result)
        raise RuntimeError(f'{len(pending)} required external items pending; cannot finalize')
    dump(destination, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    result = collect(args.require_complete)
    print(json.dumps({k: v for k, v in result.items() if k not in ('tasks', 'conditional_confirmation', 'pending')}, indent=2))
    print('Pendingitems:', len(result['pending']))
