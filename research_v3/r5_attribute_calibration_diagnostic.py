"""Fixed clean-reader reliability subsets; never replaces full-test scores.

Subset membership depends only on original clean targets and their metadata,
and is identical for all generators. This is an easier selected subset, not an
error correction or unbiased semantic accuracy estimate.
"""
import argparse
import json
import numpy as np
from v3_common import HERE, dump, lock, sha
from r5_intake import BASE, TASKS
from r5_attributes import schema, folder as reader_folder, load_factors
from r5_attribute_evaluate import backend


def freeze():
    return lock(BASE / 'attribute_calibration_diagnostic_protocol.json', {
        'source_sha256': sha(__file__), 'reader_source_sha256': sha(HERE / 'r5_attributes.py'),
        'scope': 'Keepfull8000test/rawattributes. Additionalfixedclean-target-correctobjectandboth-objectscenesubsets, pluscomplement, samefor everygenerator/checkpoint. Per-attribute clean-correctsubsets alsoreported.',
        'membership': 'Frozenreader classifications onORIGINALcleantarget comparedwithtruemetadata. Neverusesgeneratedoutputs forinclusion. No readerretraining ormodelselection.',
        'interpretation': 'Conditionalperformanceonexplicitlyeasierselectedsubsets,notpopulationaccuracy,measurementerrorcorrectionorproofthatreaderrecognizesgeneratedartifacts. Reportcounts,coverageandclass/rulebias. Failedcleanreadingsretainedinfulltest andcomplement.',
    })


def prepare(family, task):
    freeze()
    root = reader_folder(family, task)
    cal = json.loads((root / 'calibration_test.json').read_text())
    ver = json.loads((root / 'calibration_test_verification.json').read_text())
    assert ver['summary_sha256'] == sha(root / 'calibration_test.json') and ver['all_rows_bitexact']
    assert cal['predictions_sha256'] == sha(root / 'calibration_test.npz')
    names, counts = schema(family)
    labels, _ = load_factors(family, task)
    truth = labels[64000:, 1]
    clean = np.load(root / 'calibration_test.npz')['prediction'].reshape(8000, 2, 2, len(names))[:, 1]
    correct = clean == truth
    objects = correct.all(-1)
    scenes = objects.all(-1)
    destination = root / 'clean_reliability_subsets'
    destination.mkdir(exist_ok=True)
    context = {'diagnostic_protocol_sha256': sha(BASE / 'attribute_calibration_diagnostic_protocol.json'),
               'calibration_sha256': sha(root / 'calibration_test.json'), 'calibration_verification_sha256': sha(root / 'calibration_test_verification.json')}
    lock(destination / 'context.json', context)
    arrays = {'attribute_clean_correct': correct, 'object_clean_correct': objects, 'scene_clean_correct': scenes}
    if (destination / 'masks.npz').exists():
        old = np.load(destination / 'masks.npz')
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, old[name])
        return destination
    np.savez_compressed(destination / 'masks.npz', **arrays)
    distributions = {}
    for i, (name, count) in enumerate(zip(names, counts)):
        distributions[name] = {'all_objects': np.bincount(truth[..., i].ravel(), minlength=count).tolist(),
                               'clean_correct_objects': np.bincount(truth[..., i][objects], minlength=count).tolist(),
                               'both_object_clean_correct_scenes': np.bincount(truth[scenes, :, i].ravel(), minlength=count).tolist()}
    dump(destination / 'summary.json', {'context': context, 'masks_sha256': sha(destination / 'masks.npz'),
                                       'objects_total': 16000, 'objects_clean_correct': int(objects.sum()),
                                       'scenes_total': 8000, 'scenes_both_objects_clean_correct': int(scenes.sum()),
                                       'attribute_class_counts': distributions, 'full_test_replaced': False})
    return destination


def evaluate(family, task, kind, mode, init=0, split_seed=890101):
    subset = prepare(family, task)
    mod = backend(family)
    mod.training.SPLIT_SEED = split_seed
    root = mod.unit(task, kind, init, mode) / 'attributes'
    summary = json.loads((root / 'summary.json').read_text())
    ver = json.loads((root / 'verification.json').read_text())
    assert ver['summary_sha256'] == sha(root / 'summary.json') and ver['all_object_readouts_replayed'] == 16000
    assert summary['rows_sha256'] == sha(root / 'rows.npz')
    rows = np.load(root / 'rows.npz')
    masks = np.load(subset / 'masks.npz')
    names, _ = schema(family)
    eq = rows['prediction'] == rows['truth']
    np.testing.assert_array_equal(masks['attribute_clean_correct'], rows['clean_readout'] == rows['truth'])
    groups = {}
    for name, mask in [('all_objects', np.ones((8000, 2), bool)), ('clean_correct_objects', masks['object_clean_correct']), ('clean_wrong_objects', ~masks['object_clean_correct'])]:
        groups[name] = {'objects': int(mask.sum()), 'all_attributes_accuracy': float(eq.all(-1)[mask].mean()) if mask.any() else None,
                        'attributes': {attribute: float(eq[..., j][mask].mean()) if mask.any() else None for j, attribute in enumerate(names)}}
    for name, mask in [('all_scenes', np.ones(8000, bool)), ('both_clean_correct_scenes', masks['scene_clean_correct']), ('any_clean_wrong_scenes', ~masks['scene_clean_correct'])]:
        groups[name] = {'scenes': int(mask.sum()), 'both_objects_all_attributes_accuracy': float(eq.all((1, 2))[mask].mean()) if mask.any() else None}
    per_attribute = {name: {'objects': int(masks['attribute_clean_correct'][..., j].sum()),
                            'accuracy_on_clean_attribute_correct': float(eq[..., j][masks['attribute_clean_correct'][..., j]].mean()) if masks['attribute_clean_correct'][..., j].any() else None} for j, name in enumerate(names)}
    value = {'protocol_sha256': sha(BASE / 'attribute_calibration_diagnostic_protocol.json'), 'verified_raw_attribute_summary_sha256': sha(root / 'summary.json'),
             'subset_summary_sha256': sha(subset / 'summary.json'), 'groups': groups, 'per_attribute_subsets': per_attribute,
             'interpretation': 'All-model fixed, easier subsets; never substitute forfulltest orclaimunbiasedcorrectedaccuracy. Clean-correctreader canstill misreadgeneratedartifacts.'}
    lock(root / 'clean_calibration_diagnostic.json', value)
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['freeze', 'prepare', 'evaluate'])
    p.add_argument('--family', choices=['dsprites', 'clevr', 'clevrtex'])
    p.add_argument('--task', choices=TASKS)
    p.add_argument('--kind', choices=['plain', 'c4', 'object', 'copy'])
    p.add_argument('--mode', choices=['validation_selected', 'fixed20epoch'], default='validation_selected')
    p.add_argument('--init', type=int, default=0)
    p.add_argument('--split-seed', type=int, default=890101)
    args = p.parse_args()
    if args.stage == 'freeze':
        freeze()
    elif args.stage == 'prepare':
        prepare(args.family, args.task)
    else:
        evaluate(args.family, args.task, args.kind, args.mode, args.init, args.split_seed)
