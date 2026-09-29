"""Inventory or assemble a local research review bundle with explicit omissions.

Plan mode is usable while experiments run. Building requires a separate final
scope decision; it is not a way to convert partial experiments into completion.
Nothing is downloaded, uploaded, trained, or removed by this program.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
import zipfile
from v3_common import ROOT, HERE, dump, sha


ALLOWED_SUFFIXES = {'.py', '.md', '.json', '.csv', '.txt', '.png', '.pdf',
                    '.npz', '.npy', '.pt', '.sqlite', '.sh'}
RUNTIME_NAMES = {'WORKING_STATE.md', 'journal.jsonl', 'live_process.json',
                 'prediction_verification_cache.npy'}
EXCLUDED_TREES = {'__pycache__', '.matplotlib', 'release', 'fresh_reproductions'}
MANIFEST_MEMBER = 'REVIEW_BUNDLE_MANIFEST.json'


def read(path):
    return json.loads(path.read_text())


def relative(path):
    assert path.is_file() and not path.is_symlink(), path
    assert path.resolve().is_relative_to(ROOT), path
    return path.relative_to(ROOT).as_posix()


def dependency_files():
    """Local historical dependencies; no conversations or participant records."""
    paths = set()
    for name in ('work', 'followup', 'generation_recovery', 'completion_v2'):
        paths.update((ROOT / name).glob('*.py'))
    for name in ('data', 'followup/data', 'generation_recovery/data_v1',
                 'generation_recovery/confirmation_v1/data', 'completion_v2/generation/data'):
        paths.update((ROOT / name).rglob('*metadata*.json'))
    for name in ('followup/data/object_world_v1', 'followup/data/object_image_edit_v1',
                 'completion_v2/perception/data', 'completion_v2/perception_detail/data'):
        for scene in (ROOT / name).rglob('scenes.json'):
            paths.add(scene)
            paths.update(p for p in (scene.with_name('tasks.json'), scene.with_name('edits.json')) if p.exists())
    for name in ('followup/data/dynamics_world_v1', 'completion_v2/dynamics/data'):
        paths.update((ROOT / name).rglob('trajectories.npz'))
    for name in ('requirements.lock.txt', 'THIRD_PARTY_NOTICES.md',
                 'completion_v2/NEXT_RESEARCH_PLAN_KO.md', 'completion_v2/artifact_manifest.json',
                 'completion_v2/svib/data_manifest.json',
                 'completion_v2/perception_detail/runs/alpha_s0/model.pt',
                 'followup/reports/evaluator_calibration_v1/locked_threshold.json',
                 'generation_recovery/reports/p0_diversity_reader_lock_v1.json'):
        path = ROOT / name
        assert path.is_file(), f'Missing historical dependency: {name}'
        paths.add(path)
    # The exact prior metadata list is part of the R4 split audit.
    for path in (HERE / 'r4/data').glob('d*/manifest.json'):
        for row in read(path)['prior_sources']:
            old = ROOT / row['path']
            assert old in paths and sha(old) == row['sha256'], old
    return paths


def upstream_dependencies():
    base = ROOT / 'followup/external/svib'
    paths = {base / 'LICENSE'}
    for family in ('dsprites', 'clevr', 'clevrtex'):
        paths.update(base / 'data_creation' / family / name for name in ('macros.py', 'rule.py', 'utils.py', 'create_data.py'))
    paths.update((base / 'data_creation/dsprites/spriteworld').rglob('*.py'))
    return {'repository': 'https://github.com/systematic-visual-imagination/svib',
            'commit': '7eec57ef3c2ea7a2c7389cb6cd470eb3600fa57c',
            'required_destination': 'followup/external/svib',
            'included_in_bundle': False,
            'reason': 'Retrieve the official fixed checkout with original CC0 and embedded Spriteworld Apache-2.0 notices; do not treat the whole mirror as OrbitLab-authored material.',
            'files': [{'path': relative(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(paths)]}


def payload_dependencies():
    result = []
    for family, tasks in [('dsprites', ('Single_Atomic', 'Multiple_Atomic', 'Single_Non-Atomic', 'Multiple_Non-Atomic')),
                          ('clevr', ('Single_Atomic',)), ('clevrtex', ('Single_Atomic',))]:
        base = HERE / 'r5/data' / f'{family}_hard'
        for task in tasks:
            path = base / task / 'payload.bin'
            mp = base / 'manifest.json' if family == 'dsprites' else base / task / 'manifest.json'
            manifest = read(mp)
            prefix = task + '/' if family == 'dsprites' else ''
            result.append({'family': family, 'task': task, 'path': relative(path), 'bytes': path.stat().st_size,
                           'sha256': manifest['files'][prefix + 'payload.bin'], 'manifest_path': relative(mp),
                           'manifest_sha256': sha(mp), 'archive_sha256': manifest['archive_sha256'],
                           'included_in_bundle': False, 'restore_program': 'research_v3/restore_external_payload.py',
                           'digest_basis': 'Frozen intake manifest; payload is not rehashed by plan mode.'})
    return result


def selection():
    paths = dependency_files()
    skipped = Counter()
    for path in HERE.rglob('*'):
        if not path.is_file():
            continue
        rel = path.relative_to(HERE)
        reason = None
        if any(part in EXCLUDED_TREES for part in rel.parts):
            reason = 'runtime_cache_or_prior_bundle_or_separate_replication'
        elif rel.parts[:2] == ('review', 'payload_recovery_check'):
            reason = 'recovery_test_duplicate_and_machine_specific_receipts'
        elif rel.parts[0] == 'review' and path.name.startswith('package_'):
            reason = 'packager_own_snapshot_avoids_recursive_inventory'
        elif path.name == 'payload.bin':
            reason = 'official_payload_restore_separately'
        elif path.name in RUNTIME_NAMES or path.suffix in ('.tmp', '.log', '.partial', '.pyc', '.lock'):
            reason = 'runtime_or_temporary'
        elif path.suffix not in ALLOWED_SUFFIXES and path.name not in ('svib_LICENSE', 'clevr_download_prefix.bin'):
            reason = 'unrecognized_extension_requires_review'
        if reason:
            skipped[reason] += 1
        else:
            paths.add(path)
    return sorted(paths), dict(skipped)


def inventory():
    paths, skipped = selection()
    rows = [{'path': relative(p), 'bytes': p.stat().st_size} for p in paths]
    sizes = Counter()
    for row in rows:
        parts = PurePosixPath(row['path']).parts
        group = '/'.join(parts[:2]) if parts[0] == 'research_v3' and len(parts) > 2 else parts[0]
        sizes[group] += row['bytes']
    result = {'source_sha256': sha(__file__), 'snapshot_unix': time.time(),
              'mode': 'inventory_only_not_a_final_bundle', 'files': rows,
              'included_file_count': len(rows), 'included_uncompressed_bytes': sum(r['bytes'] for r in rows),
              'included_bytes_by_group': dict(sizes), 'excluded_file_counts': skipped,
              'external_payloads': payload_dependencies(), 'upstream_dependency': upstream_dependencies(),
              'complete_history_included': False, 'fresh_install_tested': False,
              'independent_retraining': False,
              'limitations': ['Official archives/payloads and third-party checkout must be obtained separately and hash-verified.',
                              'A full 81,297-file historical preservation audit requires the original archive tree, not this selected review bundle.',
                              'Cross-device bitwise replay is not promised; retain the frozen device, precision, batch and thread provenance.',
                              'Inventory while jobs run is provisional. Build refuses missing final scope decision and changed files.']}
    dump(HERE / 'review/package_inventory.json', result)
    return result


def final_decision():
    path = HERE / 'review/final_scope_decision.json'
    assert path.exists(), 'Final scope decision does not exist; experiments/reporting remain unfinished'
    value = read(path)
    assert value.get('ready_for_review_package') is True
    assert value.get('scientific_scope_complete') is True
    assert value.get('required_pending') == []
    required = {'external_scope', 'resource_ledger', 'human_applicability', 'next_study', 'historical_preservation'}
    assert required <= value.get('evidence', {}).keys()
    for evidence in value['evidence'].values():
        p = (HERE / evidence['path']).resolve()
        assert p.is_relative_to(HERE) and sha(p) == evidence['sha256']
    return path, value


def verify_archive(path):
    """Read every archived byte; never extract paths or load pickle checkpoints."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP member'
        manifest = json.loads(archive.read(MANIFEST_MEMBER))
        assert set(names) == {MANIFEST_MEMBER, *(v['path'] for v in manifest['files'])}
        for row in manifest['files']:
            name = PurePosixPath(row['path'])
            assert not name.is_absolute() and '..' not in name.parts
            assert archive.getinfo(row['path']).file_size == row['bytes']
            with archive.open(row['path']) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256'], row['path']
    return {'archive_sha256': sha(path), 'archive_bytes': path.stat().st_size,
            'every_member_fully_read_and_hashed': True, 'files_verified': len(manifest['files']),
            'source_bytes_verified': sum(v['bytes'] for v in manifest['files']),
            'does_not_verify_scientific_claims_or_retrain': True}


def build(name):
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name)
    decision_path, decision = final_decision()
    plan = inventory()
    assert plan['excluded_file_counts'].get('unrecognized_extension_requires_review', 0) == 0
    # No transient prediction cache may be lost during packaging an active job.
    assert not list((HERE / 'r5').rglob('prediction_verification_cache.npy'))
    assert read(HERE / 'r5/external_scope_snapshot.json')['external_scope_complete']
    destination = HERE / 'release' / name
    assert not destination.exists(), 'Use a fresh release name; old bundles are immutable'
    destination.mkdir(parents=True)
    files = []
    for item in plan['files']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['bytes']
        files.append({**item, 'sha256': sha(path)})
    manifest = {**plan, 'mode': 'final_review_bundle', 'files': files,
                'final_scope_decision_sha256': sha(decision_path),
                'scientific_scope_complete': decision['scientific_scope_complete']}
    temporary = destination / 'orbitlab-research-v3-review.zip.partial'
    final = temporary.with_suffix('')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        archive.writestr(MANIFEST_MEMBER, json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        for item in files:
            path = ROOT / item['path']
            archive.write(path, item['path'])
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], f'Changed during packaging: {path}'
    receipt = verify_archive(temporary)
    # The decision itself must not have changed while the ZIP was assembled.
    assert sha(decision_path) == manifest['final_scope_decision_sha256']
    temporary.rename(final)
    dump(destination / 'verification.json', {**receipt, 'archive': final.name,
                                            'final_scope_decision_sha256': manifest['final_scope_decision_sha256']})
    print(json.dumps({'archive': str(final), **receipt}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('plan', 'build', 'verify'))
    parser.add_argument('--name')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.stage == 'plan':
        result = inventory()
        print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'upstream_dependency', 'external_payloads')}, indent=2))
    elif args.stage == 'build':
        assert args.name, '--name is required for build'
        build(args.name)
    else:
        assert args.archive, '--archive is required for verify'
        print(json.dumps(verify_archive(args.archive), indent=2))
