"""Freeze the follow-up protocol and verify the original release."""
import json
import re
import time
from common import HERE, ROOT, dump, r, o


def main():
    manifest = json.loads((ROOT / 'RELEASE_MANIFEST.json').read_text())
    bad = []
    for f in manifest['files']:
        p = ROOT / f['path']
        if not p.exists() or r.sha(p) != f['sha256']:
            bad.append(f['path'])
    dump(HERE / 'reports/original_integrity_start.json', {
        'checked_unix': time.time(), 'manifest_sha256': r.sha(ROOT / 'RELEASE_MANIFEST.json'),
        'checked_files': len(manifest['files']), 'mismatches': bad,
        'environment': o.environment(), 'all_match': not bad})
    if bad:
        raise RuntimeError(f'Original artifacts differ: {bad}')
    status_path = HERE / 'status.json'
    if not status_path.exists():
        tasks = {}
        for line in (HERE / 'PLAN_KO.md').read_text().splitlines():
            m = re.match(r'\| ([A-D][0-9]{2}) \| (.*?) \| (.*?) \|', line)
            if m:
                tasks[m[1]] = {'description': m[2], 'completion_evidence_required': m[3],
                               'status': 'pending', 'evidence': []}
        tasks['A01']['status'] = 'complete'
        tasks['A01']['evidence'] = ['reports/original_integrity_start.json']
        dump(status_path, {'started_unix': time.time(), 'goal_status': 'active', 'tasks': tasks})
    config = {
        'frozen_unix': time.time(),
        'purpose': 'Exploratory diagnostics and matched-flow intervention on the original three frozen C4 AEs.',
        'audit_seed': 9212301, 'audit_n_per_split': 128,
        'fresh_probe_seed': 9212302, 'fresh_probe_n_per_split': 256,
        'old_train_probe_n': 256,
        'flow_modes': ['plain', 'augment', 'equivariant'], 'ae_seeds': [0, 1, 2],
        'flow_init_seed_offset': 81000, 'flow_sampling_seed_offset': 82000,
        'flow_augmentation_seed_offset': 83000, 'flow_eval_seed_offset': 84000,
        'flow_steps': 4000, 'flow_batch': 128, 'flow_lr': .001,
        'flow_eval_n': 576, 'flow_eval_steps': [16, 32, 64],
        'template_validity_thresholds': [.55, .65, .75, .85],
        'primary_template_threshold': .65,
        'human_review_n_per_model_seed_split': 20, 'review_shuffle_seed': 9212399,
        'checkpoint_selection': 'fixed final step; no test-driven model selection',
        'same_ae_comparison': 'All three flow modes receive identical normalized C4 latent pool, labels, base minibatches/noise/t, initial raw weights. Augmentation rotates BOTH target and noise using a separate RNG.',
        'limits': ['Single original training data seed for A07; A09 adds data seeds.',
                   'Diagnostic fresh test becomes development data once inspected.',
                   'Automatic scores are not human-validated generation success.']}
    p = HERE / 'configs/stage_a_v1.json'
    if not p.exists():
        dump(p, config)
    print(json.dumps({'original_files_checked': len(manifest['files']), 'all_match': not bad,
                      'tasks': len(json.loads(status_path.read_text())['tasks']),
                      'config_sha256': r.sha(p)}), flush=True)


if __name__ == '__main__':
    main()
