"""Fresh same-seed disk replication without overwriting any original artifacts.

This orchestrator reuses immutable sources but redirects the shared data module
to a new directory. It does not bypass the old experiment's hash guards and is
not a new independent dataset seed or a completed replication until executed.
"""
import argparse
import json
import re
import time
import traceback
from v3_common import HERE, dump, lock, sha
import r5_disk_data as d
import r5_disk_core as c
import r5_disk_train as tr
import r5_disk_preflight as preflight
import r5_disk_evaluate as ev
import r5_disk_evaluation_driver as closing
import r5_disk_report as report


def main(name, execute):
    assert re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', name), 'Use a short new alphanumeric run name'
    parent = HERE / 'fresh_reproductions' / name
    spec = {'source_sha256': sha(__file__), 'output': str(parent.relative_to(HERE)), 'data_seed': d.SEED,
            'planned_updates': sum(c.STEPS.values()), 'planned_pilot_updates': 500, 'local_seconds_cap': 43200, 'disk_cap_bytes': 30000000000,
            'scope': 'Independent rerun ofthe SAMEseed/splits;not3x3confirmation. Newtiming/manifestsareexpectedtobehashdifferent. HistoricalR3overlapcomparison coversonlylocallyavailableR3families andisreported. Alloriginalprotocols,checkpoints,anddata remainuntouched.'}
    if not execute:
        print(json.dumps(spec, indent=2))
        return
    d.BASE = parent / 'disks'
    if parent.exists() and not (parent / 'replication_spec.json').exists():
        raise RuntimeError('Nonempty/unrecognized output path; choose a new name')
    lock(parent / 'replication_spec.json', spec)
    started = parent / 'training_started.json'
    if not started.exists():
        dump(started, {'unix': time.time(), 'note': 'Freshdiskreplication resourceclock; independent oforiginalR5stage'})
    try:
        for split, count in [('train', 2), ('val', 2)] + [('test', n) for n in d.COUNTS]:
            d.prepare_group(split, count)
        if not (d.BASE / 'data_verification.json').exists():
            d.verify()
        preflight.main()
        for arm in c.ARMS:
            tr.train(arm)
        ev.freeze()
        ev.verify_training()
        for split, count in [('val', 2)] + [('test', n) for n in d.COUNTS]:
            ev.perceive(split, count)
            ev.perceive(split, count, verify=True)
        for count in d.COUNTS:
            ev.controls(count)
            ev.controls(count, verify=True)
            for arm in ('bounded', 'contact', 'coarse'):
                for state in ('true', 'learned'):
                    for force in ('known', 'inferred'):
                        tr.resource_check()
                        ev.evaluate(count, arm, state, force)
                        ev.evaluate(count, arm, state, force, verify=True)
        closing.close()
        report.main()
        dump(parent / 'replication_complete.json', {'spec_sha256': sha(parent / 'replication_spec.json'), 'verification_sha256': sha(d.BASE / 'verification.json'),
                                                   'report_receipt_sha256': sha(d.BASE / 'report_receipt.json'), 'new_independent_data_seed': False, 'visual_review_complete': False})
    except Exception:
        dump(parent / 'failure.json', {'unix': time.time(), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--execute', action='store_true', help='Actually generate/train/evaluate; omission prints the plan only')
    a = p.parse_args()
    main(a.name, a.execute)
