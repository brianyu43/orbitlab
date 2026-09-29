"""Append frozen reader-reliability diagnostics once raw full-test scores verify."""
import argparse
import os
import time
import traceback
from v3_common import dump, lock, sha
from r5_intake import BASE, TASKS
from r5_attribute_evaluate import backend
from r5_attributes import folder
from r5_attribute_calibration_diagnostic import freeze, prepare, evaluate


def main(dsprites_pid, auxiliary_pid):
    freeze()
    lock(BASE / 'calibration_diagnostic_driver_protocol.json', {'source_sha256': sha(__file__), 'diagnostic_protocol_sha256': sha(BASE / 'attribute_calibration_diagnostic_protocol.json'),
        'scope': 'Prepareclean-onlysubsetasreaderfinishes;appenddiagnosticsforall42developmentpixelunits afterrawattributeverification. No generator-dependentthresholds.'})
    completed = []
    for family, task in [('dsprites', task) for task in TASKS] + [('clevr', 'Single_Atomic'), ('clevrtex', 'Single_Atomic')]:
        pid = dsprites_pid if family == 'dsprites' else auxiliary_pid
        cal = folder(family, task) / 'calibration_test_verification.json'
        while not cal.exists():
            os.kill(pid, 0)
            time.sleep(20)
        prepare(family, task)
        for kind in ('copy', 'plain', 'c4', 'object'):
            for mode in (['validation_selected'] if kind == 'copy' else ['validation_selected', 'fixed20epoch']):
                destination = backend(family).unit(task, kind, 0, mode) / 'attributes' / 'verification.json'
                while not destination.exists():
                    os.kill(pid, 0)
                    dump(BASE / 'calibration_diagnostic_progress.json', {'pid': os.getpid(), 'stage': 'waiting_verified_raw_attributes', 'family': family, 'task': task, 'kind': kind, 'mode': mode, 'unix': time.time()})
                    time.sleep(20)
                evaluate(family, task, kind, mode)
                completed.append({'family': family, 'task': task, 'kind': kind, 'mode': mode})
    dump(BASE / 'calibration_diagnostic_complete.json', {'driver_protocol_sha256': sha(BASE / 'calibration_diagnostic_driver_protocol.json'), 'development_units': completed, 'count': len(completed)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dsprites-attributes-pid', type=int, required=True)
    p.add_argument('--3d-auxiliary-pid', type=int, required=True, dest='auxiliary_pid')
    a = p.parse_args()
    try:
        main(a.dsprites_attributes_pid, a.auxiliary_pid)
    except Exception:
        dump(BASE / 'calibration_diagnostic_failure.json', {'unix': time.time(), 'traceback': traceback.format_exc()})
        raise
