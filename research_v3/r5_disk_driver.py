"""Sequential independent components with durable progress and failures."""
import json
import os
import subprocess
import time
import traceback
from v3_common import HERE, ROOT, dump, lock, sha
import r5_disk_data as d
import r5_disk_core as c
import r5_disk_train as tr


def main():
    tr.freeze()
    lock(d.BASE / 'driver_protocol.json', {'source_sha256': sha(__file__), 'training_protocol_sha256': sha(d.BASE / 'training_protocol.json'),
                                        'scope': 'Five fixed-budget independentlytrained components. Evaluation and verification are separate; this receipt is not integratedsuccess.', 'order': list(c.ARMS)})
    assert (d.BASE / 'preflight.json').exists()
    for arm in c.ARMS:
        assert (d.BASE / 'benchmark' / arm / 'run.json').exists()
        log = d.BASE / f'{arm}_train.log'
        with log.open('a') as stream:
            p = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(HERE / 'r5_disk_train.py'), '--arm', arm], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            dump(d.BASE / 'training_progress.json', {'driver_pid': os.getpid(), 'child_pid': p.pid, 'arm': arm, 'unix': time.time()})
            code = p.wait()
        assert code == 0, f'{arm} failed; see {log}'
    dump(d.BASE / 'training_complete.json', {'driver_protocol_sha256': sha(d.BASE / 'driver_protocol.json'), 'updates': sum(c.STEPS.values()),
                                           'runs': {arm: sha(d.BASE / 'runs' / arm / 'run.json') for arm in c.ARMS}, 'unix': time.time(), 'evaluation_complete': False})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(d.BASE / 'training_failure.json', {'unix': time.time(), 'traceback': traceback.format_exc()})
        raise
