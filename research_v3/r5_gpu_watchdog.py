"""Recover only our suspended dispatcher if the GPU coordinator disappears."""
import argparse
import json
import os
import signal
import time
from v3_common import dump, sha
from r5_intake import BASE
from r5_gpu_parallel_driver import process


def main(pid):
    dump(BASE / 'gpu_watchdog_started.json', {'source_sha256': sha(__file__), 'controller_pid': pid, 'watchdog_pid': os.getpid(), 'unix': time.time()})
    while True:
        info = process(pid)
        if info is None or info[0].startswith('Z'):
            break
        assert 'r5_gpu_parallel_driver.py' in info[1], 'PID reused; do not touch unrelated processes'
        time.sleep(5)
    progress = BASE / 'gpu_parallel_progress.json'
    value = json.loads(progress.read_text()) if progress.exists() else {}
    result = {'controller_pid': pid, 'unix': time.time(), 'additional_workers_terminated': [], 'dispatcher_resumed': None}
    if value.get('controller_pid') == pid and value.get('stage') == 'parallel_MPS_training':
        parent = value['temporarily_stopped_dispatcher_pid']
        parent_info = process(parent)
        if parent_info and 'T' in parent_info[0] and 'research_v3/r5_' in parent_info[1] and 'driver.py' in parent_info[1]:
            for kind, child in value.get('additional_workers', {}).items():
                info = process(child)
                if info and not info[0].startswith('Z'):
                    assert ('r5_train.py' in info[1] or 'r5_3d_train.py' in info[1]) and f'--kind {kind}' in info[1]
                    os.kill(child, signal.SIGTERM)
                    result['additional_workers_terminated'].append(child)
            # Wait for worker exits before releasing original dispatcher into
            # those same outputdirectories. It restores atomic partialprogress.
            for child in result['additional_workers_terminated']:
                while True:
                    info = process(child)
                    if info is None or info[0].startswith('Z'):
                        break
                    time.sleep(1)
            os.kill(parent, signal.SIGCONT)
            result['dispatcher_resumed'] = parent
    dump(BASE / 'gpu_watchdog_exit.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controller-pid', type=int, required=True)
    a = p.parse_args()
    main(a.controller_pid)
