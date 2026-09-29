"""Measure three-way GPU scheduling before choosing an additional worker."""
import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import time
from v3_common import HERE, ROOT, dump, lock, sha
from r5_intake import BASE
from r5_gpu_parallel_driver import process


def main(pids):
    root = BASE / 'gpu_schedule_benchmark'
    assert not (root / 'three_worker_summary.json').exists()
    for pid in pids:
        info = process(pid)
        assert info and 'r5_train.py' in info[1] and '--task Single_Atomic' in info[1] and 'T' not in info[0]
    lock(root / 'three_worker_protocol.json', {'source_sha256': sha(__file__), 'worker_source_sha256': sha(HERE / 'r5_gpu_schedule_benchmark.py'),
        'temporarily_suspended_owned_training_pids': pids, 'scope': 'Same discarded-weight200updatebenchmark, addserialC4andconcurrentplain/C4/object. No scientificweights/data/epochschange. Resumealloriginalworkers infinally.',
        'decision': 'Increasefrom2to3primaryworkers onlyifnormalizedserialcompute/parallelwall speedup>=1.1timesmeasuredtwo-workergain. Thisiscompute-only estimate,notrealjobcompletion claim.'})
    children = []
    def launch(kind, name):
        log = (root / f'{name}.log').open('w')
        p = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(HERE / 'r5_gpu_schedule_benchmark.py'), '--kind', kind, '--name', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append(p)
        code = p.wait()
        log.close()
        assert code == 0
    stopped = []
    started = time.time()
    try:
        for pid in pids:
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
        launch('c4', 'serial_c4')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(launch, kind, 'triple_' + kind) for kind in ('plain', 'c4', 'object')]
            for f in futures:
                f.result()
        records = {key: json.loads((root / f'{key}.json').read_text()) for key in ['serial_plain', 'serial_c4', 'serial_object', 'triple_plain', 'triple_c4', 'triple_object']}
        serial = sum(records['serial_' + kind]['seconds'] for kind in ('plain', 'c4', 'object'))
        parallel = max(records['triple_' + kind]['seconds'] for kind in ('plain', 'c4', 'object'))
        prior = json.loads((root / 'summary.json').read_text())['estimated_compute_throughput_speedup']
        dump(root / 'three_worker_summary.json', {'protocol_sha256': sha(root / 'three_worker_protocol.json'), 'serial_timed_seconds_sum': serial,
            'three_worker_timed_seconds_max': parallel, 'estimated_compute_throughput_speedup': serial / parallel,
            'two_worker_speedup': prior, 'relative_to_two_worker': serial / parallel / prior,
            'recommend_three_workers': serial / parallel >= prior * 1.1, 'runs': records,
            'scope': 'Compute-onlythroughput withdifferentindependentarchitectures,firstserialplain/objecttimingsreusedfromimmediatelyprecedingbenchmark. NotisolatedfullPNG-throughputorend-to-endcompletionmeasurement.'})
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for pid in stopped:
            os.kill(pid, signal.SIGCONT)
        dump(root / 'three_worker_resume_receipt.json', {'training_pids': stopped, 'all_resumed': True, 'pause_seconds': time.time() - started})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pause-pids', type=int, nargs='+', required=True)
    a = p.parse_args()
    main(a.pause_pids)
