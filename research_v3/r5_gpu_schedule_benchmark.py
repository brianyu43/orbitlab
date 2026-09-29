"""Discarded-weight scheduling benchmark, never a scientific training run."""
import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import time
import numpy as np
import torch
from v3_common import HERE, ROOT, dump, sha
from r5_intake import BASE
from r5_data import Pairs
from r5_models import make


def worker(kind, name):
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available()
    store = Pairs('Single_Atomic')
    # Exact native inputs and original operations/optimizer. Resident batches
    # isolate compute scheduling; this is not full PNG-I/O throughput evidence.
    pairs = [store.batch('train', np.arange(k * 32, (k + 1) * 32), torch.device('mps')) for k in range(4)]
    store.close()
    torch.manual_seed(950000)
    model = make(kind).to('mps')
    opt = torch.optim.Adam(model.parameters(), lr=.0005)
    logs = []
    torch.mps.synchronize()
    began = time.perf_counter()
    for step in range(220):
        x, y = pairs[step % 4]
        loss = (model(x) - y).square().mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        assert torch.isfinite(norm)
        opt.step()
        if step == 19:
            torch.mps.synchronize()
            began = time.perf_counter()
        if step in (99, 219):
            logs.append(float(loss.detach().cpu()))
    torch.mps.synchronize()
    seconds = time.perf_counter() - began
    dump(BASE / 'gpu_schedule_benchmark' / f'{name}.json', {'kind': kind, 'timed_updates': 200, 'warmup_updates': 20, 'seconds': seconds,
        'updates_per_second': 200 / seconds, 'model_and_updates_unchanged': True, 'weights_discarded': True, 'device': 'MPSfloat32', 'CPU_threads': 2,
        'scope': 'Residentfourtrainingbatches; excludesPNGloadandtransfer. Samefinitechecks/clip/Adam asprimary loop.', 'loss_samples': logs})


def coordinator(pid):
    folder = BASE / 'gpu_schedule_benchmark'
    folder.mkdir(exist_ok=True)
    command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'command='], text=True)
    assert 'r5_train.py' in command and '--kind c4' in command, 'Only the verified activeC4experiment may be temporarily suspended'
    assert not (folder / 'summary.json').exists(), 'Inspect prior benchmark instead of rerunning'
    dump(folder / 'protocol.json', {'source_sha256': sha(__file__), 'temporarily_suspended_own_training_pid': pid,
        'scope': 'UserrequestedMacGPUspeedup. Briefcontrolledsuspension ofoneownedMPSworker tobenchmark1vs2simultaneousfixed32batchMPSjobs. Originaltrainingresumes infinally; no scientificcheckpoint orRNGchanged. Its recordedwalltime includespause.',
        'hardware': 'Hosthardwaremetadata storedseparately; no serialnumber or credentials collected.'})
    python = str(ROOT / '.venv/bin/python')
    children = []

    def launch(kind, name):
        log = (folder / f'{name}.log').open('w')
        child = subprocess.Popen([python, str(HERE / 'r5_gpu_schedule_benchmark.py'), '--kind', kind, '--name', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        code = child.wait()
        log.close()
        assert code == 0, f'Benchmark{name} failed'

    began = time.time()
    os.kill(pid, signal.SIGSTOP)
    try:
        launch('plain', 'serial_plain')
        launch('object', 'serial_object')
        parallel_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(launch, kind, 'parallel_' + kind) for kind in ('plain', 'object')]
            for future in futures:
                future.result()
        elapsed = time.perf_counter() - parallel_start
        recs = {key: json.loads((folder / f'{key}.json').read_text()) for key in ['serial_plain', 'serial_object', 'parallel_plain', 'parallel_object']}
        serial = sum(recs[k]['seconds'] for k in ['serial_plain', 'serial_object'])
        parallel = max(recs[k]['seconds'] for k in ['parallel_plain', 'parallel_object'])
        dump(folder / 'summary.json', {'protocol_sha256': sha(folder / 'protocol.json'), 'serial_timed_seconds_sum': serial,
            'parallel_timed_seconds_max': parallel, 'estimated_compute_throughput_speedup': serial / parallel,
            'parallel_total_wall_seconds_including_startup': elapsed, 'runs': recs,
            'decision_rule': 'Useconcurrent2primaryMPSworkers onlyifcompute aggregate>=1.1xandmemorysafe; thenmonitorrealPNG-inclusivejobtiming. No guaranteeentireexperimentfinishesatthisspeedup.'})
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        os.kill(pid, signal.SIGCONT)
        dump(folder / 'resume_receipt.json', {'training_pid': pid, 'resumed': True, 'pause_seconds': time.time() - began})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['plain', 'c4', 'object'])
    p.add_argument('--name')
    p.add_argument('--coordinate-pid', type=int)
    a = p.parse_args()
    if a.coordinate_pid:
        coordinator(a.coordinate_pid)
    else:
        worker(a.kind, a.name)
