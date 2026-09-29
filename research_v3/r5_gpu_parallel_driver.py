"""Add two-worker MPS scheduling without changing any frozen scientific source.

Temporarily suspend only the original dispatcher while its active training child
and an independent model train. Resume that dispatcher after all group models
finish so original validation selection, evaluation, completion receipts and
dependent-process IDs stay intact. Errors resume the dispatcher in finally.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import time
import traceback
from v3_common import HERE, ROOT, dump, lock, sha
from r5_intake import BASE, TASKS
from r5_train import resource_check


def process(pid):
    p = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'command='], capture_output=True, text=True)
    value = p.stdout.strip()
    return None if not value else value.split(None, 1)


def option(parts, key, default=None):
    return parts[parts.index(key) + 1] if key in parts else default


def folder(family, task, kind, split_seed, init):
    base = BASE / 'external' / f'{family}_hard'
    if split_seed != 890101:
        base = base / 'confirmation' / f'p{split_seed}'
    return base / 'runs' / task / f'{kind}_s{init}'


def complete(family, task, kind, split_seed, init):
    root = folder(family, task, kind, split_seed, init)
    if not (root / 'run.json').exists():
        return False
    rec = json.loads((root / 'run.json').read_text())
    assert rec['steps'] == 36000 and rec['epochs'] == 20
    for epoch, digest in rec['checkpoints'].items():
        assert sha(root / f'epoch{epoch}.pt') == digest
    return True


def matches(info, family, task, split_seed, init):
    if info is None or info[0].startswith('Z'):
        return False
    parts = shlex.split(info[1])
    expected = 'r5_train.py' if family == 'dsprites' else 'r5_3d_train.py'
    return (any(v.endswith('/' + expected) or v == expected or v.endswith('research_v3/' + expected) for v in parts)
            and option(parts, '--task') == task and option(parts, '--family', 'dsprites') == family
            and int(option(parts, '--split-seed', 890101)) == split_seed and int(option(parts, '--init', 0)) == init)


def run_group(parent, progress_file, family, task, kinds, split_seed=890101, init=0):
    label = f'{family}_{task}_p{split_seed}_s{init}'
    record_path = BASE / 'gpu_parallel' / f'{label}.json'
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old.get('completed'):
            assert all(complete(family, task, kind, split_seed, init) for kind in kinds)
            return
    while True:
        if all(complete(family, task, kind, split_seed, init) for kind in kinds):
            return
        resource_check()
        parent_info = process(parent)
        assert parent_info and 'research_v3/r5_' in parent_info[1], 'Original dispatcher not live'
        assert 'T' not in parent_info[0], 'Dispatcher already stopped; inspect previous owner before resuming'
        if progress_file.exists():
            current = json.loads(progress_file.read_text())
            child = current.get('child_pid')
            info = process(child) if child else None
            if matches(info, family, task, split_seed, init):
                break
        dump(BASE / 'gpu_parallel_progress.json', {'controller_pid': os.getpid(), 'stage': 'waiting_original_training_dispatch', 'group': label, 'parent_pid': parent, 'unix': time.time()})
        time.sleep(5)
    started = time.time()
    own = {}
    timings = []
    suspended = False
    try:
        os.kill(parent, signal.SIGSTOP)
        suspended = True
        # Re-read after stop to close the dispatcher-boundary race.
        current = json.loads(progress_file.read_text())
        child = current.get('child_pid')
        info = process(child) if child else None
        if not matches(info, family, task, split_seed, init):
            raise RuntimeError('Dispatcher changed phase at boundary; safely resume and inspect before retry')
        active_kind = option(shlex.split(info[1]), '--kind')
        assert active_kind in kinds
        pending = [kind for kind in kinds if kind != active_kind and not complete(family, task, kind, split_seed, init)]
        record = {'protocol_sha256': sha(BASE / 'gpu_parallel_protocol.json'), 'family': family, 'task': task, 'split_seed': split_seed, 'init': init,
                  'original_dispatcher_pid': parent, 'original_training_child_pid': child, 'original_active_kind': active_kind,
                  'started_unix': started, 'completed': False, 'max_simultaneous_primary_training_workers': 2}
        dump(record_path, record)
        original_done = False
        while True:
            resource_check()
            if not original_done:
                info = process(child)
                if info is None or info[0].startswith('Z'):
                    assert complete(family, task, active_kind, split_seed, init), 'Original worker ended without complete run; resume dispatcher to expose error'
                    original_done = True
            for kind, (p, stream, launch_time) in list(own.items()):
                code = p.poll()
                if code is not None:
                    stream.close()
                    assert code == 0 and complete(family, task, kind, split_seed, init), f'Concurrent worker {kind} failed; partial checkpoint preserved'
                    timings.append({'kind': kind, 'child_pid': p.pid, 'process_wall_seconds': time.time() - launch_time})
                    del own[kind]
            active = int(not original_done) + len(own)
            while pending and active < 2:
                kind = pending.pop(0)
                args = [str(ROOT / '.venv/bin/python'), str(HERE / ('r5_train.py' if family == 'dsprites' else 'r5_3d_train.py')),
                        '--task', task, '--kind', kind, '--split-seed', str(split_seed), '--init', str(init)]
                if family != 'dsprites':
                    args += ['--family', family]
                stream = (BASE / 'logs' / f'gpu_parallel_{label}_{kind}.log').open('a')
                p = subprocess.Popen(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                own[kind] = (p, stream, time.time())
                active += 1
            dump(BASE / 'gpu_parallel_progress.json', {'controller_pid': os.getpid(), 'stage': 'parallel_MPS_training', 'group': label,
                'temporarily_stopped_dispatcher_pid': parent, 'original_worker_pid': None if original_done else child,
                'additional_workers': {kind: v[0].pid for kind, v in own.items()}, 'pending_kinds': pending, 'unix': time.time()})
            if original_done and not own and not pending:
                break
            time.sleep(5)
        assert all(complete(family, task, kind, split_seed, init) for kind in kinds)
        record.update(completed=True, completed_unix=time.time(), own_worker_walltimes=timings,
                      run_receipts={kind: sha(folder(family, task, kind, split_seed, init) / 'run.json') for kind in kinds})
        dump(record_path, record)
        for kind in kinds:
            dump(folder(family, task, kind, split_seed, init) / 'gpu_schedule_amendment.json', {
                'scheduling_protocol_sha256': sha(BASE / 'gpu_parallel_protocol.json'), 'group_receipt_sha256': sha(record_path),
                'scientific_model_data_batch_updates_unchanged': True, 'timing': 'Processseconds includeconcurrentGPUusage; originalactiveworker spansserialandparallel periods. Notanisolatedarchitecturespeedbenchmark.'})
    finally:
        for p, stream, _ in own.values():
            if p.poll() is None:
                p.terminate()
                p.wait()
            stream.close()
        if suspended and process(parent) is not None:
            os.kill(parent, signal.SIGCONT)
            dump(BASE / 'gpu_parallel' / f'{label}_dispatcher_resume.json', {'parent_pid': parent, 'resumed': True, 'unix': time.time()})


def main(dsprites_pid, three_d_pid, confirmation_pid):
    benchmark = json.loads((BASE / 'gpu_schedule_benchmark/summary.json').read_text())
    assert benchmark['estimated_compute_throughput_speedup'] >= 1.1
    assert json.loads((BASE / 'gpu_schedule_benchmark/resume_receipt.json').read_text())['resumed']
    lock(BASE / 'gpu_parallel_protocol.json', {'source_sha256': sha(__file__), 'benchmark_sha256': sha(BASE / 'gpu_schedule_benchmark/summary.json'),
        'authorization': 'User explicitlyrequestedaggressiveMacGPUusefortimesavings duringexistingR5 execution.',
        'device': 'AppleM5ProMPSfloat32;64GBunifiedmemory;18CPUlogicalcores; originaltwoCPUthreads perworker retained.',
        'change': 'Atmost2independentprimaryMPS trainingworkers withinthesameofficialtask/resplit/initgroup. Preserveoriginalfamily/taskorder andsequentialvalidation/testdispatch. Modelarchitectures,weightsinitializations,dataorders,batch32,36000updates,optimizer,precision/checkpointselection unchanged.',
        'safety': 'TemporarilySIGSTOPonlyverifiedoriginaldispatcher,nottrainingworker. Afterallgroupmodelrunreceiptscomplete,SIGCONTdispatcher. Existingdependencypids/verificationdrivers intact. finallyresumesdispatcher andterminatesownunfinishedworkers; atomiccheckpoints retained.',
        'budget': 'ExistingR5global12h/30GBcaps unchanged. InterleavedGPUworkchangeswalltiming; reportbenchmarkandactualprocesssecondswithconcurrencydisclosure. Shortvalidation/readoutGPUjobs canalsooverlap;two-worker limitrefers toprimarytraining.',
        'no_new_scientific_results': 'Schedulingbenchmark usesdiscardedweightsanddoesnotselectmodelsorclaimaccuracy.'})
    for task in TASKS:
        run_group(dsprites_pid, BASE / 'development_progress.json', 'dsprites', task, ['plain', 'c4', 'object'])
    for family in ('clevr', 'clevrtex'):
        run_group(three_d_pid, BASE / '3d_development_progress.json', family, 'Single_Atomic', ['plain', 'c4', 'object'])
    while not (BASE / 'confirmation_cells.json').exists():
        assert process(confirmation_pid), 'Confirmation dispatcher ended; inspect its failure record'
        resource_check()
        time.sleep(10)
    cells = json.loads((BASE / 'confirmation_cells.json').read_text())['cells']
    groups = {}
    for cell in cells:
        key = (cell['family'], cell['task'], cell['split_seed'], cell['init'])
        groups.setdefault(key, []).append(cell['kind'])
    for (family, task, seed, init), kinds in groups.items():
        run_group(confirmation_pid, BASE / 'confirmation_progress.json', family, task, kinds, seed, init)
    dump(BASE / 'gpu_parallel_complete.json', {'protocol_sha256': sha(BASE / 'gpu_parallel_protocol.json'), 'all_scheduled_training_groups_complete': True,
        'full_R5_complete': False, 'note': 'Originaldispatchersstillperformlastselection/evaluation;thisisnotcompleteexternalverification.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dsprites-pid', type=int, required=True)
    p.add_argument('--3d-pid', type=int, required=True, dest='three_d_pid')
    p.add_argument('--confirmation-pid', type=int, required=True)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit('Controller terminated; resume originaldispatcher')))
    try:
        main(args.dsprites_pid, args.three_d_pid, args.confirmation_pid)
    except BaseException:
        dump(BASE / 'gpu_parallel_failure.json', {'unix': time.time(), 'traceback': traceback.format_exc()})
        raise
