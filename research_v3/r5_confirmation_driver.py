"""Execute only validation-gated confirmations after all development families.

These are three train/val resplits x three initializations of the SAME official
data, not three independent external datasets. Hard caps remain authoritative.
"""
import argparse
import json
import os
import subprocess
import time
import traceback
from v3_common import HERE, ROOT, dump, lock, sha
from r5_intake import BASE, TASKS
from r5_attribute_evaluate import backend
from r5_train import resource_check


def run(label, script, args):
    resource_check()
    log = BASE / 'logs' / f'confirm_{label}.log'
    with log.open('a') as stream:
        p = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(HERE / script), *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        dump(BASE / 'confirmation_progress.json', {'driver_pid': os.getpid(), 'child_pid': p.pid, 'stage': label, 'unix': time.time()})
        code = p.wait()
    assert code == 0, f'{label} failed; inspect {log}'


def main(wait_pid):
    tasks = [('dsprites', task) for task in TASKS] + [('clevr', 'Single_Atomic'), ('clevrtex', 'Single_Atomic')]
    protocols = {}
    for family, task in tasks:
        mod = backend(family)
        mod.training.freeze()
        mod.freeze()
        protocols[family] = {'training': sha(BASE / f'{family}_training_protocol.json'), 'evaluation': sha(BASE / f'{family}_evaluation_protocol.json')}
    lock(BASE / 'confirmation_driver_protocol.json', {'source_sha256': sha(__file__), 'protocols': protocols,
        'tasks': [list(v) for v in tasks], 'partition_seeds': [890201, 890202, 890203], 'initializations': [0, 1, 2],
        'selection': 'Onlyselectedcandidate fromfrozendevelopmentvalidation gate plusplain. No modelreselection orcheckpointchoiceusingtest. Each18runs/positivetask36000updates;9same-datasetresplits/initializations,not9independentdatasets.',
        'scope': 'Waitall6developmenttasks, thenpositivegatedtasksinoriginalorder. Fulltrain/valverification,testselected+fixed20,fullpixelreplay,independentfrozenattribute readout+replay,calibrationdiagnostic. Inputcopyisidenticaltoalreadyverifieddevelopmentcopyandnotretrained.',
        'resources': 'Same12hglobalR5/30GBcap. Allpendingcells preservedexplicitlyifcap; no shortenedepochs orsilentcompletion.',
        'reader': 'Samefrozeninstrumenttrained ondevelopmentpartition. It nevertrains onofficialtest. Someconfirmationvalidationframesmayhavebeeninstrumenttrainframes; attributesonlyscored onofficialtest. No diagnosticgrader fittingusingconfirmationoutputs.'})
    dependency = BASE / 'clevrtex_development_complete.json'
    while not dependency.exists():
        os.kill(wait_pid, 0)
        resource_check()
        dump(BASE / 'confirmation_progress.json', {'driver_pid': os.getpid(), 'stage': 'waiting_all_six_development_tasks', 'live_dependency_pid': wait_pid, 'unix': time.time()})
        time.sleep(30)
    selected = []
    cells = []
    for family, task in tasks:
        path = BASE / 'external' / f'{family}_hard' / 'evaluation' / task / 'development_selection.json'
        rec = json.loads(path.read_text())
        candidate = rec['selected_candidate']
        selected.append({'family': family, 'task': task, 'candidate': candidate, 'selection_sha256': sha(path)})
        if candidate is not None:
            for split_seed in [890201, 890202, 890203]:
                for init in [0, 1, 2]:
                    for kind in ['plain', candidate]:
                        cells.append({'family': family, 'task': task, 'split_seed': split_seed, 'init': init, 'kind': kind})
    lock(BASE / 'confirmation_cells.json', {'selected': selected, 'cells': cells, 'updates_required': 36000 * len(cells)})
    completed = []
    for cell in cells:
        family, task, split_seed, init, kind = (cell[k] for k in ['family', 'task', 'split_seed', 'init', 'kind'])
        prefix = f'{family}_{task}_p{split_seed}_s{init}_{kind}'
        args = ['--task', task, '--kind', kind, '--init', str(init), '--split-seed', str(split_seed)]
        extra = [] if family == 'dsprites' else ['--family', family]
        train_script = 'r5_train.py' if family == 'dsprites' else 'r5_3d_train.py'
        eval_script = 'r5_evaluate.py' if family == 'dsprites' else 'r5_3d_evaluate.py'
        run(prefix + '_train', train_script, [*extra, *args])
        run(prefix + '_train_verify', 'r5_verify_external.py', ['training', '--family', family, *args])
        for mode in ['validation_selected', 'fixed20epoch']:
            for stage in ['evaluate', 'verify']:
                run(prefix + '_' + mode + '_' + stage, eval_script, [stage, *extra, *args, '--mode', mode])
            reader = BASE / 'attributes' / family / task / 'calibration_test_verification.json'
            assert reader.exists(), 'Readout must be independently calibrated before confirmationattributes'
            for verify in [False, True]:
                run(prefix + '_' + mode + '_attributes_' + str(verify), 'r5_attribute_evaluate.py', ['--family', family, *args, '--mode', mode, *(['--verify'] if verify else [])])
            run(prefix + '_' + mode + '_calibration_diagnostic', 'r5_attribute_calibration_diagnostic.py', ['evaluate', '--family', family, *args, '--mode', mode])
        completed.append(cell)
        dump(BASE / 'confirmation_completed_cells.json', {'completed': completed, 'total': len(cells), 'remaining': len(cells) - len(completed)})
    dump(BASE / 'confirmation_complete.json', {'driver_protocol_sha256': sha(BASE / 'confirmation_driver_protocol.json'), 'cells_sha256': sha(BASE / 'confirmation_cells.json'),
                                            'completed_cells': len(cells), 'updates': 36000 * len(cells), 'skipped_failed_gate_tasks': sum(v['candidate'] is None for v in selected), 'full_R5_complete': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--development-driver-pid', type=int, required=True)
    args = p.parse_args()
    try:
        main(args.development_driver_pid)
    except Exception:
        dump(BASE / 'confirmation_failure.json', {'unix': time.time(), 'traceback': traceback.format_exc()})
        raise
