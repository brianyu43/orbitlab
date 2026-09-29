"""Complete all disk swap cells and independent replay before gate accounting."""
import argparse
import json
import os
import subprocess
import time
import traceback
from v3_common import HERE, ROOT, dump, lock, sha
import r5_disk_data as d
import r5_disk_evaluate as ev
import r5_disk_train as tr


def run(label, args):
    tr.resource_check()
    log = d.BASE / 'logs' / f'{label}.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        p = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(HERE / 'r5_disk_evaluate.py'), *args], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        dump(d.BASE / 'evaluation_progress.json', {'driver_pid': os.getpid(), 'child_pid': p.pid, 'stage': label, 'unix': time.time()})
        code = p.wait()
    assert code == 0, f'{label} failed; see {log}'


def record(count, arm, state, force, horizon, scenario='baseline', group='all'):
    p = d.BASE / 'evaluation' / 'test' / f'n{count}' / f'{arm}_{state}_{force}' / 'summary.json'
    rec = json.loads(p.read_text())
    return next(v for v in rec['aggregates'] if v['horizon'] == horizon and v['scenario'] == scenario and v['group'] == group)


def close():
    receipts = {}
    rgb = trajectories = 0
    for p in sorted((d.BASE / 'evaluation').rglob('verification.json')):
        v = json.loads(p.read_text())
        assert v['summary_sha256'] == sha(p.with_name('summary.json'))
        receipts[str(p.relative_to(d.BASE))] = sha(p)
        rgb += v.get('full_RGB_replays_bitexact', 0)
        trajectories += v.get('full_trajectory_replays', 0)
    assert len(receipts) == 57  # 5 perception + 4 controls +48swapunits.
    b = record(4, 'bounded', 'true', 'known', 61)
    x = record(4, 'contact', 'true', 'known', 61)
    cb = record(4, 'bounded', 'true', 'known', 61, 'reverse_impulse')
    cx = record(4, 'contact', 'true', 'known', 61, 'reverse_impulse')
    missing = json.loads((d.BASE / 'evaluation/test/n6/perception/summary.json').read_text())['groups']['all']['metrics']['missing_fraction']
    conditions = {'count4_position_20percent_better': x['state']['position_mae_px'] <= .8 * b['state']['position_mae_px'],
                  'count4_nontarget_response_not_worse_5percent': cx['state']['nontarget_response_error_px'] <= 1.05 * cb['state']['nontarget_response_error_px'],
                  'reader_count6_missing_atmost5percent': missing <= .05}
    for n in d.COUNTS:
        a = record(n, 'contact', 'true', 'known', 61)['state']
        conditions.update({f'n{n}_failed_below1percent': a['failed_path'] < .01, f'n{n}_wall_below1percent': a['wall_path_gt1px'] < .01,
                           f'n{n}_overlap_below5percent': a['overlap_path_gt0_5px'] < .05})
    for k in range(3):
        a = record(2, 'contact', 'true', 'known', 16, group=f'stratum{k}')['state']
        conditions[f'n2_stratum{k}_16step_atmost1px'] = a['position_mae_px'] <= 1
    gate = all(conditions.values())
    dump(d.BASE / 'development_gate.json', {'evaluation_protocol_sha256': sha(d.BASE / 'evaluation_protocol.json'), 'conditions': conditions,
                                          'passed': gate, 'confirmation_required': gate, 'confirmation_completed': False,
                                          'interpretation': 'One development dataset. Passing requiresnewthree-dataset/three-init confirmations; failurestopsautomaticexpansion. Allswapresults retained.'})
    dump(d.BASE / 'verification.json', {'evaluation_protocol_sha256': sha(d.BASE / 'evaluation_protocol.json'), 'data_verification_sha256': sha(d.BASE / 'data_verification.json'),
                                       'training_verification_sha256': sha(d.BASE / 'training_verification.json'), 'unit_verifications': receipts,
                                       'full_RGB_replays': rgb, 'full_trajectory_replays': trajectories, 'gate_sha256': sha(d.BASE / 'development_gate.json'),
                                       'all_development_units_complete': True, 'full_R5_complete': False, 'independent_human_evaluation': False})


def main(wait_pid):
    ev.freeze()
    lock(d.BASE / 'evaluation_driver_protocol.json', {'source_sha256': sha(__file__), 'evaluation_protocol_sha256': sha(d.BASE / 'evaluation_protocol.json'),
        'scope': 'All5reader/forcegroups,48factorialtrajectoryunits withbothrenderers andthreeinterventions,4controlgroups, fullsecondforwardreplay, gateaccounting. Completefixeddevelopment only,notallR5.'})
    while not (d.BASE / 'training_complete.json').exists():
        if (d.BASE / 'training_failure.json').exists():
            raise RuntimeError('Training failed; inspect preserved failure record')
        os.kill(wait_pid, 0)
        dump(d.BASE / 'evaluation_progress.json', {'driver_pid': os.getpid(), 'stage': 'waiting_for_training', 'live_dependency_pid': wait_pid, 'unix': time.time()})
        time.sleep(20)
    assert (d.BASE / 'metric_preflight.json').exists()
    run('verify_training', ['verify_training'])
    for split, count in [('val', 2)] + [('test', n) for n in d.COUNTS]:
        args = ['perceive', '--split', split, '--count', str(count)]
        run(f'{split}_n{count}_perceive', args)
        run(f'{split}_n{count}_perceive_verify', [*args, '--verify'])
    for count in d.COUNTS:
        args = ['controls', '--count', str(count)]
        run(f'n{count}_controls', args)
        run(f'n{count}_controls_verify', [*args, '--verify'])
        for arm in ('bounded', 'contact', 'coarse'):
            for state in ('true', 'learned'):
                for force in ('known', 'inferred'):
                    label = f'n{count}_{arm}_{state}_{force}'
                    args = ['evaluate', '--count', str(count), '--arm', arm, '--state', state, '--force', force]
                    run(label, args)
                    run(label + '_verify', [*args, '--verify'])
    close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--wait-pid', type=int, required=True)
    args = p.parse_args()
    try:
        main(args.wait_pid)
    except Exception:
        dump(d.BASE / 'evaluation_failure.json', {'unix': time.time(), 'traceback': traceback.format_exc()})
        raise
