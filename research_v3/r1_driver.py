"""Development, frozen gate, conditional confirmation, with single-writer lock."""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
import r1_core as c
from v3_common import ROOT, HERE, dump, lock, sha, event, status

def run(script, *args):
    log = c.BASE/'execution'/('_'.join([script.removesuffix('.py'), *map(str,args)]).replace('--','')+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    event('R1_command_start', command=[script, *args], log=str(log.relative_to(HERE)))
    with log.open('a') as f:
        p = subprocess.Popen([sys.executable, str(HERE/script), *map(str,args)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        dump(c.BASE/'live_process.json', {'driver_pid': os.getpid(), 'worker_pid': p.pid, 'command': [script,*args], 'started_unix': time.time()})
        result = p.wait()
    if result: raise RuntimeError(f'Exit {result}; inspect {log}')
    event('R1_command_complete', command=[script,*args])

def select():
    baseline = 'alpha_full'; grid = {}; sources = {}
    for arm in c.ARMS:
        grid[arm] = {}
        for split in ['val','ood']:
            grid[arm][split] = {}
            for renderer in c.RENDERERS:
                p=c.BASE/f'evaluation/d{c.DEVELOPMENT}_s0/{arm}/{split}/{renderer}/summary.json'
                grid[arm][split][renderer] = json.loads(p.read_text())['metrics']['quotient_edit']
                sources[str(p.relative_to(c.BASE))] = sha(p)
    def rank(arm):
        cells = [grid[arm][s][r] for s in ['val','ood'] for r in c.RENDERERS[1:]]
        drops = [grid[arm][s][c.RENDERERS[0]]-grid[arm][s][r] for s in ['val','ood'] for r in c.RENDERERS[1:]]
        return (-sum(cells)/len(cells),max(drops),c.ARMS.index(arm))
    candidate = min([a for a in c.ARMS if a != baseline],key=rank)
    comparisons = [{'split': s,'renderer':r,'gain':grid[candidate][s][r]-grid[baseline][s][r],
        'drop':grid[candidate][s][c.RENDERERS[0]]-grid[candidate][s][r]} for s in ['val','ood'] for r in c.RENDERERS[1:]]
    passed = all(v['gain']>=.05 and v['drop']<=.05 for v in comparisons)
    result={'baseline':baseline,'candidate':candidate,'development_gate_passed':passed,'comparisons':comparisons,'grid':grid,'sources':sources,
        'action':'confirm_baseline_and_candidate' if passed else 'stop_expansion_and_verify_development_failure_analysis',
        'interpretation':'A failed gate is not a robustness claim. No confirmation test results were used for selection.'}
    lock(c.BASE/'selection.json',result);return result

def main():
    c.BASE.mkdir(parents=True,exist_ok=True)
    with (c.BASE/'driver.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time();c.freeze();status('R1','development_running')
        run('r1_data.py','--seed',c.DEVELOPMENT)
        for arm in c.ARMS: run('r1_train.py','--seed',c.DEVELOPMENT,'--arm',arm,'--benchmark')
        for arm in c.ARMS:
            run('r1_train.py','--seed',c.DEVELOPMENT,'--arm',arm)
            run('r1_evaluate.py','--seed',c.DEVELOPMENT,'--arm',arm)
        run('r1_evaluate.py','--seed',c.DEVELOPMENT,'--arm','template')
        decision=select();status('R1','development_gate_evaluated',selection='r1/selection.json',gate_passed=decision['development_gate_passed'])
        if decision['development_gate_passed']:
            for seed in c.CONFIRMATION:
                run('r1_data.py','--seed',seed)
                for init in [0,1,2]:
                    for arm in [decision['baseline'],decision['candidate']]:
                        if time.time()-started>43200: raise RuntimeError('Resource time cap; retain incomplete experiment')
                        run('r1_train.py','--seed',seed,'--init',init,'--arm',arm)
                        run('r1_evaluate.py','--seed',seed,'--init',init,'--arm',arm)
                run('r1_evaluate.py','--seed',seed,'--arm','template')
        dump(c.BASE/'execution_complete.json',{'ended_unix':time.time(),'driver_seconds':time.time()-started,'selection_sha256':sha(c.BASE/'selection.json'),'verification':'pending'})
        status('R1','experiments_finished_verification_pending')
        print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__': main()
