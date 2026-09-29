"""R3 development and conditional confirmation, with explicit scientific gates."""
import fcntl,json,os,subprocess,sys,time
import r3_core as c
from v3_common import ROOT,HERE,dump,lock,sha,event,status

def run(script,*args):
    out=c.BASE/'execution'/('_'.join([script.removesuffix('.py'),*map(str,args)]).replace('--','')+'.log');out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('a') as f:
        p=subprocess.Popen([sys.executable,str(HERE/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(c.BASE/'live_process.json',{'driver_pid':os.getpid(),'worker_pid':p.pid,'command':[script,*args],'started_unix':time.time()})
        event('R3_command_start',command=[script,*args],pid=p.pid)
        rc=p.wait()
    if rc:raise RuntimeError(f'Exit {rc}; inspect {out}')
    event('R3_command_complete',command=[script,*args])

def select():
    grid={};sources={}
    for arm in c.ARMS+['coarse_physics','force_wall']:
        grid[arm]={}
        for count in [2,3,4,6]:
            p=c.BASE/f'evaluation/d{c.DEVELOPMENT}_s0/{arm}/n{count}/known_force/summary.json'
            data=json.loads(p.read_text());grid[arm][str(count)]=data['metrics'];sources[str(p.relative_to(c.BASE))]=sha(p)
    def rank(arm):
        return sum(grid[arm][str(n)][str(h)]['all']['position_mae_all'] for n in [2,3,4,6] for h in [16,61])/8
    candidate=min(['mean','local_mean','contact_residual'],key=rank);base=grid['multi_bounded'];cand=grid[candidate]
    b=base['4']['61']['all'];p=cand['4']['61']['all'];checks={
        'count4_position_at_least_20pct_better':p['position_mae_all']<=.8*b['position_mae_all'],
        'count4_nontarget_response_within_5pct':p['nontarget_response_mae']<=1.05*b['nontarget_response_mae'],
        'all_counts_failure_below_1pct':all(cand[str(n)]['61']['all']['failure']<.01 for n in [2,3,4,6]),
        'all_counts_wall_below_1pct':all(cand[str(n)]['61']['all']['wall_over_1px']<.01 for n in [2,3,4,6]),
        'all_counts_overlap_below_5pct':all(cand[str(n)]['61']['all']['overlap_over_1px']<.05 for n in [2,3,4,6]),
        'count2_contact_strata_pilot':all(cand['2']['16'][str(k)]['position_mae_all']<=1 for k in [0,1,2])}
    passed=all(checks.values())
    result={'candidate':candidate,'baseline':'multi_bounded','development_gate_passed':passed,'checks':checks,
        'ranking':{arm:rank(arm) for arm in ['mean','local_mean','contact_residual']},
        'count4_61_baseline':b,'count4_61_candidate':p,'count4_61_coarse_reference':grid['coarse_physics']['4']['61']['all'],
        'source_hashes':sources,'next_action':'confirm_fresh_3x3' if passed else 'diagnose_failures_without_automatic_confirmation_expansion',
        'physical_prior_warning':'A contact-residual advantage over a neural-only baseline cannot be attributed solely to learned representation. Always compare the untrained coarse physics reference and exact simulator oracle.'}
    lock(c.BASE/'selection.json',result);return result

def main():
    c.BASE.mkdir(parents=True,exist_ok=True)
    with (c.BASE/'driver.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();c.freeze();status('R3','development_running')
        run('r3_data.py','--seed',c.DEVELOPMENT)
        for arm in c.ARMS:run('r3_train.py','--seed',c.DEVELOPMENT,'--arm',arm,'--benchmark')
        for arm in c.ARMS:
            if time.time()-start>43200:raise RuntimeError('Resource cap reached; retain incomplete experiment')
            run('r3_train.py','--seed',c.DEVELOPMENT,'--arm',arm)
            run('r3_evaluate.py','--seed',c.DEVELOPMENT,'--arm',arm)
        for arm in ['coarse_physics','force_wall']:run('r3_evaluate.py','--seed',c.DEVELOPMENT,'--arm',arm)
        decision=select();status('R3','development_gate_evaluated',gate_passed=decision['development_gate_passed'])
        if decision['development_gate_passed']:
            for seed in c.CONFIRMATION:
                run('r3_data.py','--seed',seed)
                for init in [0,1,2]:
                    for arm in ['multi_bounded',decision['candidate']]:
                        if time.time()-start>43200:raise RuntimeError('Resource cap reached; retain incomplete experiment')
                        run('r3_train.py','--seed',seed,'--init',init,'--arm',arm)
                        run('r3_evaluate.py','--seed',seed,'--init',init,'--arm',arm)
                for arm in ['coarse_physics','force_wall']:run('r3_evaluate.py','--seed',seed,'--arm',arm)
        dump(c.BASE/'execution_complete.json',{'ended_unix':time.time(),'seconds':time.time()-start,'selection_sha256':sha(c.BASE/'selection.json'),'verification':'pending'})
        status('R3','experiments_finished_verification_pending');print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
