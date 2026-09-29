"""Resumable sequential branch driver with durable start/exit records."""
from pathlib import Path
import argparse,json,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[1];BASE=Path(__file__).resolve().parent

def wait_for(path):
    start=time.time()
    while not path.exists():
        if time.time()-start>7200:raise TimeoutError(str(path))
        time.sleep(5)

def run(label,script,*args):
    folder=BASE/'execution';folder.mkdir(exist_ok=True)
    cmd=[sys.executable,str(BASE/script),*map(str,args)]
    with (folder/'journal.jsonl').open('a') as journal:journal.write(json.dumps({'time':time.time(),'label':label,'event':'start','command':cmd})+'\n')
    with (folder/f'{label}.log').open('a') as log:
        result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    with (folder/'journal.jsonl').open('a') as journal:journal.write(json.dumps({'time':time.time(),'label':label,'event':'exit','returncode':result.returncode})+'\n')
    if result.returncode:raise RuntimeError(f'{label} failed; see execution/{label}.log')
    print('completed',label,flush=True)

def main(branch):
    if branch=='generation':
        wait_for(BASE/'generation/runs/d880101_s0/training_summary.json')
        run('generation_dev_evaluation','generation_evaluate.py')
        for seed in [880201,880202,880203]:
            for init in range(3):
                run(f'g_train_{seed}_{init}','generation_run.py','train','--seed',seed,'--init',init)
                run(f'g_eval_{seed}_{init}','generation_evaluate.py','--seed',seed,'--init',init)
    elif branch=='perception':
        for kind in ['global','crop']:
            wait_for(BASE/f'perception/runs/{kind}_s0/run.json')
        wait_for(BASE/'perception/data/d881203/manifest.json')
        for kind in ['global','crop']:
            for init in range(3):
                run(f'p_train_{kind}_{init}','perception.py','--kind',kind,'--seed',init)
                for seed in [881201,881202,881203]:
                    run(f'p_eval_{kind}_{init}_{seed}','perception_evaluate.py','evaluate','--kind',kind,'--init',init,'--seed',seed)
    elif branch=='dynamics':
        for kind in ['one','multi','multi_bounded']:wait_for(BASE/f'dynamics/runs/d640031_s0/{kind}/run.json')
        run('d_development_evaluation','dynamics_evaluate.py')
        wait_for(BASE/'dynamics/data/manifest.json')
        for seed in [882201,882202,882203]:
            for init in range(3):
                for kind in ['one','multi','multi_bounded']:
                    run(f'd_train_{seed}_{init}_{kind}','dynamics.py','--kind',kind,'--seed',seed,'--init',init)
                for split in ['test','attribute_ood','count3','count4','force_ood']:
                    run(f'd_eval_{seed}_{init}_{split}','dynamics_evaluate.py','--seed',seed,'--init',init,'--split',split)
    (BASE/f'{branch}_driver_completed.json').write_text(json.dumps({'completed_unix':time.time(),'branch':branch})+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('branch',choices=['generation','perception','dynamics']);main(p.parse_args().branch)
