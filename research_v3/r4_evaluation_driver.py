"""Evaluate when exact checkpoint receipts exist; fail if owned trainer exits."""
import json,os,subprocess,sys,time,traceback
from v3_common import HERE,dump,sha,lock,status,event
from r4_data import BASE,DEVELOPMENT
from r4_train import root,cap_check
from r4_evaluate import freeze,CANDIDATES,ROLES,SPLITS

def wait_for(paths):
    while not all(p.exists() for p in paths):
        if (BASE/'driver_failure.json').exists():raise RuntimeError('R4 training driver recorded a failure; preserve unfinished evaluation')
        if (BASE/'training_complete.json').exists():raise RuntimeError('Training marked complete but prerequisite missing')
        progress=json.loads((BASE/'driver_progress.json').read_text())
        try:os.kill(progress['driver_pid'],0)
        except ProcessLookupError:raise RuntimeError('R4 trainer exited before required checkpoint receipt')
        cap_check();time.sleep(10)

def execute(script,args,name):
    log=BASE/'logs'/f'eval_{name}.log'
    with log.open('a') as f:
        child=subprocess.Popen([sys.executable,str(HERE/script),*args],stdout=f,stderr=subprocess.STDOUT)
        dump(BASE/'evaluation_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'stage':name,'unix':time.time()});print('R4 eval start',name,flush=True)
        while child.poll() is None:
            time.sleep(10)
            try:cap_check()
            except BaseException:child.terminate();child.wait();raise
        if child.returncode:raise RuntimeError(f'R4evaluation{name}failed exit{child.returncode}; see{log}')

def score(args,name):
    execute('r4_evaluate.py',args,name)
    execute('r4_evaluate.py',[*args,'--verify'],name+'_verify')

def main():
    freeze();lock(BASE/'evaluation_driver_protocol.json',{'source_sha256':sha(__file__),'verifier_sha256':sha(HERE/'r4_verify.py'),'evaluation_protocol_sha256':sha(BASE/'evaluation_protocol.json'),'development_only':True})
    assert json.loads((BASE/'metric_preflight.json').read_text())['evaluator_sha256']==sha(HERE/'r4_evaluate.py')
    score(['reference'],'reference')
    for split in SPLITS:score(['reconstruction','--candidate','clean','--role','clean','--split',split],'clean_'+split)
    run=root(DEVELOPMENT,0);wait_for([run/'pools/manifest.json']);execute('r4_verify.py',['pools'],'pools')
    for role in ROLES:
        for candidate in CANDIDATES:
            component=f'decoder_{"oracle" if role=="bridge" else role}_{candidate}'
            dependencies=[run/component/'run.json']
            if role=='bridge':dependencies.append(run/f'bridge_{"spatial" if candidate=="spatial" else "base"}/run.json')
            wait_for(dependencies)
            for split in SPLITS:score(['reconstruction','--candidate',candidate,'--role',role,'--split',split],f'{role}_{candidate}_{split}')
    wait_for([BASE/'training_complete.json']);execute('r4_verify.py',['training'],'training')
    for rep in ['base','spatial']:score(['flow','--representation',rep],'flow_'+rep)
    for candidate in CANDIDATES:score(['generation','--candidate',candidate],'generation_'+candidate)
    execute('r4_evaluate.py',['aggregate'],'aggregate')
    dump(BASE/'evaluation_complete.json',{'evaluation_verification_sha256':sha(BASE/'evaluation_verification.json'),'training_verification_sha256':sha(BASE/'training_verification.json'),'source_sha256':sha(__file__)})
    selected=json.loads((BASE/'selection.json').read_text());status('R4','development_evaluated_report_pending',development_gate_passed=selected['gate_passed'],selection='r4/selection.json',confirmation_started=False)
    print('R4 evaluation complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(BASE/'evaluation_driver_failure.json',{'error':str(e),'traceback':traceback.format_exc(),'unix':time.time()});event('R4_evaluation_driver_failed',error=str(e));raise
