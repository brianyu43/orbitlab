"""Incremental full validation replay after each completed development model."""
import argparse,json,os,subprocess,time,traceback
from v3_common import HERE,ROOT,sha,dump,lock
from r5_intake import BASE,TASKS
from r5_attribute_evaluate import backend
from r5_train import resource_check

def main(family,parent_pid):
    assert parent_pid>1;mod=backend(family);tr=mod.training;tasks=TASKS if family=='dsprites' else ('Single_Atomic',)
    protocol=lock(BASE/f'{family}_verification_driver_protocol.json',{'source_sha256':sha(__file__),'verifier_sha256':sha(HERE/'r5_verify_external.py'),'family':family,'tasks':list(tasks),'initialization':0,'partition_seed':890101,'all_val_forwards':len(tasks)*3*3*6400})
    def wait_for(path,label):
        while not path.exists():
            os.kill(parent_pid,0);resource_check();dump(BASE/f'{family}_verification_progress.json',{'driver_pid':os.getpid(),'stage':'waiting_'+label,'live_parent_pid':parent_pid,'unix':time.time()});time.sleep(30)
    def run(label,args):
        assert sha(HERE/'r5_verify_external.py')==protocol['verifier_sha256']
        with (BASE/'logs'/f'{label}.log').open('a') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/'r5_verify_external.py'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            dump(BASE/f'{family}_verification_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'stage':label,'unix':time.time()});code=child.wait()
        assert code==0,f'{label} exit{code}'
    for task in tasks:
        for kind in ('plain','c4','object'):
            wait_for(tr.location(task,kind,0)/'run.json',f'{task}_{kind}_training')
            run(f'verify_{family}_{task}_{kind}',['training','--family',family,'--task',task,'--kind',kind])
        for kind in ('copy','plain','c4','object'):
            for mode in (['validation_selected'] if kind=='copy' else ['validation_selected','fixed20epoch']):wait_for(mod.unit(task,kind,0,mode)/'verification.json',f'{task}_{kind}_{mode}_pixels')
        run(f'semantic_groups_{family}_{task}',['semantics','--family',family,'--task',task])
    dump(BASE/f'{family}_external_training_verified.json',{'unix':time.time(),'protocol_sha256':sha(BASE/f'{family}_verification_driver_protocol.json'),'all_validation_forwards':len(tasks)*3*3*6400,'independent_retraining':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['dsprites','clevr','clevrtex'],default='dsprites');p.add_argument('--model-driver-pid',type=int,required=True);a=p.parse_args()
    try:main(a.family,a.model_driver_pid)
    except Exception:
        dump(BASE/f'{a.family}_verification_failure.json',{'unix':time.time(),'traceback':traceback.format_exc()});raise
