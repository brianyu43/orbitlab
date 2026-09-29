"""Bounded sequential official dSprites development; no hidden confirmation."""
import json,os,subprocess,time,traceback
from v3_common import HERE,ROOT,sha,dump,lock,event
from r5_intake import BASE,TASKS
from r5_train import freeze as freeze_training,KINDS,resource_check
from r5_evaluate import freeze as freeze_evaluation

def run(label,args):
    log=BASE/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        p=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/args[0]),*args[1:]],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        dump(BASE/'development_progress.json',{'driver_pid':os.getpid(),'child_pid':p.pid,'stage':label,'started_unix':time.time(),'log':str(log.relative_to(HERE))})
        code=p.wait()
    assert code==0,f'{label} failed with exit{code}; see {log}'

def main():
    freeze_training();freeze_evaluation()
    lock(BASE/'dsprites_development_driver_protocol.json',{'source_sha256':sha(__file__),'training_protocol_sha256':sha(BASE/'dsprites_training_protocol.json'),
        'evaluation_protocol_sha256':sha(BASE/'dsprites_evaluation_protocol.json'),'order':list(TASKS),'initialization':0,'partition_seed':890101,
        'primary_training_updates':4*3*36000,'scope':'Four task development arms, then validation-only selection, full official test pixels and full replay for each task. Three-by-three selected confirmation is a separate required conditional queue; no claim it ran. Attribute reader, CLEVR/CLEVRTex and disks are additional required work.',
        'restart':'Completed hash-matched runs/units reused; partial training restores saved optimizer and RNG. Partial evaluation caches require explicit recovery; never silently replace.'})
    for task in TASKS:
        for kind in KINDS:run(f'{task}_{kind}_s0_train',['r5_train.py','--task',task,'--kind',kind])
        run(f'{task}_selection',['r5_evaluate.py','select','--task',task])
        for kind in ('copy',*KINDS):
            for mode in (['validation_selected'] if kind=='copy' else ['validation_selected','fixed20epoch']):
                for stage in ['evaluate','verify']:
                    run(f'{task}_{kind}_{mode}_{stage}',['r5_evaluate.py',stage,'--task',task,'--kind',kind,'--mode',mode])
        resource_check()
    dump(BASE/'dsprites_development_complete.json',{'unix':time.time(),'driver_protocol_sha256':sha(BASE/'dsprites_development_driver_protocol.json'),'development_full_updates':432000,'attributes_complete':False,'confirmation_completed':False,'full_R5_complete':False})
    event('R5_dsprites_development_pixels_complete',updates=432000)

if __name__=='__main__':
    try:main()
    except Exception:
        dump(BASE/'development_failure.json',{'unix':time.time(),'traceback':traceback.format_exc()});raise
