"""Read generated images only after pixel and independent-reader replays pass."""
import argparse,json,os,subprocess,time,traceback
from v3_common import HERE,ROOT,sha,dump,lock
from r5_intake import BASE,TASKS
from r5_attributes import folder as reader_folder
from r5_attribute_evaluate import backend
from r5_train import resource_check

def wait_for(path,parent_pid,label,family):
    while not path.exists():
        os.kill(parent_pid,0);resource_check()
        dump(BASE/f'{family}_attribute_evaluation_progress.json',{'driver_pid':os.getpid(),'stage':'waiting_'+label,'live_dependency_pid':parent_pid,'unix':time.time()});time.sleep(30)

def main(family,model_pid,reader_pid):
    mod=backend(family);tasks=TASKS if family=='dsprites' else ('Single_Atomic',)
    assert model_pid>1 and reader_pid>1
    protocol=lock(BASE/f'{family}_attribute_evaluation_driver_protocol.json',{'source_sha256':sha(__file__),'scorer_sha256':sha(HERE/'r5_attribute_evaluate.py'),'family':family,'tasks':list(tasks),'units_per_task':7,'all_images_per_unit':8000,'repeat_all':True,'initialization':0,'partition_seed':890101})
    for task in tasks:
        wait_for(reader_folder(family,task)/'calibration_test_verification.json',reader_pid,f'{family}_{task}_reader',family)
        for kind in ('copy','plain','c4','object'):
            for mode in (['validation_selected'] if kind=='copy' else ['validation_selected','fixed20epoch']):
                wait_for(mod.unit(task,kind,0,mode)/'verification.json',model_pid,f'{family}_{task}_{kind}_{mode}_pixels',family)
                for verify in (False,True):
                    assert sha(HERE/'r5_attribute_evaluate.py')==protocol['scorer_sha256']
                    label=f'attributes_{family}_{task}_{kind}_{mode}_{verify}';log=BASE/'logs'/f'{label}.log'
                    args=[str(ROOT/'.venv/bin/python'),str(HERE/'r5_attribute_evaluate.py'),'--family',family,'--task',task,'--kind',kind,'--mode',mode]
                    if verify:args.append('--verify')
                    with log.open('a') as stream:
                        child=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                        dump(BASE/f'{family}_attribute_evaluation_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'stage':label,'unix':time.time()});code=child.wait()
                    assert code==0,f'{label} exit{code}; see{log}'
    dump(BASE/f'{family}_generated_attributes_complete.json',{'unix':time.time(),'driver_protocol_sha256':sha(BASE/f'{family}_attribute_evaluation_driver_protocol.json'),'all_images':len(tasks)*7*8000,'all_object_readouts':len(tasks)*7*16000,'confirmation_scored':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['dsprites','clevr','clevrtex'],default='dsprites');p.add_argument('--model-driver-pid',type=int,required=True);p.add_argument('--reader-driver-pid',type=int,required=True);a=p.parse_args()
    try:main(a.family,a.model_driver_pid,a.reader_driver_pid)
    except Exception:
        dump(BASE/f'{a.family}_attribute_evaluation_failure.json',{'unix':time.time(),'traceback':traceback.format_exc()});raise
