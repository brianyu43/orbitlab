"""CPU-only independent attribute readers; source-data calibration and replay."""
import argparse,subprocess,os,time,traceback
from v3_common import HERE,ROOT,sha,dump,lock
from r5_intake import BASE,TASKS
from r5_attributes import freeze

def main(family):
    tasks=TASKS if family=='dsprites' else ('Single_Atomic',)
    for task in tasks:freeze(family,task)
    lock(BASE/f'{family}_attribute_driver_protocol.json',{'source_sha256':sha(__file__),'family':family,'tasks':list(tasks),'per_reader_updates':6000,'calibration_replay':'All6400validation+8000testscenes,source/target,twoobjects','scope':'Measurement instruments only. Generated-image attribute evaluation is separate.'})
    for task in tasks:
        units=[('train',None),('calibrate','val'),('verify_calibration','val'),('calibrate','test'),('verify_calibration','test')]
        for stage,split in units:
            label=f'attribute_{family}_{task}_{stage}_{split}';log=BASE/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
            args=[str(ROOT/'.venv/bin/python'),str(HERE/'r5_attributes.py'),stage,'--family',family,'--task',task]
            if split:args+=['--split',split]
            with log.open('a') as stream:
                child=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                dump(BASE/f'{family}_attribute_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'stage':label,'unix':time.time()});code=child.wait()
            assert code==0,f'{label} exit{code}; see{log}'
    dump(BASE/f'{family}_attribute_readers_complete.json',{'unix':time.time(),'driver_protocol_sha256':sha(BASE/f'{family}_attribute_driver_protocol.json'),'models':len(tasks),'generated_images_scored':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('family',choices=['dsprites','clevr','clevrtex']);a=p.parse_args()
    try:main(a.family)
    except Exception:
        dump(BASE/f'{a.family}_attribute_driver_failure.json',{'traceback':traceback.format_exc(),'unix':time.time()});raise
