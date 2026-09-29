"""After dSprites development, execute CLEVR then CLEVRTex without resampling."""
import argparse,json,os,subprocess,time,traceback
from v3_common import HERE,ROOT,sha,dump,lock,event
from r5_intake import BASE
import r5_3d_train as training
import r5_3d_evaluate as evaluation

def run(label,args):
    log=BASE/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
    with log.open('a') as stream:
        child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/args[0]),*args[1:]],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        dump(BASE/'3d_development_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'stage':label,'unix':time.time()});code=child.wait()
    assert code==0,f'{label} exit{code}; see{log}'

def main(prior_pid):
    for family in ('clevr','clevrtex'):
        training.FAMILY=family;training.freeze();evaluation.freeze()
    lock(BASE/'3d_development_driver_protocol.json',{'source_sha256':sha(__file__),'order':['clevr','clevrtex'],'task':'Single_Atomic','kinds':list(training.KINDS),'updates':216000,
        'sequence':'Wait for verified completion record of allfour dSprites development tasks before CLEVR; complete allCLEVRpixelunits before CLEVRTex. No GPUrun started while dependency incomplete.',
        'scope':'Three20epochdevelopmentarms per3Dfamily,5/10/20val-onlyselection,copycontrol,selected/fixed20test andfullpixelreplay. Attributes,conditionalconfirmationanddisks remain separate.'})
    dependency=BASE/'dsprites_development_complete.json'
    while not dependency.exists():
        os.kill(prior_pid,0);training.resource_check()
        dump(BASE/'3d_development_progress.json',{'driver_pid':os.getpid(),'stage':'waiting_for_dsprites_development','verified_live_dependency_pid':prior_pid,'unix':time.time()});time.sleep(30)
    complete=json.loads(dependency.read_text());assert complete['driver_protocol_sha256']==sha(BASE/'dsprites_development_driver_protocol.json') and complete['development_full_updates']==432000
    task='Single_Atomic'
    for family in ('clevr','clevrtex'):
        for kind in training.KINDS:run(f'{family}_{kind}_s0_train',['r5_3d_train.py','--family',family,'--task',task,'--kind',kind])
        run(f'{family}_selection',['r5_3d_evaluate.py','--family',family,'select','--task',task])
        for kind in ('copy',*training.KINDS):
            for mode in (['validation_selected'] if kind=='copy' else ['validation_selected','fixed20epoch']):
                for stage in ('evaluate','verify'):run(f'{family}_{kind}_{mode}_{stage}',['r5_3d_evaluate.py','--family',family,stage,'--task',task,'--kind',kind,'--mode',mode])
        dump(BASE/f'{family}_development_complete.json',{'unix':time.time(),'driver_protocol_sha256':sha(BASE/'3d_development_driver_protocol.json'),'updates':108000,'attributes_completed':False,'confirmation_completed':False})
    event('R5_3d_development_pixels_complete',updates=216000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dsprites-driver-pid',type=int,required=True);a=p.parse_args()
    try:main(a.dsprites_driver_pid)
    except Exception:
        dump(BASE/'3d_development_failure.json',{'unix':time.time(),'traceback':traceback.format_exc()});raise
