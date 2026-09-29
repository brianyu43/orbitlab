"""Sequential3Dreaders and complete3Dvalidation/attribute replays after dSprites."""
import argparse,json,os,subprocess,time,traceback
from v3_common import HERE,ROOT,sha,dump,lock
from r5_intake import BASE
from r5_train import resource_check

def main(model_pid,prior_reader_pid):
    assert model_pid>1 and prior_reader_pid>1
    protocol=lock(BASE/'3d_auxiliary_driver_protocol.json',{'source_sha256':sha(__file__),
        'dependencies':{n:sha(HERE/n) for n in ['r5_attribute_driver.py','r5_attributes.py','r5_verification_driver.py','r5_verify_external.py','r5_attribute_evaluation_driver.py','r5_attribute_evaluate.py']},
        'reader_order':['after_dsprites_readers','clevr','clevrtex'],'validation_and_attribute_scope':'All5/10/20validationforwards,allselected/fixed20testattributeoutputs,fullreplay fordevelopmentonly. Conditionalconfirmationremainsseparate.'})
    def run(label,args):
        for n,h in protocol['dependencies'].items():assert sha(HERE/n)==h
        with (BASE/'logs'/f'{label}.log').open('a') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(HERE/args[0]),*args[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            dump(BASE/'3d_auxiliary_progress.json',{'driver_pid':os.getpid(),'child_pid':child.pid,'stage':label,'unix':time.time()});code=child.wait()
        assert code==0,f'{label} exit{code}'
    while not (BASE/'dsprites_attribute_readers_complete.json').exists():
        os.kill(prior_reader_pid,0);resource_check();dump(BASE/'3d_auxiliary_progress.json',{'driver_pid':os.getpid(),'stage':'waiting_for_dsprites_readers','live_dependency_pid':prior_reader_pid,'unix':time.time()});time.sleep(30)
    for family in ('clevr','clevrtex'):run(f'{family}_attribute_readers',['r5_attribute_driver.py',family])
    for family in ('clevr','clevrtex'):
        run(f'{family}_complete_validation_replay',['r5_verification_driver.py','--family',family,'--model-driver-pid',str(model_pid)])
        run(f'{family}_generated_attribute_replay',['r5_attribute_evaluation_driver.py','--family',family,'--model-driver-pid',str(model_pid),'--reader-driver-pid',str(os.getpid())])
    dump(BASE/'3d_auxiliary_complete.json',{'unix':time.time(),'protocol_sha256':sha(BASE/'3d_auxiliary_driver_protocol.json'),'all_development_readers_validation_and_attributes_verified':True,'confirmation_complete':False,'integrated_disks_complete':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-driver-pid',type=int,required=True);p.add_argument('--dsprites-reader-driver-pid',type=int,required=True);a=p.parse_args()
    try:main(a.model_driver_pid,a.dsprites_reader_driver_pid)
    except Exception:
        dump(BASE/'3d_auxiliary_failure.json',{'unix':time.time(),'traceback':traceback.format_exc()});raise
