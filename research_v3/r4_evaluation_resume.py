"""Resume unchanged R4scores with corrected source-setting-aware pool replay."""
import json,time,traceback
from v3_common import HERE,sha,dump,lock,status,event
from r4_data import BASE,DEVELOPMENT
from r4_train import root
from r4_evaluate import freeze,CANDIDATES,ROLES,SPLITS,EVAL
from r4_evaluation_driver import execute,wait_for,score

def verified(folder):
    if not (folder/'verification.json').exists():return False
    v=json.loads((folder/'verification.json').read_text())
    return v.get('summary_sha256')==sha(folder/'summary.json')

def main():
    freeze();lock(BASE/'evaluation_resume_protocol.json',{'source_sha256':sha(__file__),'original_driver_sha256':sha(HERE/'r4_evaluation_driver.py'),'pool_verifier_sha256':sha(HERE/'r4_pool_replay_v2.py'),'evaluation_protocol_sha256':sha(BASE/'evaluation_protocol.json'),'numerical_scores_and_gates_changed':False})
    if not verified(EVAL/'reference'):score(['reference'],'reference')
    for split in SPLITS:
        if not verified(EVAL/'reconstruction/clean_clean'/split):score(['reconstruction','--candidate','clean','--role','clean','--split',split],'clean_'+split)
    run=root(DEVELOPMENT,0);wait_for([run/'pools/manifest.json']);execute('r4_pool_replay_v2.py',[],'pools_v2')
    for role in ROLES:
        for candidate in CANDIDATES:
            component=f'decoder_{"oracle" if role=="bridge" else role}_{candidate}';dependencies=[run/component/'run.json']
            if role=='bridge':dependencies.append(run/f'bridge_{"spatial" if candidate=="spatial" else "base"}/run.json')
            wait_for(dependencies)
            for split in SPLITS:
                if not verified(EVAL/'reconstruction'/f'{role}_{candidate}'/split):score(['reconstruction','--candidate',candidate,'--role',role,'--split',split],f'{role}_{candidate}_{split}')
    wait_for([BASE/'training_complete.json']);execute('r4_verify.py',['training'],'training')
    for rep in ['base','spatial']:
        if not (EVAL/'flow'/rep/'verification.json').exists():score(['flow','--representation',rep],'flow_'+rep)
    for candidate in CANDIDATES:
        if not verified(EVAL/'generation'/candidate):score(['generation','--candidate',candidate],'generation_'+candidate)
    execute('r4_evaluate.py',['aggregate'],'aggregate')
    dump(BASE/'evaluation_complete.json',{'evaluation_verification_sha256':sha(BASE/'evaluation_verification.json'),'training_verification_sha256':sha(BASE/'training_verification.json'),'pool_verification_sha256':sha(run/'pools/verification.json'),'source_sha256':sha(__file__)})
    selected=json.loads((BASE/'selection.json').read_text());status('R4','development_evaluated_report_pending',development_gate_passed=selected['gate_passed'],selection='r4/selection.json',confirmation_started=False)
    print('R4 evaluation complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(BASE/'evaluation_resume_failure.json',{'error':str(e),'traceback':traceback.format_exc(),'unix':time.time()});event('R4_evaluation_resume_failed',error=str(e));raise
