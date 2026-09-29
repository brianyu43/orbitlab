"""Sequential local R4 development jobs with liveness and durable failure logs."""
import json, os, subprocess, sys, time, traceback
from v3_common import HERE, dump, lock, sha, status, event
from r4_data import BASE, DEVELOPMENT
from r4_train import COMPONENTS, RUNS, cap_check, freeze

def execute(args, name):
    log=BASE/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(HERE/'r4_train.py'),*args]
    with log.open('a') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        dump(BASE/'driver_progress.json',{'driver_pid':os.getpid(),'child_pid':p.pid,'stage':name,'command':command,'started_unix':time.time()})
        print('R4 start',name,'pid',p.pid,flush=True)
        while p.poll() is None:
            time.sleep(10)
            try:cap_check()
            except BaseException:
                p.terminate();p.wait();raise
        if p.returncode:raise RuntimeError(f'{name}failed with exit{p.returncode}; see{log}')

def main():
    freeze();lock(BASE/'driver_protocol.json',{'source_sha256':sha(__file__),'training_protocol_sha256':sha(BASE/'training_protocol.json'),'order':COMPONENTS,'development_only':True,'confirmation':'Never launched by this driver. Evaluation gate and explicit candidate lock first.'})
    assert json.loads((BASE/'preflight.json').read_text())['passed']
    status('R4','development_training',components=COMPONENTS,full_updates_planned=100000,confirmation_started=False)
    for component in COMPONENTS:
        if not component.startswith('ae_') and not (RUNS/f'd{DEVELOPMENT}_s0/pools/manifest.json').exists():execute(['pools'],'pools')
        execute([component,'--benchmark'],component+'_benchmark')
        benchmark=json.loads((RUNS/f'd{DEVELOPMENT}_s0/benchmarks'/component/'run.json').read_text())
        # Conservative forecast for all as-yet unfinished optimizer updates.
        observed=[json.loads(p.read_text())['seconds_per_update'] for p in (RUNS/f'd{DEVELOPMENT}_s0/benchmarks').glob('*/run.json')]
        from r4_train import settings
        remaining=sum(settings(c)[0] for c in COMPONENTS if not (RUNS/f'd{DEVELOPMENT}_s0'/c/'run.json').exists())
        forecast=max(observed)*remaining
        cap_check(additional_seconds=forecast)
        dump(BASE/'resource_forecast.json',{'current_component':component,'remaining_updates':remaining,'conservative_training_seconds':forecast,'max_observed_seconds_per_update':max(observed),'evaluation_time_not_in_forecast':True})
        execute([component],component)
    runs={c:json.loads((RUNS/f'd{DEVELOPMENT}_s0'/c/'run.json').read_text()) for c in COMPONENTS}
    assert sum(v['steps'] for v in runs.values())==100000
    for role in ['oracle','learned']:
        a,b=[runs[f'decoder_{role}_{c}'] for c in ['standard','edge']];assert a['initial_state_sha256']==b['initial_state_sha256']
    dump(BASE/'training_complete.json',{'components':{c:sha(RUNS/f'd{DEVELOPMENT}_s0'/c/'run.json') for c in runs},'updates':100000,'pilot_updates':100*len(runs),'seconds':sum(v['seconds'] for v in runs.values()),'confirmation_started':False})
    status('R4','development_trained_evaluation_pending',full_updates=100000,confirmation_started=False)
    print('R4 all training complete',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        dump(BASE/'driver_failure.json',{'error':str(e),'traceback':traceback.format_exc(),'unix':time.time(),'driver_pid':os.getpid()});event('R4_driver_failed',error=str(e));raise
