"""Single writer for the R2 development benchmark and six full training arms."""
import fcntl,json,os,subprocess,sys,time
import r2_core as c
import r2_world as w
from v3_common import HERE,ROOT,dump,event,status

def run(arm,benchmark=False):
    folder=w.BASE/'execution';folder.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(HERE/'r2_train.py'),'--seed',str(w.DEVELOPMENT),'--arm',arm]
    if benchmark:command.append('--benchmark')
    with (folder/f'{arm}_{"benchmark" if benchmark else "training"}.log').open('a') as f:
        p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(w.BASE/'live_process.json',{'driver_pid':os.getpid(),'worker_pid':p.pid,'arm':arm,'benchmark':benchmark,'started_unix':time.time()})
        event('R2_train_command_start',arm=arm,benchmark=benchmark,pid=p.pid)
        rc=p.wait()
    if rc:raise RuntimeError(f'R2 {arm} exit{rc}; retained log and progress checkpoint')

def main():
    w.BASE.mkdir(parents=True,exist_ok=True)
    with (w.BASE/'training_driver.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);cfg=c.freeze();start=time.time();status('R2','benchmarking_then_development_training')
        for arm in c.ARMS:run(arm,True)
        estimate=sum(json.loads((w.BASE/f'benchmark/d{w.DEVELOPMENT}_s0/{arm}/run.json').read_text())['estimated_seconds_per_4000_steps'] for arm in c.ARMS)
        dump(w.BASE/'compute_estimate.json',{'six_arm_estimated_training_seconds':estimate,'cap_seconds':43200,'excludes':'data generation,evaluation,verification,decoder training','actual_timing_probe_steps_per_arm':100})
        if estimate>43200:raise RuntimeError('Estimated training exceeds stage cap; preserve protocol and request resource decision')
        for arm in c.ARMS:
            if time.time()-start>43200:raise RuntimeError('Stage cap reached; preserve partial outputs')
            run(arm)
        dump(w.BASE/'training_complete.json',{'arms':c.ARMS,'seconds':time.time()-start,'evaluation':'pending','decoder':'pending','confirmation':'not_selected'})
        status('R2','models_trained_evaluation_and_decoder_pending');print('R2 six-arm development training complete; evaluation remains pending',flush=True)

if __name__=='__main__':main()
