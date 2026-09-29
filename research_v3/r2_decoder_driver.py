"""Run the two frozen decoder budgets after a measured100-step pilot each."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
import r2_decoder_core as c
from v3_common import HERE,ROOT,dump,event

def run(kind,benchmark=False):
    cmd=[sys.executable,str(HERE/'r2_decoder_train.py'),'--kind',kind]
    if benchmark:cmd+=['--benchmark']
    with (c.BASE/f'{kind}_{"benchmark" if benchmark else "training"}.log').open('a') as f:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(c.BASE/'live_process.json',{'driver_pid':os.getpid(),'worker_pid':p.pid,'kind':kind,'benchmark':benchmark,'started_unix':time.time()})
        rc=p.wait()
    if rc:raise RuntimeError(f'Decoder {kind} benchmark={benchmark} exited{rc}')

def main():
    c.freeze()
    with (c.BASE/'driver.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        events=[json.loads(v) for v in (HERE/'journal.jsonl').read_text().splitlines()]
        start=min(v['time'] for v in events if v['stage'].startswith('R2'))
        estimates=[]
        for kind in ['patch','editor']:
            run(kind,True);bench=json.loads((c.BASE/f'benchmark/{kind}/run.json').read_text());elapsed=time.time()-start
            assert elapsed+bench['estimated_full_seconds']<43200,'R2 stage time cap would be exceeded'
            estimates.append({'kind':kind,'estimated_training_seconds':bench['estimated_full_seconds'],'r2_elapsed_at_start':elapsed})
            dump(c.BASE/'compute_estimate.json',estimates);run(kind)
        dump(c.BASE/'training_complete.json',{'ended_unix':time.time(),'patch_steps':2000,'editor_steps':4000,'evaluation':'pending','verification':'pending'})
        event('R2_decoder_training_complete_all')

if __name__=='__main__':main()
