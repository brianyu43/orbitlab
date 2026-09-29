"""Continue the already-running length audit with gated aggregation and reports."""
import json
import os
import subprocess
import sys
import time
from common import HERE,ROOT,dump,r


def main():
    folder=HERE/'reports/dynamics_observation_length_v1';progress=folder/'completion_pipeline_progress.json'
    if progress.exists():raise RuntimeError('Existing completion pipeline retained; inspect its session before a new run')
    started=time.time();events=[]
    while True:
        queue=json.loads((folder/'verification_queue_progress.json').read_text())
        if queue['status']=='completed':break
        if queue['status']!='running':raise RuntimeError('Audit queue is not running or complete: '+str(queue['status']))
        if time.time()-started>3600:raise TimeoutError('Waited one hour; original audit not restarted')
        dump(progress,{'status':'waiting_for_active_verification_queue','pid':os.getpid(),'source_sha256':r.sha(__file__),
            'queue_completed_stages':queue.get('completed_stages'),'queue_expected_stages':queue.get('expected_stages'),
            'queue_current':queue.get('current'),'started_unix':started,'events':events})
        time.sleep(10)
    steps=[('aggregate','aggregate_observation_length.py',['--stage','all']),
           ('aggregate_verification','verify_length_aggregate.py',['--stage','all']),
           ('report','report_observation_length.py',[]),
           ('report_verification','verify_observation_length_report.py',[])]
    for label,script,args in steps:
        log=HERE/f'logs/observation_length_final_{label}_v1.log';command=[sys.executable,str(HERE/script),*args]
        record={'stage':label,'command':command,'executor_sha256':r.sha(HERE/script),'log':str(log.relative_to(HERE)), 'started_unix':time.time()}
        with log.open('x') as output:
            process=subprocess.Popen(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1','MPLCONFIGDIR':'/tmp/orbitlab-matplotlib'})
            record['pid']=process.pid
            dump(progress,{'status':'running','pid':os.getpid(),'source_sha256':r.sha(__file__),'current':record,'started_unix':started,'events':events})
            print('length completion start',label,flush=True);code=process.wait()
        record.update(exit_code=code,finished_unix=time.time());events.append(record)
        if code:
            dump(progress,{'status':'failed','pid':os.getpid(),'current':record,'started_unix':started,'events':events})
            raise RuntimeError('Length completion stage failed: '+label)
        print('length completion done',label,flush=True)
    dump(progress,{'status':'complete','started_unix':started,'finished_unix':time.time(),'events':events,
        'source_sha256':r.sha(__file__),'scope':'All length aggregates and report numerically checked. Visual inspection and C07 synthesis remain separate.'})


if __name__=='__main__':main()
