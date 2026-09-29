"""Bounded queue for the ten remaining data/length evaluations; no new training."""
import json
import subprocess
import sys
import time
from common import HERE,dump,r

PAIRS=[(640031,4),(640031,8)]+[(ds,length) for ds in [997101,997102] for length in [1,2,4,8]]


def main():
    root=HERE/'reports/dynamics_observation_length_v1';results=[];started=time.time()
    for ds,length in PAIRS:
        log=HERE/f'logs/observation_length_eval_d{ds}_L{length}_v1.log'
        command=[sys.executable,str(HERE/'dynamics_observation_length_eval.py'),'--data-seed',str(ds),'--length',str(length),'--stage','all']
        record={'data_seed':ds,'length':length,'command':command,'log':str(log.relative_to(HERE)),
                'started_unix':time.time(),'executor_sha256':r.sha(HERE/'dynamics_observation_length_eval.py')}
        dump(root/'evaluation_queue_progress.json',{'status':'running','completed_pairs':len(results),'expected_pairs':len(PAIRS),'current':record,'results':results})
        print('length evaluation queue start',ds,length,flush=True)
        with log.open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
            record['pid']=process.pid
            dump(root/'evaluation_queue_progress.json',{'status':'running','completed_pairs':len(results),'expected_pairs':len(PAIRS),'current':record,'results':results})
            code=process.wait()
        record.update(exit_code=code,finished_unix=time.time())
        if code:
            dump(root/'evaluation_queue_progress.json',{'status':'failed','completed_pairs':len(results),'expected_pairs':len(PAIRS),'current':record,'results':results})
            raise RuntimeError(f'Length evaluation failed with exit {code}: d{ds} L{length}; outputs and log retained')
        folder=root/f'd{ds}/L{length}'
        observation=json.loads((folder/'observation_eval_summary.json').read_text())
        autonomous=json.loads((folder/'autonomous_summary.json').read_text())
        assert len(observation['results'])==156 and len(autonomous['results'])==1404
        record['observation_conditions']=156;record['autonomous_conditions']=1404;results.append(record)
        print('length evaluation queue complete',ds,length,len(results),'/10',flush=True)
    dump(root/'evaluation_queue_progress.json',{'status':'completed','completed_pairs':len(results),'expected_pairs':len(PAIRS),'results':results,'seconds':time.time()-started})


if __name__=='__main__':main()
