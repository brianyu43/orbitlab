"""Bounded replay queue following the already-running length evaluation queue."""
import json
import subprocess
import sys
import time
from common import HERE,dump,r

PAIRS=[(640031,4),(640031,8)]+[(ds,length) for ds in [997101,997102] for length in [1,2,4,8]]


def main():
    root=HERE/'reports/dynamics_observation_length_v1';results=[];started=time.time()
    for ds,length in PAIRS:
        for stage,script,summary_name,verification_name,expected in [
            ('C04','verify_observation_length_eval.py','observation_eval_summary.json','observation_eval_verification.json',156),
            ('C05_C06','verify_observation_length_autonomous.py','autonomous_summary.json','autonomous_verification.json',1404)]:
            folder=root/f'd{ds}/L{length}';summary=folder/summary_name;receipt=folder/verification_name
            record={'data_seed':ds,'length':length,'stage':stage,'started_unix':time.time(),'verifier_sha256':r.sha(HERE/script)}
            if not summary.exists():
                dump(root/'verification_queue_progress.json',{'status':'waiting_for_evaluation','completed_stages':len(results),'expected_stages':20,'current':record,'results':results})
                print('Length verification queue stopped before an unfinished evaluation:',ds,length,stage,flush=True)
                return
            if receipt.exists():
                report=json.loads(receipt.read_text());assert report['all_passed'] and report['summary_sha256']==r.sha(summary)
                assert report['verifier_sha256']==record['verifier_sha256'];record['reused_complete_receipt']=True
            else:
                log=HERE/f'logs/observation_length_{stage}_verification_d{ds}_L{length}_v1.log'
                command=[sys.executable,str(HERE/script),'--data-seed',str(ds),'--length',str(length)]
                record.update(log=str(log.relative_to(HERE)),command=command)
                print('length verification queue start',ds,length,stage,flush=True)
                with log.open('x') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT);record['pid']=process.pid
                    dump(root/'verification_queue_progress.json',{'status':'running','completed_stages':len(results),'expected_stages':20,'current':record,'results':results})
                    code=process.wait()
                record['exit_code']=code
                if code:
                    dump(root/'verification_queue_progress.json',{'status':'failed','completed_stages':len(results),'expected_stages':20,'current':record,'results':results})
                    raise RuntimeError(f'Length verification failed: d{ds} L{length} {stage}, exit {code}; evidence retained')
                report=json.loads(receipt.read_text())
            assert report['all_passed'] and report['conditions_verified']==expected
            record.update(finished_unix=time.time(),conditions_verified=expected,receipt_sha256=r.sha(receipt));results.append(record)
            print('length verification queue complete',ds,length,stage,len(results),'/20',flush=True)
    dump(root/'verification_queue_progress.json',{'status':'completed','completed_stages':20,'expected_stages':20,'results':results,'seconds':time.time()-started})


if __name__=='__main__':main()
