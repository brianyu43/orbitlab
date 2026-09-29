"""Finish fixed decoder comparisons and replay every saved pixel prediction."""
import fcntl,json,os,subprocess,sys,time
import r2_decoder_core as c
import r2_decoder_evaluate as e
from v3_common import HERE,ROOT,sha,dump,event,status

def run(arm,count,verify=False,identity=False):
    name=arm+('_identity' if identity else '');path=c.BASE/f'{name}_n{count}_{"verify" if verify else "evaluate"}.log'
    cmd=[sys.executable,str(HERE/'r2_decoder_evaluate.py'),'--arm',arm,'--count',str(count)]
    if verify:cmd.append('--verify')
    if identity:cmd.append('--identity')
    with path.open('a') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(c.BASE/'live_evaluation_process.json',{'driver_pid':os.getpid(),'worker_pid':proc.pid,'arm':name,'count':count,'verify':verify,'started_unix':time.time()})
        rc=proc.wait()
    if rc:raise RuntimeError(f'Decoder evaluation exited{rc}: {path}')

def main():
    e.freeze()
    with (c.BASE/'evaluation_driver.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for count in [2,3,4,6]:run('copy_input',count);run('copy_input',count,True)
        while not (c.BASE/'training_complete.json').exists():
            live=json.loads((c.BASE/'live_process.json').read_text())
            try:os.kill(live['driver_pid'],0)
            except ProcessLookupError:raise RuntimeError('Decoder training driver terminated without completion record; inspect before recovery')
            except PermissionError:pass
            time.sleep(10)
        for arm in e.ARMS:
            for count in [2,3,4,6]:run(arm,count);run(arm,count,True)
        for count in [2,3,4,6]:run('truth_state',count,identity=True);run('truth_state',count,True,True)
        summaries=[];checks=[]
        for path in sorted((c.BASE/'evaluation_v2').glob('*/n*/summary.json')):
            verify=path.with_name('verification.json');check=json.loads(verify.read_text());assert check['summary_sha256']==sha(path)
            summaries.append(json.loads(path.read_text()));checks.append({'condition':str(path.parent.relative_to(c.BASE)),'summary_sha256':sha(path),'verification_sha256':sha(verify),**check})
        assert len(summaries)==36 and sum(v['all_rgb_predictions_replayed'] for v in checks)==51200
        dump(c.BASE/'aggregate.json',{'conditions':summaries,'development_only':True,'interpretation':'Pixel errors/gates and analytic-state-plus-pixel gates remain separate; no independent human or real-image validation.'})
        dump(c.BASE/'verification.json',{'conditions':checks,'verified_units':36,'expected_units':36,'all_rgb_predictions_replayed':51200,'independent_retraining':False})
        dump(c.BASE/'evaluation_complete.json',{'ended_unix':time.time(),'units':36,'pixel_replays':51200});event('R2_decoder_evaluation_and_verification_complete')
        status('R2','reader_and_decoder_verified_report_pending',reader='r2/reader_verification.json',decoder='r2/decoder/verification.json')
        print('R2 decoder36conditions verified',flush=True)

if __name__=='__main__':main()
