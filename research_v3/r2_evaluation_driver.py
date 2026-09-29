"""Evaluate completed frozen arms while the authorized training driver runs."""
import fcntl,json,os,subprocess,sys,time
import r2_core as c
import r2_world as w
from v3_common import HERE,ROOT,dump,sha,event,lock,status

def run(arm,repeat=False):
    cmd=[sys.executable,str(HERE/'r2_evaluate.py'),'--arm',arm]
    if repeat:cmd.append('--repeat-current')
    path=w.BASE/'execution'/f'{arm}_{"repeat" if repeat else "evaluation"}.log'
    with path.open('a') as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        dump(w.BASE/'live_evaluation_process.json',{'driver_pid':os.getpid(),'worker_pid':proc.pid,'arm':arm,'repeat_current':repeat,'started_unix':time.time()})
        rc=proc.wait()
    if rc:raise RuntimeError(f'Evaluation failed ({rc}): {path}')

def select():
    results={};sources={}
    for arm in c.ARMS:
        cells=[]
        for count in [2,3,4,6]:
            path=w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{arm}/n{count}/summary.json';r=json.loads(path.read_text());sources[str(path.relative_to(w.BASE))]=sha(path)
            cells.append(r['metrics']['occlusion_0.5'])
        n6=json.loads((w.BASE/f'evaluation/d{w.DEVELOPMENT}_s0/{arm}/n6/summary.json').read_text())['metrics']['all']
        results[arm]={'half_occluded_joint_edit':sum(v['joint_edit_success'] for v in cells)/4,
            'half_occluded_non_target_correct':sum(v['non_target_correct'] for v in cells)/4,'count6_missing_fraction':n6['missing_fraction']}
    candidates=[]
    for mask in [0,1]:
        candidate=f'recurrent_mask{mask}';baseline=max([f'detector_mask{mask}',f'slot_mask{mask}'],key=lambda a:results[a]['half_occluded_joint_edit'])
        ca,ba=results[candidate],results[baseline]
        gain=ca['half_occluded_joint_edit']-ba['half_occluded_joint_edit'];drop=ba['half_occluded_non_target_correct']-ca['half_occluded_non_target_correct']
        candidates.append({'candidate':candidate,'baseline':baseline,'edit_gain':gain,'non_target_drop':drop,'count6_missing':ca['count6_missing_fraction'],'mask_supervision':bool(mask)})
    chosen=max(candidates,key=lambda r:(r['edit_gain'],-r['mask_supervision']))
    passed=chosen['edit_gain']>=.15 and chosen['non_target_drop']<=.02 and chosen['count6_missing']<=.05
    lock(w.BASE/'selection.json',{'chosen':chosen,'development_gate_passed':passed,'all_comparisons':candidates,'all_arms':results,'sources':sources,
        'next':'confirmation_required_plus_decoder_and_verification' if passed else 'failure_diagnosis_plus_decoder_and_verification_without_automatic_confirmation_expansion'})

def main():
    with (w.BASE/'evaluation_driver.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=json.loads((w.BASE/'evaluation_protocol.json').read_text());assert sha(HERE/'r2_evaluate.py')==frozen['evaluator_sha256']
        for arm in c.ARMS:
            path=w.BASE/f'runs/d{w.DEVELOPMENT}_s0/{arm}/run.json'
            while not path.exists():
                live=json.loads((w.BASE/'live_process.json').read_text())
                try:os.kill(live['driver_pid'],0)
                except ProcessLookupError:raise RuntimeError('Training driver terminated before this arm completed; do not silently restart')
                except PermissionError:pass
                time.sleep(10)
            run(arm)
            if arm.startswith('recurrent'):run(arm,True)
        select();dump(w.BASE/'evaluation_complete.json',{'ended_unix':time.time(),'conditions':32,'calibration_sets':8,'decoder':'pending','verification':'pending'})
        status('R2','development_evaluated_decoder_and_verification_pending')
        print('R2 development evaluation complete; decoder and verification remain pending',flush=True)

if __name__=='__main__':main()
