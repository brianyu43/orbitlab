"""Posthoc tolerance sensitivity, without replacing frozen primary metrics."""
import json
from common import HERE, dump, np, torch
from object_tasks import load
from object_oracle_study import factor_checks


def main():
    rows=[]
    for kind in ['flat','object']:
        for seed in range(3):
            root=HERE/f'runs/object_oracle_study_v1/{kind}_s{seed}'
            pred=torch.load(root/'predictions.pt',map_location='cpu',weights_only=True)
            for split in ['test','count3','count4']:
                data,meta=load(split);p=pred[split];y=data['target_states'];present=y[:,:,15]>.5
                raw=json.loads((root/'metrics.json').read_text())['splits'][split]['all']
                for tol in [.5,1.,2.,4.]:
                    checks=factor_checks(p,y)
                    checks[:,:,2:4]=(p[:,:,10:12]-y[:,:,10:12]).abs()*31.5<=tol
                    checks[:,:,4]=(p[:,:,12]-y[:,:,12]).abs()*8<=tol/2
                    success=((checks.all(-1)|~present).all(-1)&((p[:,:,15]>=.5)==present).all(-1)).numpy()
                    ids=[v['source_index'] for v in meta]
                    means=[np.mean([success[i] for i,j in enumerate(ids) if j==s]) for s in sorted(set(ids))]
                    rows.append({'kind':kind,'seed':seed,'split':split,'xy_tolerance_pixels':tol,'radius_tolerance_pixels':tol/2,
                        'full_scene_success':float(np.mean(means)),'target_position_error_pixels':raw['target_position_error_pixels']['mean'],
                        'non_target_position_drift_pixels':raw['non_target_position_drift_pixels']['mean']})
    summary=[]
    for kind in ['flat','object']:
        for split in ['test','count3','count4']:
            rr=[v for v in rows if v['kind']==kind and v['split']==split]
            summary.append({'kind':kind,'split':split,'success_by_xy_tolerance':{str(tol):float(np.mean([v['full_scene_success'] for v in rr if v['xy_tolerance_pixels']==tol])) for tol in [.5,1.,2.,4.]},
                'target_position_error_pixels':float(np.mean([v['target_position_error_pixels'] for v in rr])),
                'non_target_position_drift_pixels':float(np.mean([v['non_target_position_drift_pixels'] for v in rr]))})
    path=HERE/'reports/object_oracle_study_v1/tolerance_sensitivity.json'
    result={'rows':rows,'summary':summary,'role':'Posthoc sensitivity diagnostic, not a replacement for the frozen primary 1px/.5px thresholds.'}
    if path.exists():assert json.loads(path.read_text())==result
    else:dump(path,result)
    print('Sensitivity reproduced:',len(rows),'condition/seed/threshold rows')


if __name__=='__main__':main()
