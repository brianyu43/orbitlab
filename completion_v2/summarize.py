"""Compact seed-level aggregation with explicit completion counts."""
from pathlib import Path
import json,time
import numpy as np
BASE=Path(__file__).resolve().parent

def mean(values):return float(np.mean(values)) if values else None

def main():
    result={'updated_unix':time.time(),'scope':'Live aggregation; incomplete branches are not final results.'}
    prows=[json.loads(p.read_text()) for p in (BASE/'perception/evaluation').glob('d*/*/*/summary.json')]
    psummary={}
    for kind in ['global','crop']:
        for split in ['test','ood','count3','count4','occlusion']:
            selected=[v for v in prows if v['kind']==kind and v['split']==split]
            if not selected:continue
            psummary[f'{kind}/{split}']={'units':len(selected),'strict_edit_success':mean([v['metrics']['learned']['end_to_end_success'] for v in selected]),
                                        'foreground_mae':mean([v['metrics']['learned']['image_foreground_mae'] for v in selected]),
                                        'per_data_seed':{str(seed):mean([v['metrics']['learned']['end_to_end_success'] for v in selected if v['data_seed']==seed]) for seed in [881201,881202,881203]}}
    result['perception']={'completed_units':len(prows),'expected_units':90,'verified_units':len(list((BASE/'perception/evaluation').glob('d*/*/*/verification.json'))),
                          'summary':psummary,'independence':'Three model initializations share one original training dataset. Three new evaluation datasets, not three training-data replications.'}
    detail=[json.loads(p.read_text()) for p in (BASE/'perception_detail/evaluation').glob('d*/*/*/summary.json')]
    result['perception_detail']={'completed_units':len(detail),'expected_units':90,
        'verified_units':len(list((BASE/'perception_detail/evaluation').glob('d*/*/*/verification.json'))),
        'summary':{f'{kind}/{split}':{'units':len(vv),'strict_edit_success':mean([v['metrics']['learned']['end_to_end_success'] for v in vv])}
                   for kind in ['binary','alpha'] for split in ['test','ood','count3','count4','occlusion']
                   if (vv:=[v for v in detail if v['kind']==kind and v['split']==split])}}
    drows=[json.loads(p.read_text()) for p in (BASE/'dynamics/evaluation').glob('d*/*/*/*/summary.json') if '/d640031_' not in str(p)]
    dsummary={}
    for inp in ['oracle','rgb_measurement']:
        for variant in ['one','one_bounded','multi','multi_projected','multi_bounded','force_wall']:
            for split in ['test','attribute_ood','count3','count4','force_ood']:
                selected=[v for v in drows if v['input']==inp and v['variant']==variant and v['split']==split and v['mode']=='variable_force']
                if not selected:continue
                dsummary[f'{inp}/{variant}/{split}']={'units':len(selected),**{
                    f'h{h}_{key}':mean([v['summary'][str(h)][key]['mean'] for v in selected if v['summary'][str(h)][key]['mean'] is not None])
                    for h in [1,16,61] for key in ['position_mae','velocity_mae','nonfinite','finite_blowup','missing_objects']}}
    result['dynamics']={'completed_units':len(drows),'expected_units':1404,'summary_variable_force':dsummary,
                       'verified_units_including_development':len(list((BASE/'dynamics/evaluation').glob('d*/*/*/*/verification.json')))}
    grows=[json.loads(p.read_text()) for p in (BASE/'generation/evaluation').glob('d*/*/result.json')]
    result['generation']={'completed_evaluations':len(grows),'expected_evaluations':22,
                          'completed_training_paths':len(list((BASE/'generation/runs').glob('d*/training_summary.json'))),'expected_training_paths':10,
                          'results':[{k:v[k] for k in ['seed','init','variant','metrics']} for v in grows]}
    srows=[json.loads(p.read_text()) for p in (BASE/'svib/runs').glob('*/evaluation.json')]
    result['svib']={'completed_models':len(srows),'expected_models':6,'results':srows}
    path=BASE/'live_summary.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    print(json.dumps({k:{key:value for key,value in v.items() if key.startswith(('completed','expected','verified'))} for k,v in result.items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':main()
