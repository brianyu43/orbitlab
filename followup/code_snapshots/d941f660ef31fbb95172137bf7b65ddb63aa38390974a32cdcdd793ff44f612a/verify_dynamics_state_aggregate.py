"""Check cross-initialization study aggregates directly against episode CSVs."""
import csv
import json
import numpy as np
from common import HERE,dump,r


def main():
    root=HERE/'reports/dynamics_state_study_v1';agg=json.loads((root/'aggregate.json').read_text())
    assert json.loads((root/'verification.json').read_text())['all_passed']
    assert agg['source_summary_sha256']==r.sha(root/'summary.json')
    assert agg['script_sha256']==r.sha(HERE/'aggregate_dynamics_state_study.py')
    cache={};checked=0
    for group in agg['groups']:
        mode,kind,split,category=(group[k] for k in ['mode','kind','split','category'])
        values=[]
        for seed in range(3):
            key=(mode,kind,seed)
            if key not in cache:cache[key]=list(csv.DictReader((HERE/f'runs/dynamics_state_study_v1/{mode}/{kind}_s{seed}/rows.csv').open()))
            rows=sorted([v for v in cache[key] if v['split']==split and v['category']==category],key=lambda v:int(v['episode']))
            values.append(rows)
        ids=[[v['episode'] for v in rows] for rows in values];assert ids[0]==ids[1]==ids[2];assert len(ids[0])==group['episodes']
        for metric,stat in group['metrics'].items():
            a=np.array([[float(v[metric]) for v in rows] for rows in values]);seeds=a.mean(1)
            np.testing.assert_allclose(seeds,stat['values_by_seed'],atol=1e-12,rtol=0)
            assert abs(seeds.mean()-stat['mean'])<1e-12 and abs(seeds.std(ddof=1)-stat['initialization_sample_sd'])<1e-12
            scene=a.mean(0);rng=np.random.default_rng(9123);samples=scene[rng.integers(len(scene),size=(1000,len(scene)))].mean(1)
            np.testing.assert_allclose(np.quantile(samples,[.025,.975]),stat['scene_ci95_conditional_on_three_models'],atol=1e-12,rtol=0);checked+=1
    for pair in agg['paired_comparisons']:
        other=pair['comparison'].split(' minus ')[1]
        def means(kind):
            v=next(g for g in agg['groups'] if g['mode']==pair['mode'] and g['split']==pair['split'] and g['category']=='all' and g['kind']==kind)
            return np.array(v['metrics']['velocity_mae_pixels_per_frame']['values_by_seed'])
        diff=means('joint_exact')-means(other)
        np.testing.assert_allclose(diff,pair['differences_by_seed'],atol=1e-12,rtol=0)
        assert abs(diff.mean()-pair['mean_difference'])<1e-12 and bool((diff<0).all())==pair['joint_lower_in_all_three']
    comparison=list(csv.DictReader((root/'comparison_rows.csv').open()));assert len(comparison)==len(agg['groups'])
    for row,g in zip(comparison,agg['groups']):
        for key in ['mode','kind','split','category']:assert row[key]==g[key]
        for field,metric,key in [('velocity_mae','velocity_mae_pixels_per_frame','mean'),('initialization_sd','velocity_mae_pixels_per_frame','initialization_sample_sd'),('position_mae','position_mae_pixels','mean')]:
            assert abs(float(row[field])-g['metrics'][metric][key])<1e-12
    dump(root/'aggregate_verification.json',{'all_passed':True,'model_episode_csv_files':len(cache),'groups_checked':len(agg['groups']),
        'metric_mean_initialization_sd_and_scene_interval_checks':checked,'paired_comparisons_checked':len(agg['paired_comparisons']),
        'aggregate_sha256':r.sha(root/'aggregate.json'),'comparison_rows_sha256':r.sha(root/'comparison_rows.csv'),'verifier_sha256':r.sha(__file__)})
    print('Cross-seed aggregates verified:',checked,'metrics and',len(agg['paired_comparisons']),'paired comparisons')


if __name__=='__main__':main()
