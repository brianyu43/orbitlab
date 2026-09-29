"""Summarize all seeds, retaining numerical failures and scene-level sampling."""
import json
import numpy as np
from common import HERE,dump,r
from dynamics_autonomous import HORIZONS
from dynamics_autonomous_study import NAME


def main():
    root=HERE/f'reports/{NAME}';verification=json.loads((root/'verification.json').read_text())
    assert verification['all_passed'] and verification['summary_sha256']==r.sha(root/'summary.json')
    c=json.loads((HERE/f'configs/{NAME}.json').read_text());results=[];ci=[]
    rng=np.random.default_rng(997031)
    for method in c['methods']:
        for group in c['groups']:
            mode,split=group['mode'],group['split']
            for kind in c['predictors']:
                seed_metrics=[];arrays=[]
                for seed in c['seeds']:
                    folder=HERE/f'runs/{NAME}/{method}_s{seed}/{mode}/{split}/{kind}'
                    m=json.loads((folder/'metrics.json').read_text());seed_metrics.append(m)
                    assert m['files']['trajectories.npz']==r.sha(folder/'trajectories.npz')
                    with np.load(folder/'trajectories.npz') as a:
                        arrays.append({key:a[key].copy() for key in ['metric_keys','metric_values']+
                                       (['response_metric_keys','response_metric_values'] if 'response_metric_values' in a else [])})
                for category in ['factual','counterfactual','response']:
                    if seed_metrics[0][category] is None:continue
                    for ih,h in enumerate(HORIZONS):
                        metrics={}
                        for key in seed_metrics[0][category][ih]['metrics']:
                            vals=[m[category][ih]['metrics'][key] for m in seed_metrics]
                            present=[v['mean'] for v in vals if v['mean'] is not None]
                            counts=[v['finite_episodes'] for v in vals]
                            weighted=sum(v['mean']*v['finite_episodes'] for v in vals if v['mean'] is not None)
                            metrics[key]={'seed_means':[v['mean'] for v in vals],
                                'mean_of_seed_means':float(np.mean(present)) if present else None,
                                'pooled_finite_mean':float(weighted/sum(counts)) if sum(counts) else None,
                                'finite_counts_by_seed':counts,'episodes_per_seed':group['episodes']}
                        results.append({'method':method,'mode':mode,'split':split,'predictor':kind,
                                        'category':category,'horizon':h,'metrics':metrics})
                for category,keys_name,values_name,primary in [
                    ('factual','metric_keys','metric_values',['finite_fraction','position_mae_pixels_finite','velocity_mae_pixels_per_frame_finite']),
                    ('response','response_metric_keys','response_metric_values',['both_branches_finite','target_position_response_mae_finite','nontarget_position_response_mae_finite'])]:
                    if keys_name not in arrays[0]:continue
                    keys=list(arrays[0][keys_name]);values=np.stack([v[values_name] for v in arrays])
                    draws=rng.integers(0,values.shape[1],size=(512,values.shape[1]))
                    for ih,h in enumerate(HORIZONS):
                        for key in primary:
                            v=values[:,:,ih,keys.index(key)]
                            sums=np.nansum(v,axis=0);counts=np.isfinite(v).sum(axis=0)
                            sampled_sum=sums[draws].sum(axis=1);sampled_count=counts[draws].sum(axis=1)
                            means=np.divide(sampled_sum,sampled_count,out=np.full(512,np.nan),where=sampled_count>0)
                            valid=np.isfinite(means)
                            center=float(sums.sum()/counts.sum()) if counts.sum() else None
                            interval=np.quantile(means[valid],[.025,.975]).tolist() if valid.any() else [None,None]
                            ci.append({'method':method,'mode':mode,'split':split,'predictor':kind,'category':category,
                                       'horizon':h,'metric':key,'pooled_finite_mean':center,'scene_bootstrap_95':interval,
                                       'finite_seed_episode_pairs':int(counts.sum()),'total_seed_episode_pairs':int(v.size),
                                       'finite_bootstrap_replicates':int(valid.sum())})
    rows=[]
    for item in results:
        for key,value in item['metrics'].items():
            rows.append({**{k:v for k,v in item.items() if k!='metrics'},'metric':key,
                         'mean_of_seed_means':value['mean_of_seed_means'],'pooled_finite_mean':value['pooled_finite_mean'],
                         **{f'seed_{i}_mean':v for i,v in enumerate(value['seed_means'])},
                         **{f'seed_{i}_finite_episodes':v for i,v in enumerate(value['finite_counts_by_seed'])}})
    r.write_csv(root/'all_metrics.csv',rows)
    dump(root/'aggregate.json',{'results':results,'scene_intervals':ci,'bootstrap_replicates':512,'bootstrap_seed':997031,
        'scope':'Fixed three initializations; resample original episodes jointly across seeds within each condition. Finite-only estimates always retain denominators. These are within-dataset intervals, not independent data-seed evidence or significance claims.',
        'summary_sha256':r.sha(root/'summary.json'),'verification_sha256':r.sha(root/'verification.json'),
        'aggregator_sha256':r.sha(__file__),'csv_sha256':r.sha(root/'all_metrics.csv')})
    print('aggregate groups',len(results),'metric rows',len(rows),'scene intervals',len(ci),flush=True)


if __name__=='__main__':main()
