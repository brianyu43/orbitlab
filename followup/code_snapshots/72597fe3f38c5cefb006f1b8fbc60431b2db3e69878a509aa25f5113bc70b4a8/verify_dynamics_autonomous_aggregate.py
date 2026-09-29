"""Audit every cross-seed mean and independently regenerate scene intervals."""
import json
import numpy as np
from common import HERE,dump,r
from dynamics_autonomous_study import NAME
from dynamics_autonomous import HORIZONS


def main():
    root=HERE/f'reports/{NAME}';a=json.loads((root/'aggregate.json').read_text())
    c=json.loads((HERE/f'configs/{NAME}.json').read_text());s=json.loads((root/'summary.json').read_text())
    assert a['summary_sha256']==r.sha(root/'summary.json') and a['verification_sha256']==r.sha(root/'verification.json')
    lookup={(x['method'],x['seed'],x['mode'],x['split'],x['predictor']):x for x in s['results']}
    agg={(x['method'],x['mode'],x['split'],x['predictor'],x['category'],x['horizon']):x for x in a['results']}
    assert len(agg)==len(a['results'])
    expected={(m,g['mode'],g['split'],k,category,h) for m in c['methods'] for g in c['groups'] for k in c['predictors']
              for category in (['factual','counterfactual','response'] if g['counterfactual_episodes'] else ['factual']) for h in HORIZONS}
    assert set(agg)==expected and a['csv_sha256']==r.sha(root/'all_metrics.csv')
    checks=0
    for row in a['results']:
        condition=[lookup[row['method'],seed,row['mode'],row['split'],row['predictor']] for seed in c['seeds']]
        ih=HORIZONS.index(row['horizon'])
        for key,value in row['metrics'].items():
            v=[x[row['category']][ih]['metrics'][key] for x in condition]
            assert value['seed_means']==[x['mean'] for x in v]
            assert value['finite_counts_by_seed']==[x['finite_episodes'] for x in v]
            good=[x for x in v if x['mean'] is not None]
            expected=sum(x['mean'] for x in good)/len(good) if good else None
            pooled=sum(x['mean']*x['finite_episodes'] for x in good)/sum(x['finite_episodes'] for x in good) if good else None
            if expected is None:assert value['mean_of_seed_means'] is None and value['pooled_finite_mean'] is None
            else:
                np.testing.assert_allclose(value['mean_of_seed_means'],expected,rtol=1e-12)
                np.testing.assert_allclose(value['pooled_finite_mean'],pooled,rtol=1e-12)
            checks+=1
    # Same PRNG and units, but direct selected values instead of stored per-scene sufficient statistics.
    rng=np.random.default_rng(a['bootstrap_seed']);ci_index=0
    for method in c['methods']:
        for g in c['groups']:
            for kind in c['predictors']:
                arrays=[]
                for seed in c['seeds']:
                    path=HERE/f"runs/{NAME}/{method}_s{seed}/{g['mode']}/{g['split']}/{kind}/trajectories.npz"
                    with np.load(path) as x:arrays.append({k:x[k].copy() for k in x.files if 'metric' in k})
                for category,keys_name,values_name,primary in [
                    ('factual','metric_keys','metric_values',['finite_fraction','position_mae_pixels_finite','velocity_mae_pixels_per_frame_finite']),
                    ('response','response_metric_keys','response_metric_values',['both_branches_finite','target_position_response_mae_finite','nontarget_position_response_mae_finite'])]:
                    if keys_name not in arrays[0]:continue
                    keys=list(arrays[0][keys_name]);values=np.stack([v[values_name] for v in arrays])
                    draws=rng.integers(0,g['episodes'],size=(512,g['episodes']))
                    for ih,h in enumerate(HORIZONS):
                        for key in primary:
                            row=a['scene_intervals'][ci_index];ci_index+=1
                            assert [row[x] for x in ['method','mode','split','predictor','category','horizon','metric']]==[method,g['mode'],g['split'],kind,category,h,key]
                            v=values[:,:,ih,keys.index(key)];means=[]
                            # Vectorized resampling keeps every initialization of each selected scene together.
                            for indices in draws:
                                selected=v[:,indices];finite=np.isfinite(selected)
                                if finite.any():means.append(float(selected[finite].mean()))
                            expected=np.quantile(means,[.025,.975]) if means else [np.nan,np.nan]
                            reported=[np.nan if z is None else z for z in row['scene_bootstrap_95']]
                            np.testing.assert_allclose(reported,expected,atol=1e-8,rtol=1e-10,equal_nan=True)
                            assert row['finite_seed_episode_pairs']==int(np.isfinite(v).sum()) and row['total_seed_episode_pairs']==v.size
    assert ci_index==len(a['scene_intervals'])
    # Check first autonomous step against the independently verified C04 artifacts.
    first_checks=0
    for item in s['results']:
        if item['first_step_C04_max_abs_error'] is None:continue
        method,seed,mode,split,kind=[item[k] for k in ['method','seed','mode','split','predictor']]
        folder=HERE/f'runs/{NAME}/{method}_s{seed}/{mode}/{split}'
        with np.load(folder/'initial_inputs.npz') as x:initial=x['initial'].copy();order=x['order'].copy()
        with np.load(folder/kind/'trajectories.npz') as x:motion=x['motion'][:,1].copy()
        actual=np.zeros((len(initial),6,4))
        for i in range(len(initial)):
            for j,color in enumerate(order[i]):
                if initial[i,j,6]>.5:actual[i,color]=motion[i,j]
        parent='rgb_cnn' if method=='oracle' else method;info='true_state_true_context' if method=='oracle' else 'estimated_state_estimated_context'
        with np.load(HERE/f'runs/dynamics_observation_eval_v1/{parent}_s{seed}/{mode}/{split}/predictions.npz') as x:expected=x[info+'__'+kind][...,:4]
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
        assert float(np.max(np.abs(actual-expected)))==item['first_step_C04_max_abs_error'];first_checks+=len(initial)
    dump(root/'aggregate_verification.json',{'all_passed':True,'cross_seed_metric_checks':checks,'scene_bootstrap_intervals_checked':ci_index,
        'first_steps_compared_to_C04':first_checks,'all_aggregate_combinations_present':len(agg),
        'aggregate_sha256':r.sha(root/'aggregate.json'),'verifier_sha256':r.sha(__file__),
        'scope':'All cross-seed means and scene bootstrap intervals; first-step continuity with C04. No independent data-seed claim.'})
    print('cross-seed metrics',checks,'intervals',ci_index,'C04 first steps',first_checks,flush=True)


if __name__=='__main__':main()
