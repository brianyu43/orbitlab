"""Independent raw-row and unique-scene audit of every preview aggregate."""
import csv
import json
import numpy as np
from common import HERE, dump, r
from svib_preview_data import NAME, ALPHAS, alpha_name


def numbers(folder, metrics):
    rows = list(csv.DictReader((folder/'rows.csv').open()))
    assert [int(x['episode']) for x in rows] == list(range(len(rows)))
    values = np.array([[np.nan if x[k] == '' else (1. if x[k] == 'True' else 0.) if x[k] in ['True','False'] else float(x[k]) for k in metrics] for x in rows])
    change = np.array([x['has_change'] == 'True' for x in rows])
    return values, change


def audit_stat(stat, matrix, mask):
    sub = np.ma.masked_invalid(matrix[:, mask])
    assert np.array_equal(np.ma.getmaskarray(sub), np.broadcast_to(np.ma.getmaskarray(sub)[0], sub.shape))
    valid = ~np.ma.getmaskarray(sub)[0]
    assert stat['finite_scenes'] == int(valid.sum()) and stat['total_scenes'] == int(mask.sum())
    assert stat['initializations'] == len(matrix)
    if not valid.any():
        assert all(stat[k] is None for k in ['mean','scene_ci95','seed_sd','seed_min','seed_max'])
        assert stat['per_seed_mean'] == [None]*len(matrix)
        return
    per_seed = np.array(sub.mean(axis=1)); scenes = np.array(sub[:, valid].mean(axis=0))
    np.testing.assert_allclose(stat['mean'], float(sub.mean()), atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(stat['per_seed_mean'], per_seed, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose([stat['seed_min'],stat['seed_max']], [per_seed.min(),per_seed.max()], atol=1e-10, rtol=1e-12)
    if len(matrix)>1: np.testing.assert_allclose(stat['seed_sd'], per_seed.std(ddof=1), atol=1e-10, rtol=1e-12)
    else: assert stat['seed_sd'] is None
    random = np.random.default_rng(9123)
    resampled = np.array([scenes[index].mean() for index in random.integers(len(scenes),size=(1000,len(scenes)))])
    np.testing.assert_allclose(stat['scene_ci95'],np.quantile(resampled,[.025,.975]),atol=1e-10,rtol=1e-12)


def main():
    root=HERE/f'reports/{NAME}/aggregate_v1';p=root/'summary.json';summary=json.loads(p.read_text())
    config_path=HERE/'configs/svib_preview_aggregate_v1.json';config=json.loads(config_path.read_text())
    assert summary['config_sha256']==r.sha(config_path)
    assert config['source_sha256']==r.sha(HERE/'aggregate_svib_preview.py')
    for source,sha in summary['verification_evidence'].items():assert r.sha(HERE/source)==sha
    methods=['identity','train_target_mean','plain','c4'];splits=['train','val','test_id','heldout'];categories=['all','changed','unchanged']
    metrics=['pixel_mse','image_squared_error_sum','identity_pixel_mse','foreground_pixel_mse','changed_pixel_mse','unchanged_pixel_mse','pixel_mse_minus_identity','lower_pixel_mse_than_identity']
    comparisons=[('c4','plain'),('plain','identity'),('c4','identity'),('plain','train_target_mean'),('c4','train_target_mean')]
    assert config['metrics']==metrics and config['methods']==methods and config['alphas']==ALPHAS
    assert len(summary['scene_blocks'])==64
    data={};sources={};total_rows=0
    for block in summary['scene_blocks']:
        path=HERE/block['path'];assert r.sha(path)==block['sha256'];m=json.loads(path.read_text());folder=path.parent
        a,k,s=m['alpha'],m['method'],m['split'];key=(a,k,s);assert key not in data
        assert m['metrics']==metrics and m['seeds']==([0,1,2] if k in ['plain','c4'] else [-1])
        for name,sha in m['files'].items():assert r.sha(folder/name)==sha
        for source,sha in m['sources'].items():assert r.sha(HERE/source)==sha;sources[source]=sha
        raw=[];changed=None
        for seed in m['seeds']:
            original=HERE/(f'reports/{NAME}/baselines/{alpha_name(a)}/{s}/{k}' if seed==-1 else f'runs/{NAME}/{alpha_name(a)}/{k}_s{seed}/{s}')
            v,c=numbers(original,metrics);raw.append(v)
            if changed is None:changed=c
            else:np.testing.assert_array_equal(c,changed)
            entry=json.loads((original/'metrics.json').read_text())
            assert entry['input_sha256']==m['input_sha256'] and entry['target_sha256']==m['target_sha256']
            total_rows+=len(v)
        raw=np.stack(raw)
        with np.load(folder/'scenes.npz') as z:
            assert set(z.files)=={'values','episode','has_change'}
            np.testing.assert_array_equal(z['values'],raw);np.testing.assert_array_equal(z['has_change'],changed)
            np.testing.assert_array_equal(z['episode'],np.arange(len(changed)))
        if s=='heldout':assert len(changed)==100 and changed.sum()==74
        data[key]=(raw,changed,m['input_sha256'],m['target_sha256'])
    assert set(data)=={(a,k,s) for a in ALPHAS for k in methods for s in splits}
    unique=set();paired_unique=set()
    for stat in summary['conditions']:
        a,k,s,category,metric=[stat[x] for x in ['alpha','method','split','category','metric']]
        key=(a,k,s,category,metric);assert key not in unique;unique.add(key)
        matrix,change,*_=data[(a,k,s)];mask=np.ones(len(change),bool) if category=='all' else change if category=='changed' else ~change
        audit_stat(stat,matrix[...,metrics.index(metric)],mask)
    for stat in summary['paired_conditions']:
        a,left,right,s,category,metric=[stat[x] for x in ['alpha','left','right','split','category','metric']]
        key=(a,left,right,s,category,metric);assert key not in paired_unique;paired_unique.add(key)
        lhs,change,inp,tgt=data[(a,left,s)];rhs,c,ri,rt=data[(a,right,s)]
        assert inp==ri and tgt==rt;np.testing.assert_array_equal(change,c)
        mask=np.ones(len(change),bool) if category=='all' else change if category=='changed' else ~change
        j=metrics.index(metric);audit_stat(stat,lhs[...,j]-rhs[...,j],mask)
    assert unique=={(a,k,s,c,m) for a in ALPHAS for k in methods for s in splits for c in categories for m in metrics}
    assert paired_unique=={(a,l,r,s,c,m) for a in ALPHAS for l,r in comparisons for s in splits for c in categories for m in metrics}
    cost_keys=set()
    for row in summary['costs']:
        a,k,seed=row['alpha'],row['method'],row['seed'];assert (a,k,seed) not in cost_keys;cost_keys.add((a,k,seed))
        folder=HERE/f'runs/{NAME}/{alpha_name(a)}/{k}_s{seed}';run=json.loads((folder/'run.json').read_text());entry=json.loads((folder/'heldout/metrics.json').read_text())
        assert row['training_seconds']==run['training_seconds'] and row['cpu_forward_seconds_100_images']==entry['cpu_forward_seconds']
        assert row['parameters']==run['parameters']==1266467 and row['steps']==run['steps']==2000
        assert row['rotation_mean_absolute_gap']==entry['equivariance']['mean_absolute_gap']
        assert row['rotation_maximum_absolute_gap']==entry['equivariance']['maximum_absolute_gap']
    assert cost_keys=={(a,k,s) for a in ALPHAS for k in ['plain','c4'] for s in [0,1,2]}
    assert len(unique)==summary['condition_count']==1536 and len(paired_unique)==summary['paired_condition_count']==1920
    assert total_rows==6400
    dump(root/'verification.json',{'all_passed':True,'scene_blocks_verified':64,'raw_row_instances_checked':total_rows,
        'aggregate_conditions_checked':1536,'paired_conditions_checked':1920,'compute_records_checked':24,
        'initializations_not_pooled_as_independent_scenes':True,'shared_external_test_unique_scenes':100,
        'summary_sha256':r.sha(p),'verifier_sha256':r.sha(__file__),
        'scope':'Every preview aggregate and paired difference checked from saved per-scene rows. Conditional scene intervals do not establish population or independent-data significance.'})
    print('SVIB aggregate audit complete',len(unique),len(paired_unique),flush=True)


if __name__=='__main__':main()
