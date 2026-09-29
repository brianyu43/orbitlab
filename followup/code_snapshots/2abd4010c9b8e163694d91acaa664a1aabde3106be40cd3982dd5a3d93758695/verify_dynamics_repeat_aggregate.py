"""Audit every exported scalar against source metrics and recompute data-unit stats."""
import argparse
import csv
import json
import math
from functools import lru_cache
import numpy as np
from common import HERE,dump,r
from dynamics_repeat_data import NAME

DATA=[640031,997101,997102]
IDENTITY=['stage','method','mode','split','predictor','information','category','horizon','metric']
EXPECTED={'state':(8970,1170),'observation':(280800,31200),'autonomous':(1199448,133272)}


@lru_cache(maxsize=128)
def read(path):return json.loads(path.read_text())


def number(value):return None if value=='' or value is None else float(value)


def check(actual,expected):
    actual=number(actual)
    if expected is None or not math.isfinite(expected):assert actual is None,(actual,expected)
    else:assert actual is not None and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12),(actual,expected)


def reference(row):
    stage=row['stage'];ds=int(row['data_seed']);seed=int(row['initialization']);mode=row['mode'];split=row['split']
    kind=row['predictor'];category=row['category'];metric=row['metric'];method=row['method']
    if stage=='state':
        if seed==-1:
            root='dynamics_state_baselines_v1' if ds==640031 else f'{NAME}/state_baselines'
            items=read(HERE/f'reports/{root}/summary.json')['results']
            item=next(x for x in items if (x.get('data_seed',640031),x['mode'],x['split'],x['kind'],x['category'])==(ds,mode,split,kind,category))
            return item['metrics'][metric]['mean'],item['episodes'],item['episodes']
        root='dynamics_state_study_v1' if ds==640031 else f'{NAME}/d{ds}/state'
        item=read(HERE/f'runs/{root}/{mode}/{kind}_s{seed}/metrics.json')['splits'][split][category]
        return item[metric]['mean'],item['episodes'],item['episodes']
    if stage=='observation':
        root='dynamics_observation_eval_v1' if ds==640031 else f'{NAME}/d{ds}/observation_eval'
        item=read(HERE/f'runs/{root}/{method}_s{seed}/{mode}/{split}/metrics.json')
        categories=item['inference'] if row['information']=='inference' else item['forecasts'][row['information']+'__'+kind]
        if category not in categories:return None,0,0
        item=categories[category];stat=item['metrics'][metric]
        return None if stat is None else stat['mean'],None,item['episodes']
    root='dynamics_autonomous_v1' if ds==640031 else f'{NAME}/d{ds}/autonomous'
    entries=read(HERE/f'runs/{root}/{method}_s{seed}/{mode}/{split}/{kind}/metrics.json')[category]
    item=next(x for x in entries if x['horizon']==int(row['horizon']))['metrics'][metric]
    return item['mean'],item['finite_episodes'],item['total_episodes']


def summarize(v):
    a=np.asarray(v,dtype=float);a=a[np.isfinite(a)]
    return dict(mean=float(a.mean()) if len(a) else None,sd=float(a.std(ddof=1)) if len(a)>1 else None,
                min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None,available=len(a))


def audit(stage):
    root=HERE/f'reports/{NAME}/across_data';manifest=read(root/f'{stage}_manifest.json')
    assert manifest['data_seeds']==DATA and manifest['independent_data_units']==3
    assert manifest['script_sha256']==r.sha(HERE/'aggregate_dynamics_repeats.py')
    for family in ['source_sha256','verification_sha256']:
        for path,sha in manifest[family].items():assert r.sha(HERE/path)==sha,path
    for file,sha in manifest['files'].items():assert r.sha(root/file)==sha,file
    groups={};count=0
    with (root/f'{stage}_inputs.csv').open() as f:
        for row in csv.DictReader(f):
            identity=tuple(row[k] for k in IDENTITY);assert row['stage']==stage
            ds=int(row['data_seed']);seed=int(row['initialization']);i=DATA.index(ds);j=0 if seed==-1 else seed
            assert seed in [-1,0,1,2]
            expected=reference(row)
            for key,value in zip(['value','finite_episodes','total_episodes'],expected):check(row[key],value)
            if identity not in groups:groups[identity]=(np.full((3,3),np.nan),set(),seed==-1)
            values,seen,deterministic=groups[identity];assert deterministic==(seed==-1) and (i,j) not in seen
            value=number(row['value']);values[i,j]=np.nan if value is None else value;seen.add((i,j));count+=1
    assert (count,len(groups))==EXPECTED[stage]
    assert manifest['input_scalar_records']==count and manifest['metric_combinations']==len(groups)
    seen_metrics=set();metric_checks=0
    with (root/f'{stage}_metrics.csv').open() as f:
        for row in csv.DictReader(f):
            identity=tuple(row[k] for k in IDENTITY);assert identity not in seen_metrics;seen_metrics.add(identity)
            values,seen,deterministic=groups[identity];assert len(seen)==(3 if deterministic else 9)
            means=[]
            for i,ds in enumerate(DATA):
                stats=summarize(values[i]);means.append(stats['mean'])
                for key in ['mean','sd','available']:check(row[f'data_{ds}_{key}'],stats[key]);metric_checks+=1
            for key,value in summarize(means).items():check(row['across_data_'+key],value);metric_checks+=1
    assert seen_metrics==set(groups)
    expected_pairs=set()
    for identity in groups:
        if identity[4]!='joint_exact':continue
        for comparator in ['flat','object','interaction','augmented','wrong_exact','relaxed','inertial','force_wall']:
            other=identity[:4]+(comparator,)+identity[5:]
            if other in groups:expected_pairs.add((identity,comparator))
    seen_pairs=set();pair_checks=0
    with (root/f'{stage}_paired.csv').open() as f:
        for row in csv.DictReader(f):
            identity=tuple(row[k] for k in IDENTITY);comparator=row['comparison'].removeprefix('joint_exact minus ')
            key=identity,comparator;assert key in expected_pairs and key not in seen_pairs;seen_pairs.add(key)
            other=identity[:4]+(comparator,)+identity[5:]
            a=groups[identity][0];b,_,deterministic=groups[other]
            diff=a-(b[:,:1] if deterministic else b);means=[]
            for i,ds in enumerate(DATA):
                stat=summarize(diff[i]);means.append(stat['mean'])
                check(row[f'data_{ds}_paired_mean'],stat['mean']);check(row[f'data_{ds}_pairs_available'],stat['available']);pair_checks+=2
            for key,value in summarize(means).items():check(row['across_data_'+key],value);pair_checks+=1
            check(row['data_units_with_negative_difference'],sum(v is not None and v<0 for v in means));pair_checks+=1
    assert seen_pairs==expected_pairs and len(seen_pairs)==manifest['paired_combinations']
    dump(root/f'{stage}_verification.json',{'all_passed':True,'stage':stage,'source_scalar_records_checked':count,
        'metric_combinations_checked':len(groups),'aggregate_stat_checks':metric_checks,'paired_combinations_checked':len(seen_pairs),
        'paired_stat_checks':pair_checks,'independent_data_units':3,'manifest_sha256':r.sha(root/f'{stage}_manifest.json'),
        'verifier_sha256':r.sha(__file__),'scope':'Every CSV input checked against original per-condition metrics; NumPy recomputation of all data-unit and paired statistics with exact combination coverage. No nine-iid interpretation.'})
    print('verified across-data aggregate',stage,count,'source scalars',len(groups),'metric groups',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['state','observation','autonomous','all'],default='all');args=p.parse_args()
    for stage in ['state','observation','autonomous'] if args.stage=='all' else [args.stage]:audit(stage)


if __name__=='__main__':main()
