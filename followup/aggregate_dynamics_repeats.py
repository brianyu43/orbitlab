"""All C03-C06 metrics across three data units, retaining crossed initializations."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from common import HERE,dump,r
from dynamics_repeat_data import NAME

DATA=[640031,997101,997102]
IDENTITY=['stage','method','mode','split','predictor','information','category','horizon','metric']


def finite(value):return value is not None and math.isfinite(float(value))


def stats(values):
    v=[float(x) for x in values if finite(x)]
    return {'mean':statistics.mean(v) if v else None,'sd':statistics.stdev(v) if len(v)>1 else None,
            'min':min(v) if v else None,'max':max(v) if v else None,'available':len(v)}


def gate(path,evidence):
    d=json.loads(path.read_text());assert d['all_passed'],path
    evidence[str(path.relative_to(HERE))]=r.sha(path)
    return d


def collect(stage):
    records={};evidence={};sources={}
    def source(path):
        sources[str(path.relative_to(HERE))]=r.sha(path)
        return json.loads(path.read_text())
    def add(ds,seed,meta,metric,value,count=None,total=None):
        ident=tuple(meta.get(k,'') if k!='metric' else metric for k in IDENTITY)
        key=ident,ds,seed;assert key not in records,key
        assert value is None or finite(value),(key,value)
        records[key]={'value':value,'finite_episodes':count,'total_episodes':total}
    if stage=='state':
        gate(HERE/'reports/dynamics_state_study_v1/verification.json',evidence)
        checked=gate(HERE/f'reports/{NAME}/state_verification.json',evidence)
        assert checked['summary_sha256']==r.sha(HERE/f'reports/{NAME}/state_summary.json')
        for ds in DATA:
            root=HERE/'runs/dynamics_state_study_v1' if ds==640031 else HERE/f'runs/{NAME}/d{ds}/state'
            for mode in ['isotropic','fixed_gravity','variable_force']:
                for kind in ['flat','object','interaction','augmented','wrong_exact','joint_exact','relaxed']:
                    for seed in [0,1,2]:
                        metric=source(root/f'{mode}/{kind}_s{seed}/metrics.json')
                        for split,categories in metric['splits'].items():
                            for category,item in categories.items():
                                meta=dict(stage=stage,method='oracle',mode=mode,split=split,predictor=kind,category=category)
                                for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                                    add(ds,seed,meta,key,item[key]['mean'],item['episodes'],item['episodes'])
        gate(HERE/'reports/dynamics_state_baselines_v1/verification.json',evidence)
        b=gate(HERE/f'reports/{NAME}/state_baselines/verification.json',evidence)
        assert b['summary_sha256']==r.sha(HERE/f'reports/{NAME}/state_baselines/summary.json')
        for path,default_ds in [(HERE/'reports/dynamics_state_baselines_v1/summary.json',640031),
                                (HERE/f'reports/{NAME}/state_baselines/summary.json',None)]:
            summary=source(path)
            for item in summary['results']:
                ds=item.get('data_seed',default_ds);meta=dict(stage=stage,method='oracle',**{k:item[k] for k in ['mode','split','category']},predictor=item['kind'])
                for key,stat in item['metrics'].items():add(ds,-1,meta,key,stat['mean'],item['episodes'],item['episodes'])
    elif stage=='observation':
        for ds in DATA:
            root=HERE/'reports/dynamics_observation_eval_v1' if ds==640031 else HERE/f'reports/{NAME}/d{ds}'
            v=gate(root/('verification.json' if ds==640031 else 'observation_eval_verification.json'),evidence)
            path=root/('summary.json' if ds==640031 else 'observation_eval_summary.json')
            if ds!=640031:assert v['summary_sha256']==r.sha(path)
            summary=source(path);assert len(summary['results'])==156
            for item in summary['results']:
                folder=HERE/('runs/dynamics_observation_eval_v1' if ds==640031 else f'runs/{NAME}/d{ds}/observation_eval')
                p=folder/f"{item['method']}_s{item['seed']}/{item['mode']}/{item['split']}/metrics.json"
                assert item==source(p)
                seed=item['seed'];base=dict(stage=stage,**{k:item[k] for k in ['method','mode','split']})
                for component,categories in [('inference',item['inference']),*item['forecasts'].items()]:
                    if component=='inference':meta={**base,'information':'inference'}
                    else:
                        info,predictor=component.split('__');meta={**base,'information':info,'predictor':predictor}
                    categories_expected=['all','no_past_contact','past_contact','past_pair_contact','past_wall_contact'] if component=='inference' else ['all','no_past_contact','past_contact']
                    for category in categories_expected:
                        entry=categories.get(category,{'episodes':0,'metrics':{k:None for k in categories['all']['metrics']}})
                        for key,stat in entry['metrics'].items():
                            add(ds,seed,{**meta,'category':category},key,None if stat is None else stat['mean'],0 if entry['episodes']==0 else None,entry['episodes'])
    else:
        assert stage=='autonomous'
        for ds in DATA:
            root=HERE/'reports/dynamics_autonomous_v1' if ds==640031 else HERE/f'reports/{NAME}/d{ds}'
            v=gate(root/('verification.json' if ds==640031 else 'autonomous_verification.json'),evidence)
            path=root/('summary.json' if ds==640031 else 'autonomous_summary.json');assert v['summary_sha256']==r.sha(path)
            summary=source(path);assert len(summary['results'])==1404
            for item in summary['results']:
                folder=HERE/('runs/dynamics_autonomous_v1' if ds==640031 else f'runs/{NAME}/d{ds}/autonomous')
                p=folder/f"{item['method']}_s{item['seed']}/{item['mode']}/{item['split']}/{item['predictor']}/metrics.json"
                assert item==source(p)
                base=dict(stage=stage,**{k:item[k] for k in ['method','mode','split','predictor']})
                for category in ['factual','counterfactual','response']:
                    if item[category] is None:continue
                    for horizon in item[category]:
                        for key,stat in horizon['metrics'].items():
                            add(ds,item['seed'],{**base,'category':category,'horizon':horizon['horizon']},key,stat['mean'],
                                stat['finite_episodes'],stat['total_episodes'])
    return records,evidence,sources


def aggregate(stage):
    records,evidence,sources=collect(stage);root=HERE/f'reports/{NAME}/across_data';root.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(dict)
    for (identity,ds,seed),value in records.items():grouped[identity][ds,seed]=value
    rows=[];inputs=[];pairs=[]
    for identity,values in sorted(grouped.items()):
        meta=dict(zip(IDENTITY,identity));row=meta.copy();data_means=[]
        assert {ds for ds,_ in values}==set(DATA),(identity,'missing data unit')
        for ds in DATA:
            seeds=sorted(s for d,s in values if d==ds);assert seeds in [[-1],[0,1,2]]
            current=stats([values[ds,s]['value'] for s in seeds]);data_means.append(current['mean'])
            for key in ['mean','sd','available']:row[f'data_{ds}_{key}']=current[key]
            for seed in seeds:
                inputs.append({**meta,'data_seed':ds,'initialization':seed,**values[ds,seed]})
        summary=stats(data_means)
        for key,value in summary.items():row['across_data_'+key]=value
        rows.append(row)
        if meta['predictor']=='joint_exact':
            for comparator in ['flat','object','interaction','augmented','wrong_exact','relaxed','inertial','force_wall']:
                other=tuple(comparator if k=='predictor' else v for k,v in zip(IDENTITY,identity))
                if other not in grouped:continue
                compared=grouped[other];paired={**meta,'comparison':f'joint_exact minus {comparator}'};means=[]
                for ds in DATA:
                    differences=[]
                    for seed in [0,1,2]:
                        a=values[ds,seed]['value'];b=compared.get((ds,seed),compared.get((ds,-1),{})).get('value')
                        if finite(a) and finite(b):differences.append(a-b)
                    st=stats(differences);paired[f'data_{ds}_paired_mean']=st['mean'];paired[f'data_{ds}_pairs_available']=st['available'];means.append(st['mean'])
                for k,v in stats(means).items():paired['across_data_'+k]=v
                paired['data_units_with_negative_difference']=sum(finite(x) and x<0 for x in means)
                pairs.append(paired)
    for name,values in [('inputs',inputs),('metrics',rows),('paired',pairs)]:r.write_csv(root/f'{stage}_{name}.csv',values)
    dump(root/f'{stage}_manifest.json',{'stage':stage,'data_seeds':DATA,'initialization_seeds':[0,1,2],'independent_data_units':3,
        'input_scalar_records':len(inputs),'metric_combinations':len(rows),'paired_combinations':len(pairs),
        'source_sha256':sources,'verification_sha256':evidence,'script_sha256':r.sha(__file__),
        'files':{f'{stage}_{n}.csv':r.sha(root/f'{stage}_{n}.csv') for n in ['inputs','metrics','paired']},
        'scope':'Equal weight to each data-generation seed after averaging fixed initializations. Data-unit SD and range are descriptive, not nine iid runs or a significance test. Initialization -1 denotes a deterministic C03 reference; C04 analytic inference repeated across seeds remains one deterministic estimator per dataset. Finite-conditioned quantities retain input denominators; no nonfinite prediction is silently converted to a success. Empty contact strata stay missing with total=0. C04 matched-only metric denominators are unavailable in source summaries and remain blank; category totals are not substituted.'})
    print('aggregated repeats',stage,len(inputs),'inputs',len(rows),'metrics',len(pairs),'paired',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['state','observation','autonomous','all'],default='all');args=p.parse_args()
    for stage in ['state','observation','autonomous'] if args.stage=='all' else [args.stage]:aggregate(stage)


if __name__=='__main__':main()
