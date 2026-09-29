"""Crossed-seed context omission summaries and paired differences, with all strata."""
import json
import statistics
from collections import defaultdict
from common import HERE,dump,r
from dynamics_context_omission import NAME,DATA,KINDS,MASKS,freeze

FIELDS=['condition','kind','mode','split','category','metric']
METRICS=['position_mae_pixels','velocity_mae_pixels_per_frame']


def stats(values):
    v=[float(x) for x in values if x is not None]
    return {'mean':statistics.mean(v) if v else None,'sd':statistics.stdev(v) if len(v)>1 else None,
            'min':min(v) if v else None,'max':max(v) if v else None,'available':len(v)}


def summarize(values):
    out={};means=[]
    for ds in DATA:
        s=stats([values[ds,seed] for seed in [0,1,2]])
        out.update({f'data_{ds}_{k}':v for k,v in s.items()});means.append(s['mean'])
    out.update({f'across_data_{k}':v for k,v in stats(means).items()})
    return out


def main():
    freeze();root=HERE/f'reports/{NAME}';vp=root/'verification.json';verified=json.loads(vp.read_text())
    assert verified['all_passed'] and verified['conditions_verified']==108
    assert verified['summary_sha256']==r.sha(root/'summary.json')
    assert verified['verifier_sha256']==r.sha(HERE/'verify_context_omission.py')
    sources={};records={};inputs=[]
    for receipt in verified['results']:
        for path,sha in receipt['artifacts'].items():assert r.sha(HERE/path)==sha,path
    for ds in DATA:
        for condition in MASKS:
            for kind in KINDS:
                for seed in [0,1,2]:
                    folder=HERE/f'runs/{NAME}/d{ds}/{condition}/{kind}_s{seed}';path=folder/'metrics.json'
                    m=json.loads(path.read_text());sources[str(path.relative_to(HERE))]=r.sha(path)
                    for group,categories in m['groups'].items():
                        mode,split=group.split('__')
                        for category,entry in categories.items():
                            for metric in METRICS:
                                stat=entry[metric];value=None if stat is None else stat['mean']
                                key=(condition,kind,mode,split,category,metric);record=(key,ds,seed)
                                assert record not in records;records[record]=value
                                inputs.append({**dict(zip(FIELDS,key)),'data_seed':ds,'seed':seed,'value':value,'episodes':entry['episodes']})
    grouped=defaultdict(dict)
    for (key,ds,seed),value in records.items():grouped[key][ds,seed]=value
    summaries=[];pairs=[];pair_inputs=[]
    for key,values in sorted(grouped.items()):
        assert set(values)=={(ds,seed) for ds in DATA for seed in [0,1,2]}
        summaries.append({**dict(zip(FIELDS,key)),**summarize(values)})
    base_keys=sorted({key[2:] for key in grouped})
    contrasts=[('context',condition,'full',kind,kind) for condition in MASKS if condition!='full' for kind in KINDS]
    contrasts += [('model',condition,condition,left,right) for condition in MASKS for left,right in
                  [('wrong_exact','interaction'),('joint_exact','interaction'),('joint_exact','wrong_exact')]]
    for axis,lc,rc,lk,rk in contrasts:
        for tail in base_keys:
            meta={'contrast':axis,'left_condition':lc,'right_condition':rc,'left_kind':lk,'right_kind':rk,
                  **dict(zip(FIELDS[2:],tail))};values={}
            for ds in DATA:
                for seed in [0,1,2]:
                    left=records[((lc,lk,*tail),ds,seed)];right=records[((rc,rk,*tail),ds,seed)]
                    diff=None if left is None or right is None else left-right;values[ds,seed]=diff
                    pair_inputs.append({**meta,'data_seed':ds,'seed':seed,'left':left,'right':right,'difference':diff})
            pairs.append({**meta,**summarize(values)})
    out=root/'across_data';out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('inputs',inputs),('metrics',summaries),('paired_inputs',pair_inputs),('paired',pairs)]:r.write_csv(out/f'{name}.csv',rows)
    dump(out/'manifest.json',{'source_sha256':sources,'verification_sha256':r.sha(vp),'script_sha256':r.sha(__file__),
        'files':{f'{name}.csv':r.sha(out/f'{name}.csv') for name in ['inputs','metrics','paired_inputs','paired']},
        'input_scalars':len(inputs),'metric_groups':len(summaries),'paired_input_scalars':len(pair_inputs),'paired_groups':len(pairs),
        'data_seeds':DATA,'seeds':[0,1,2],'independent_data_units':3,
        'scope':'Within-data fixed-initialization average, then equal-weight means/SD/range over 3 independent data units. Matched data, initialization, episode/contact strata for all condition/model contrasts. Three initializations are not three new datasets. Empty strata remain missing. All models train on variable_force, including evaluation in other modes. No population significance or empirical Bayes lower-bound claim.'})
    print('context aggregate',len(inputs),'inputs',len(summaries),'groups',len(pairs),'paired groups',flush=True)


if __name__=='__main__':main()
