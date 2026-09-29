"""Reconstruct every cross-data scalar and paired summary independently."""
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np
from common import HERE,dump,r
from dynamics_context_omission import NAME,DATA


FIELDS=['condition','kind','mode','split','category','metric']
PAIR_FIELDS=['contrast','left_condition','right_condition','left_kind','right_kind','mode','split','category','metric']


def check(value,expected):
    if expected is None:assert value==''
    else:np.testing.assert_allclose(float(value),expected,atol=1e-12,rtol=1e-12)


def check_stats(row,prefix,values):
    vals=np.array([x for x in values if x is not None],np.float64)
    checks={'mean':vals.mean() if len(vals) else None,'sd':vals.std(ddof=1) if len(vals)>1 else None,
        'min':vals.min() if len(vals) else None,'max':vals.max() if len(vals) else None,'available':len(vals)}
    for key,value in checks.items():check(row[prefix+key],value)
    return checks['mean']


def main():
    root=HERE/f'reports/{NAME}';out=root/'across_data';m=json.loads((out/'manifest.json').read_text())
    v=json.loads((root/'verification.json').read_text());assert v['all_passed'] and m['verification_sha256']==r.sha(root/'verification.json')
    assert m['script_sha256']==r.sha(HERE/'aggregate_context_omission.py')
    for file,sha in m['files'].items():assert r.sha(out/file)==sha
    raw={};source_count=0;episode_keys={}
    for file,sha in m['source_sha256'].items():
        assert r.sha(HERE/file)==sha;path=HERE/file;source_count+=1
        parts=path.parts;ds=int(parts[-4][1:]);condition=parts[-3];kind,seed=parts[-2].rsplit('_s',1)
        key_hash=hashlib.sha256()
        with (path.parent/'rows.csv').open() as stream:
            for row in csv.DictReader(stream):
                key_hash.update(json.dumps([row[k] for k in ['mode','split','episode','category','transitions']],separators=(',',':')).encode())
        if ds in episode_keys:assert episode_keys[ds]==key_hash.hexdigest()
        else:episode_keys[ds]=key_hash.hexdigest()
        for group,categories in json.loads(path.read_text())['groups'].items():
            mode,split=group.split('__')
            for category,entry in categories.items():
                for metric in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                    s=entry[metric];key=(condition,kind,mode,split,category,metric,ds,int(seed))
                    assert key not in raw;raw[key]=(None if s is None else s['mean'],entry['episodes'])
    assert source_count==108
    grouped=defaultdict(dict);rows=list(csv.DictReader((out/'inputs.csv').open()));seen=set()
    assert len(rows)==m['input_scalars']==16200
    for row in rows:
        meta=tuple(row[k] for k in FIELDS);unit=int(row['data_seed']),int(row['seed']);key=(*meta,*unit)
        assert key not in seen;seen.add(key);value,n=raw[key];check(row['value'],value);assert int(row['episodes'])==n
        grouped[meta][unit]=value
    assert seen==set(raw)
    def check_groups(filename,ident,values,count):
        rr=list(csv.DictReader((out/filename).open()));assert len(rr)==count;seen=set()
        for row in rr:
            key=tuple(row[k] for k in ident);assert key not in seen;seen.add(key);v=values[key];means=[]
            assert len(v)==9
            for ds in DATA:means.append(check_stats(row,f'data_{ds}_',[v[ds,s] for s in [0,1,2]]))
            check_stats(row,'across_data_',means)
        assert seen==set(values)
    check_groups('metrics.csv',FIELDS,grouped,m['metric_groups'])
    paired=defaultdict(dict);seen=set();pr=list(csv.DictReader((out/'paired_inputs.csv').open()))
    assert len(pr)==m['paired_input_scalars']==28350
    for row in pr:
        meta=tuple(row[k] for k in PAIR_FIELDS);ds,seed=int(row['data_seed']),int(row['seed']);unit=ds,seed
        assert (meta,unit) not in seen;seen.add((meta,unit));tail=tuple(row[k] for k in FIELDS[2:])
        left,nl=raw[(row['left_condition'],row['left_kind'],*tail,ds,seed)]
        right,nr=raw[(row['right_condition'],row['right_kind'],*tail,ds,seed)];assert nl==nr
        if row['contrast']=='context':assert row['right_condition']=='full' and row['left_condition']!='full' and row['left_kind']==row['right_kind']
        else:assert row['contrast']=='model' and row['left_condition']==row['right_condition'] and (row['left_kind'],row['right_kind']) in [('wrong_exact','interaction'),('joint_exact','interaction'),('joint_exact','wrong_exact')]
        diff=None if left is None or right is None else left-right
        check(row['left'],left);check(row['right'],right);check(row['difference'],diff);paired[meta][unit]=diff
    check_groups('paired.csv',PAIR_FIELDS,paired,m['paired_groups'])
    assert m['metric_groups']==1800 and m['paired_groups']==3150
    dump(out/'verification.json',{'all_passed':True,'source_files_checked':108,'source_scalars_checked':len(raw),
        'aggregate_groups_checked':len(grouped),'paired_scalar_differences_checked':len(pr),'paired_aggregate_groups_checked':len(paired),
        'same_episode_denominators_for_each_pair':True,'same_episode_category_transition_keys_checked':episode_keys,'missing_strata_preserved':True,
        'manifest_sha256':r.sha(out/'manifest.json'),'verifier_sha256':r.sha(__file__)})
    print('context aggregate audit passed',len(grouped),len(paired),flush=True)


if __name__=='__main__':main()
