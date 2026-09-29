"""Compact, fully indexed scene-paired length statistics; no failure imputation."""
import csv
import json
import numpy as np
from common import HERE,r
from dynamics_observation_length_data import NAME,DATA

LENGTHS=[1,2,4,8]
SEEDS=[0,1,2]
STRATA=['all','no_past_contact','past_contact','past_pair_contact','past_wall_contact']
PAIRS=[(2,1),(4,1),(8,1),(4,2),(8,2),(8,4)]
HORIZONS=[1,4,8,16,32,61]


def number(value):
    if value=='':return np.nan
    if value in ['True','False']:return float(value=='True')
    return float(value)


def masks(events):
    contact=events.any(axis=(1,2));pair=events[...,0].any(1);wall=events[...,1].any(1)
    return np.stack([np.ones(len(events),bool),~contact,contact,pair,wall])


def stats(values,axis):
    x=np.asarray(values,np.float64);valid=np.isfinite(x);n=valid.sum(axis)
    mean=np.divide(np.where(valid,x,0).sum(axis),n,out=np.full(n.shape,np.nan),where=n>0)
    residual=np.where(valid,x-np.expand_dims(mean,axis),0)
    variance=np.divide(np.square(residual).sum(axis),n-1,out=np.full(n.shape,np.nan),where=n>1)
    lo=np.where(valid,x,np.inf).min(axis);hi=np.where(valid,x,-np.inf).max(axis)
    return {'mean':mean,'sd':np.sqrt(variance),'min':np.where(n>0,lo,np.nan),'max':np.where(n>0,hi,np.nan),'n':n}


def reduce_scenes(values,groups):
    valid=groups[:,:,None]&np.isfinite(values)[None,:,:]
    count=valid.sum(1);total=groups.sum(1)
    summed=np.where(valid,values[None,:,:],0).sum(1)
    mean=np.divide(summed,count,out=np.full(count.shape,np.nan),where=count>0)
    return mean,count,total


def pair_scenes(left,right,groups):
    assert left.shape==right.shape
    common=np.isfinite(left)&np.isfinite(right)
    difference=np.subtract(left,right,out=np.full(left.shape,np.nan),where=common)
    mean,count,total=reduce_scenes(difference,groups)
    left_mean,lc,_=reduce_scenes(np.where(common,left,np.nan),groups)
    right_mean,rc,_=reduce_scenes(np.where(common,right,np.nan),groups)
    assert np.array_equal(lc,count) and np.array_equal(rc,count)
    return mean,count,total,left_mean,right_mean


def add_source(path,sources):
    sources[str(path.relative_to(HERE))]=r.sha(path)


def load_unit(ds,length,seed,meta,sources):
    method,mode,split=[meta[k] for k in ['method','mode','split']]
    base=HERE/f'runs/{NAME}/d{ds}/L{length}'
    label_path=HERE/f'data/{NAME}/d{ds}/inputs/{mode}/{split}/labels.npz';add_source(label_path,sources)
    with np.load(label_path) as archive:events=archive['past_events'].copy()
    if meta['stage']=='observation':
        folder=base/f'observation_eval/{method}_s{seed}/{mode}/{split}';path=folder/'metrics.json';add_source(path,sources)
        summary=json.loads(path.read_text());assert (summary['data_seed'],summary['length'],summary['seed'])==(ds,length,seed)
        columns=[];keys=[];n=len(events)
        for filename in ['inference_rows.csv','forecast_rows.csv']:
            path=folder/filename;assert r.sha(path)==summary['files'][filename];add_source(path,sources)
        inference=list(csv.DictReader((folder/'inference_rows.csv').open()))
        forecast=list(csv.DictReader((folder/'forecast_rows.csv').open()))
        assert [int(x['episode']) for x in inference]==list(range(n))
        for key in summary['inference']['all']['metrics']:
            columns.append([number(x[key]) for x in inference]);keys.append({'component':'inference','predictor':'','horizon':0,'metric':key})
        for component,entry in summary['forecasts'].items():
            information,predictor=component.split('__');rr=[x for x in forecast if (x['information'],x['predictor'])==(information,predictor)]
            assert [int(x['episode']) for x in rr]==list(range(n))
            for key in entry['all']['metrics']:
                columns.append([number(x[key]) for x in rr]);keys.append({'component':information,'predictor':predictor,'horizon':1,'metric':key})
        matrix=np.asarray(columns,np.float64).T
        # Every source summary uses its own visible-window contact grouping.
        own=masks(events[:,8-length:]);means,counts,totals=reduce_scenes(matrix,own)
        expected=np.full(means.shape,np.nan);known=np.zeros(means.shape,bool)
        for k,key in enumerate(keys):
            categories=summary['inference'] if key['component']=='inference' else summary['forecasts'][key['component']+'__'+key['predictor']]
            for j,stratum in enumerate(STRATA):
                if stratum not in categories:continue
                entry=categories[stratum];assert entry['episodes']==totals[j];value=entry['metrics'][key['metric']]
                known[j,k]=True
                if value is None:assert counts[j,k]==0
                else:expected[j,k]=value['mean']
        np.testing.assert_allclose(means[known],expected[known],atol=1e-10,rtol=1e-12,equal_nan=True)
    else:
        folder=base/f"autonomous/{method}_s{seed}/{mode}/{split}/{meta['predictor']}";path=folder/'metrics.json';add_source(path,sources)
        summary=json.loads(path.read_text());assert (summary['data_seed'],summary['length'],summary['seed'])==(ds,length,seed)
        path=folder/'trajectories.npz';assert r.sha(path)==summary['files']['trajectories.npz'];add_source(path,sources)
        pieces=[];keys=[]
        with np.load(path) as archive:
            for branch,array_name,key_name in [('factual','metric_values','metric_keys'),('counterfactual','cf_metric_values','metric_keys'),('response','response_metric_values','response_metric_keys')]:
                if array_name not in archive:assert summary[branch] is None;continue
                array=archive[array_name].copy();names=archive[key_name].tolist();assert array.shape==(len(events),6,len(names))
                pieces.append(array.reshape(len(events),-1))
                for h in HORIZONS:
                    for name in names:keys.append({'component':branch,'predictor':meta['predictor'],'horizon':h,'metric':name})
                expected_mean=[];expected_count=[];expected_total=[]
                for j,h in enumerate(HORIZONS):
                    row=next(x for x in summary[branch] if x['horizon']==h)
                    for k,name in enumerate(names):
                        entry=row['metrics'][name];expected_mean.append(np.nan if entry['mean'] is None else entry['mean'])
                        expected_count.append(entry['finite_episodes']);expected_total.append(entry['total_episodes'])
                means,counts,totals=reduce_scenes(array.reshape(len(events),-1),np.ones((1,len(events)),bool))
                np.testing.assert_array_equal(counts[0],expected_count);assert (np.array(expected_total)==totals[0]).all()
                np.testing.assert_allclose(means[0],expected_mean,atol=1e-10,rtol=1e-12,equal_nan=True)
        matrix=np.concatenate(pieces,axis=1)
    assert len(matrix)==len(events) and len(keys)==matrix.shape[1]
    return matrix,keys,events


def compute_block(meta):
    sources={};keys=None;result={}
    for di,ds in enumerate(DATA):
        for si,seed in enumerate(SEEDS):
            matrices=[];all_events=None
            for li,length in enumerate(LENGTHS):
                matrix,k,events=load_unit(ds,length,seed,meta,sources)
                if keys is None:
                    keys=k;m=len(k);shape=(4,3,3,5,m);pair_shape=(6,3,3,5,m)
                    result={'unit_mean':np.full(shape,np.nan),'unit_count':np.zeros(shape,np.int64),'unit_total':np.zeros(shape[:-1],np.int64),
                        'pair_unit_mean':np.full(pair_shape,np.nan),'pair_unit_count':np.zeros(pair_shape,np.int64),'pair_unit_total':np.zeros(pair_shape[:-1],np.int64),
                        'pair_unit_left_mean':np.full(pair_shape,np.nan),'pair_unit_right_mean':np.full(pair_shape,np.nan)}
                else:assert k==keys
                if all_events is None:all_events=events
                else:np.testing.assert_array_equal(events,all_events)
                means,counts,totals=reduce_scenes(matrix,masks(events[:,8-length:]))
                result['unit_mean'][li,di,si]=means;result['unit_count'][li,di,si]=counts;result['unit_total'][li,di,si]=totals
                matrices.append(matrix)
            fixed=masks(all_events)
            for pi,(left,right) in enumerate(PAIRS):
                mean,count,total,lm,rm=pair_scenes(matrices[LENGTHS.index(left)],matrices[LENGTHS.index(right)],fixed)
                result['pair_unit_mean'][pi,di,si]=mean;result['pair_unit_count'][pi,di,si]=count;result['pair_unit_total'][pi,di,si]=total
                result['pair_unit_left_mean'][pi,di,si]=lm;result['pair_unit_right_mean'][pi,di,si]=rm
            # Truth-input forecasts and oracle recurrence cannot depend on observed length.
            equal_columns=[j for j,k in enumerate(keys) if (meta['stage']=='autonomous' and meta['method']=='oracle') or (meta['stage']=='observation' and k['component']=='true_state_true_context')]
            if equal_columns:
                for matrix in matrices[1:]:np.testing.assert_array_equal(matrix[:,equal_columns],matrices[0][:,equal_columns])
    for prefix in ['', 'pair_']:
        within=stats(result[prefix+'unit_mean'],axis=2)
        result.update({prefix+'data_'+key:value for key,value in within.items()})
        overall=stats(within['mean'],axis=1)
        result.update({prefix+'across_data_'+key:value for key,value in overall.items()})
    return result,keys,sources
