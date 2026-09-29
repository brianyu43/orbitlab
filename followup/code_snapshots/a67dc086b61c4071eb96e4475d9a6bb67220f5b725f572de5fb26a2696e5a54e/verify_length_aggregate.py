"""Independent raw-scene, common-cohort and hierarchical aggregate checks."""
import argparse
import csv
import json
import time
import numpy as np
from common import HERE,dump,r
from dynamics_observation_length_data import NAME
from aggregate_observation_length import CONFIG,freeze,block_metas,folder_for

DATA=[640031,997101,997102];LENGTHS=[1,2,4,8];PAIRS=[(2,1),(4,1),(8,1),(4,2),(8,2),(8,4)]
STRATA=['all','no_past_contact','past_contact','past_pair_contact','past_wall_contact']
HORIZONS=[1,4,8,16,32,61]


def scalar(text):
    if not text:return float('nan')
    if text=='True':return 1.
    if text=='False':return 0.
    return float(text)


def independent_unit(meta,ds,length,seed):
    mode,split,method=[meta[k] for k in ['mode','split','method']]
    with np.load(HERE/f'data/{NAME}/d{ds}/world/{mode}/{split}/trajectories.npz') as a:events=a['events'][:,:7].copy()
    base=HERE/f'runs/{NAME}/d{ds}/L{length}';n=len(events)
    keys=[];vectors=[]
    if meta['stage']=='observation':
        folder=base/f'observation_eval/{method}_s{seed}/{mode}/{split}'
        summary=json.loads((folder/'metrics.json').read_text())
        rows={int(x['episode']):x for x in csv.DictReader((folder/'inference_rows.csv').open())};assert set(rows)==set(range(n))
        for metric in summary['inference']['all']['metrics']:
            keys.append({'component':'inference','predictor':'','horizon':0,'metric':metric})
            vectors.append(np.array([scalar(rows[i][metric]) for i in range(n)]))
        rows={}
        for row in csv.DictReader((folder/'forecast_rows.csv').open()):
            key=row['information'],row['predictor'],int(row['episode']);assert key not in rows;rows[key]=row
        assert len(rows)==20*n
        for component,categories in summary['forecasts'].items():
            information,predictor=component.split('__')
            for metric in categories['all']['metrics']:
                keys.append({'component':information,'predictor':predictor,'horizon':1,'metric':metric})
                vectors.append(np.array([scalar(rows[information,predictor,i][metric]) for i in range(n)]))
    else:
        folder=base/f"autonomous/{method}_s{seed}/{mode}/{split}/{meta['predictor']}"
        with np.load(folder/'trajectories.npz') as a:
            for branch,value_key,name_key in [('factual','metric_values','metric_keys'),('counterfactual','cf_metric_values','metric_keys'),('response','response_metric_values','response_metric_keys')]:
                if value_key not in a:continue
                values=a[value_key];names=a[name_key].tolist();assert values.shape==(n,6,len(names))
                for j,h in enumerate(HORIZONS):
                    for k,metric in enumerate(names):
                        keys.append({'component':branch,'predictor':meta['predictor'],'horizon':h,'metric':metric})
                        vectors.append(values[:,j,k].astype(np.float64))
    assert len({tuple(x.values()) for x in keys})==len(keys)
    return np.column_stack(vectors),keys,events


def independent_masks(events):
    any_contact=np.count_nonzero(events,axis=(1,2))>0
    return [np.ones(len(events),bool),~any_contact,any_contact,
        np.count_nonzero(events[:,:,0],axis=1)>0,np.count_nonzero(events[:,:,1],axis=1)>0]


def reference_means(matrix,groups):
    means=[];counts=[];totals=[]
    for group in groups:
        selected=np.ma.masked_invalid(matrix[group]);count=selected.count(axis=0)
        mean=selected.mean(axis=0).filled(np.nan) if group.any() else np.full(matrix.shape[1],np.nan)
        means.append(mean);counts.append(count);totals.append(int(group.sum()))
    return np.asarray(means),np.asarray(counts),np.asarray(totals)


def reference_stats(values,axis):
    masked=np.ma.masked_invalid(values);n=masked.count(axis=axis)
    return {'mean':masked.mean(axis).filled(np.nan),'sd':np.where(n>1,masked.std(axis=axis,ddof=1).filled(np.nan),np.nan),
            'min':masked.min(axis).filled(np.nan),'max':masked.max(axis).filled(np.nan),'n':n}


def close(a,b):np.testing.assert_allclose(a,b,atol=1e-12,rtol=5e-12,equal_nan=True)


def verify_block(meta,manifest,arrays):
    keys=manifest['scalar_keys'];oracle_columns=[i for i,key in enumerate(keys)
        if (meta['stage']=='autonomous' and meta['method']=='oracle') or (meta['stage']=='observation' and key['component']=='true_state_true_context')]
    samples=0
    for di,ds in enumerate(DATA):
        for seed in [0,1,2]:
            matrices=[];events=None
            for li,length in enumerate(LENGTHS):
                values,got,ev=independent_unit(meta,ds,length,seed);assert got==keys
                if events is None:events=ev
                else:np.testing.assert_array_equal(ev,events)
                mean,count,total=reference_means(values,independent_masks(ev[:,8-length:]))
                close(arrays['unit_mean'][li,di,seed],mean)
                np.testing.assert_array_equal(arrays['unit_count'][li,di,seed],count)
                np.testing.assert_array_equal(arrays['unit_total'][li,di,seed],total)
                matrices.append(values);samples+=values.size
            if oracle_columns:
                for matrix in matrices[1:]:np.testing.assert_array_equal(matrix[:,oracle_columns],matrices[0][:,oracle_columns])
            groups=independent_masks(events)
            for pi,(long,short) in enumerate(PAIRS):
                left=matrices[LENGTHS.index(long)];right=matrices[LENGTHS.index(short)]
                both=np.isfinite(left)&np.isfinite(right);left=np.where(both,left,np.nan);right=np.where(both,right,np.nan)
                mean,count,total=reference_means(left-right,groups);lm,_,_=reference_means(left,groups);rm,_,_=reference_means(right,groups)
                close(arrays['pair_unit_mean'][pi,di,seed],mean)
                close(arrays['pair_unit_left_mean'][pi,di,seed],lm);close(arrays['pair_unit_right_mean'][pi,di,seed],rm)
                np.testing.assert_array_equal(arrays['pair_unit_count'][pi,di,seed],count)
                np.testing.assert_array_equal(arrays['pair_unit_total'][pi,di,seed],total)
    for prefix in ['', 'pair_']:
        within=reference_stats(arrays[prefix+'unit_mean'],axis=2)
        for key,value in within.items():close(arrays[prefix+'data_'+key],value)
        overall=reference_stats(within['mean'],axis=1)
        for key,value in overall.items():close(arrays[prefix+'across_data_'+key],value)
    return samples


def preflight():
    from observation_length_aggregation import load_unit,pair_scenes,reduce_scenes,stats
    source={};cases=[]
    for stage in ['observation','autonomous']:
        meta={'stage':stage,'method':'measurement_mlp','mode':'variable_force','split':'test','predictor':'joint_exact'}
        matrices=[]
        for length in [1,2]:
            actual,keys,events=independent_unit(meta,640031,length,0)
            expected,kk,ev=load_unit(640031,length,0,meta,source)
            assert keys==kk;np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(events,ev)
            groups=independent_masks(events[:,8-length:]);a=reference_means(actual,groups);b=reduce_scenes(expected,np.array(groups))
            for x,y in zip(a,b):close(x,y)
            matrices.append(actual)
        groups=independent_masks(events);common=np.isfinite(matrices[0])&np.isfinite(matrices[1])
        left=np.where(common,matrices[1],np.nan);right=np.where(common,matrices[0],np.nan)
        expected=pair_scenes(matrices[1],matrices[0],np.array(groups))
        refs=reference_means(left-right,groups)
        for a,b in zip(expected[:3],refs):close(a,b)
        close(expected[3],reference_means(left,groups)[0]);close(expected[4],reference_means(right,groups)[0])
        cases.append({'stage':stage,'scalar_columns':len(keys),'episodes':len(actual)})
    toy=np.array([[[1,2],[np.nan,5],[3,9]],[[np.nan,2],[np.nan,2],[np.nan,2]]])
    for axis in [0,1]:
        a=reference_stats(toy,axis);b=stats(toy,axis)
        for key in a:close(a[key],b[key])
    dump(HERE/f'reports/{NAME}/aggregation_verifier_preflight.json',{'all_passed':True,'actual_source_cases':cases,
        'independent_raw_loader_and_scene_pair_arithmetic_agree':True,'verifier_sha256':r.sha(__file__),
        'aggregate_core_sha256':r.sha(HERE/'observation_length_aggregation.py'),
        'scope':'Verifier preflight on first-data lengths 1/2, not complete cross-data aggregation.'})
    print('independent length aggregate verifier preflight passed',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['observation','autonomous','all'],default='all')
    parser.add_argument('--available',action='store_true');parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    if args.preflight_only:preflight();return
    freeze();vp=HERE/f'reports/{NAME}/aggregation_verifier_preflight.json';v=json.loads(vp.read_text());assert v['all_passed'] and v['verifier_sha256']==r.sha(__file__)
    root=HERE/f'reports/{NAME}/aggregates_v1'
    for stage in ['observation','autonomous'] if args.stage=='all' else [args.stage]:
        results=[];start=time.time();metas=block_metas(stage)
        assert len(metas)==(52 if stage=='observation' else 468)
        for meta in metas:
            folder=folder_for(meta);path=folder/'manifest.json'
            if not path.exists():
                if args.available:continue
                raise RuntimeError(f'Aggregate block not ready: {path}')
            manifest=json.loads(path.read_text());assert manifest['meta']==meta
            assert manifest['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
            assert manifest['axes']=={'lengths':LENGTHS,'data_seeds':DATA,'initializations':[0,1,2],'strata':STRATA,'pairs_long_minus_short':[list(p) for p in PAIRS]}
            for file,sha in manifest['files'].items():assert r.sha(folder/file)==sha
            for file,sha in manifest['sources'].items():assert r.sha(HERE/file)==sha
            receipt=folder/'verification.json'
            if receipt.exists():
                result=json.loads(receipt.read_text());assert result['all_passed'] and result['block_manifest_sha256']==r.sha(path)
                assert result['verifier_sha256']==r.sha(__file__)
            else:
                with np.load(folder/'statistics.npz') as archive:arrays={key:archive[key] for key in archive.files}
                assert list(arrays['unit_mean'].shape)==manifest['unit_shape'] and list(arrays['pair_unit_mean'].shape)==manifest['pair_unit_shape']
                samples=verify_block(meta,manifest,arrays)
                result={'all_passed':True,'meta':meta,'raw_scene_scalar_values_checked':samples,'unit_cells_checked':int(arrays['unit_mean'].size),
                    'paired_unit_cells_checked':int(arrays['pair_unit_mean'].size),'block_manifest_sha256':r.sha(path),'verifier_sha256':r.sha(__file__)}
                dump(receipt,result)
            results.append(result);print('verified length aggregate',stage,meta['method'],meta['predictor'],meta['mode'],meta['split'],len(results),flush=True)
        complete=len(results)==len(metas)
        report={'all_passed':complete,'all_available_checked_passed':True,'blocks_verified':len(results),'expected_blocks':len(metas),
            'raw_scene_scalar_values_checked':sum(x['raw_scene_scalar_values_checked'] for x in results),
            'unit_cells_checked':sum(x['unit_cells_checked'] for x in results),'paired_unit_cells_checked':sum(x['paired_unit_cells_checked'] for x in results),
            'results':results,'verifier_sha256':r.sha(__file__),'seconds':time.time()-start}
        dump(root/f'{stage}_verification_progress.json',report)
        if complete:
            summary=root/f'{stage}_manifest.json';s=json.loads(summary.read_text());assert s['block_count']==len(metas)
            assert {tuple(x['meta'].values()) for x in s['blocks']}=={tuple(x.values()) for x in metas}
            for item in s['blocks']:assert item['sha256']==r.sha(HERE/item['path'])
            for file,sha in s['verification_sha256'].items():assert r.sha(HERE/file)==sha and json.loads((HERE/file).read_text())['all_passed']
            report['manifest_sha256']=r.sha(summary);dump(root/f'{stage}_verification.json',report)


if __name__=='__main__':main()
