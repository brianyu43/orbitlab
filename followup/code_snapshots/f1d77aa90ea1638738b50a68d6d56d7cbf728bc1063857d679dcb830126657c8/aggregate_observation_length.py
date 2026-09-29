"""Stream all paired length results into compact, provenance-bound tensor blocks."""
import argparse
import json
import time
import numpy as np
from common import HERE,dump,r
from dynamics_observation_length_data import NAME,DATA
from observation_length_aggregation import LENGTHS,SEEDS,STRATA,PAIRS,compute_block

CONFIG='dynamics_observation_length_aggregate_v1'


def freeze():
    preflight=HERE/f'reports/{NAME}/aggregation_preflight.json';p=json.loads(preflight.read_text());assert p['all_passed']
    for path,sha in p['source_sha256'].items():assert r.sha(HERE/path)==sha,path
    path=HERE/f'configs/{CONFIG}.json'
    if not path.exists():
        dump(path,{'data_seeds':DATA,'lengths':LENGTHS,'seeds':SEEDS,'strata':STRATA,'pairs_long_minus_short':PAIRS,
            'unit_axes':['length','data_seed','initialization','visible_window_stratum','scalar_key'],
            'pair_unit_axes':['length_pair','data_seed','initialization','fixed_L8_stratum','scalar_key'],
            'within_data':'Equal-weight mean across available fixed initializations; SD/range/count retained.',
            'across_data':'Equal-weight mean across 3 independent data units; descriptive SD/range/count, no iid initialization or significance claim.',
            'pairing':'Same data/initialization/episode; difference computed only where both values are finite. Common-cohort left/right means and finite/total denominators retained. Full-cohort failure indicators remain separate metrics.',
            'empty':'NaN means missing; count=0. Never impute zero error or success.',
            'preflight_sha256':r.sha(preflight),'source_sha256':{f:r.sha(HERE/f) for f in ['aggregate_observation_length.py','observation_length_aggregation.py','planning_C07_LENGTH_AGGREGATION_KO.md']},
            'parent_config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'frozen_unix':time.time()})
    c=json.loads(path.read_text());assert c['preflight_sha256']==r.sha(preflight)
    for f,sha in c['source_sha256'].items():assert r.sha(HERE/f)==sha,f
    assert c['parent_config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    return c


def gates(stage):
    evidence={}
    name='observation_eval' if stage=='observation' else 'autonomous'
    script='verify_observation_length_eval.py' if stage=='observation' else 'verify_observation_length_autonomous.py'
    for ds in DATA:
        for length in LENGTHS:
            root=HERE/f'reports/{NAME}/d{ds}/L{length}';path=root/f'{name}_verification.json'
            report=json.loads(path.read_text());assert report['all_passed']
            assert (report['data_seed'],report['length'])==(ds,length)
            assert report['conditions_verified']==(156 if stage=='observation' else 1404)
            assert report['summary_sha256']==r.sha(root/f'{name}_summary.json')
            assert report['verifier_sha256']==r.sha(HERE/script)
            evidence[str(path.relative_to(HERE))]=r.sha(path)
    return evidence


def block_metas(stage):
    groups=[g for g in json.loads((HERE/f'data/{NAME}/manifest.json').read_text())['groups'] if g['data_seed']==DATA[0] and g['split'] not in ['train','val']]
    methods=['rgb_cnn','measurement_mlp','rgb_analytic','true_positions_analytic'] if stage=='observation' else ['oracle','rgb_cnn','measurement_mlp','rgb_analytic']
    predictors=[''] if stage=='observation' else ['flat','object','interaction','augmented','wrong_exact','joint_exact','relaxed','inertial','force_wall']
    return [{'stage':stage,'method':method,'predictor':predictor,'mode':g['mode'],'split':g['split']}
            for method in methods for predictor in predictors for g in groups]


def folder_for(meta):
    root=HERE/f'reports/{NAME}/aggregates_v1'/meta['stage']/meta['method']
    if meta['predictor']:root/=meta['predictor']
    return root/meta['mode']/meta['split']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['observation','autonomous','all'],default='all')
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();freeze()
    if args.prepare_only:return
    root=HERE/f'reports/{NAME}/aggregates_v1';root.mkdir(parents=True,exist_ok=True)
    for stage in ['observation','autonomous'] if args.stage=='all' else [args.stage]:
        evidence=gates(stage);items=[];start=time.time();metas=block_metas(stage)
        for meta in metas:
            folder=folder_for(meta);path=folder/'manifest.json'
            if path.exists():
                item=json.loads(path.read_text());assert item['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
                assert item['meta']==meta
                for filename,sha in item['files'].items():assert r.sha(folder/filename)==sha
                for source,sha in item['sources'].items():assert r.sha(HERE/source)==sha
            else:
                if folder.exists():raise RuntimeError(f'Incomplete aggregate block retained: {folder}')
                arrays,keys,sources=compute_block(meta);folder.mkdir(parents=True)
                np.savez_compressed(folder/'statistics.npz',**arrays)
                item={'meta':meta,'scalar_keys':keys,'sources':sources,
                    'axes':{'lengths':LENGTHS,'data_seeds':DATA,'initializations':SEEDS,'strata':STRATA,'pairs_long_minus_short':PAIRS},
                    'unit_shape':list(arrays['unit_mean'].shape),'pair_unit_shape':list(arrays['pair_unit_mean'].shape),
                    'unit_scalar_cells':int(arrays['unit_mean'].size),'paired_unit_scalar_cells':int(arrays['pair_unit_mean'].size),
                    'files':{'statistics.npz':r.sha(folder/'statistics.npz')},'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json')}
                dump(path,item)
            items.append({'meta':meta,'path':str(path.relative_to(HERE)),'sha256':r.sha(path),
                          'unit_scalar_cells':item['unit_scalar_cells'],'paired_unit_scalar_cells':item['paired_unit_scalar_cells']})
            dump(root/f'{stage}_progress.json',{'completed_blocks':len(items),'expected_blocks':len(metas),'seconds':time.time()-start})
            print('length aggregate block',stage,meta['method'],meta['predictor'],meta['mode'],meta['split'],len(items),flush=True)
        dump(root/f'{stage}_manifest.json',{'blocks':items,'block_count':len(items),'verification_sha256':evidence,
            'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'unit_scalar_cells':sum(x['unit_scalar_cells'] for x in items),
            'paired_unit_scalar_cells':sum(x['paired_unit_scalar_cells'] for x in items),'seconds':time.time()-start,
            'scope':'All raw summary metrics and horizons retained. Individual-length strata use visible history; paired strata fix L8 history. Common-finite episode differences preserve matched cohorts; separate failure metrics use their original full denominators.'})


if __name__=='__main__':main()
