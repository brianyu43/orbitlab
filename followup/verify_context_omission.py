"""Incremental checkpoint, sampling, context masking and prediction audit."""
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_context_omission import NAME,DATA,KINDS,MASKS,freeze,world,source_model
from dynamics_models import Transition,load_split


def sampling(c,seed):
    generator=torch.Generator().manual_seed(c['sampling_seed_base']+seed);sh=hashlib.sha256();ph=hashlib.sha256()
    for _ in range(c['steps']):
        indices=torch.cat([torch.randint(256*19,(32,),generator=generator),torch.randint(256*19,(32,),generator=generator)])
        order=torch.rand(64,4,generator=generator).argsort(-1);sh.update(indices.numpy().tobytes());ph.update(order.numpy().tobytes())
    return sh.hexdigest(),ph.hexdigest(),generator.get_state()


@torch.no_grad()
def verify(c,ds,condition,kind,seed,data,sample):
    folder=HERE/f'runs/{NAME}/d{ds}/{condition}/{kind}_s{seed}';run=json.loads((folder/'run.json').read_text())
    metrics=json.loads((folder/'metrics.json').read_text());config_sha=r.sha(HERE/f'configs/{NAME}.json')
    assert (run['data_seed'],run['condition'],run['kind'],run['seed'])==(ds,condition,kind,seed)
    assert metrics['config_sha256']==run['config_sha256']==config_sha
    assert metrics['checkpoint_sha256']==run['checkpoint_sha256']==r.sha(HERE/run['checkpoint_path'])
    original=source_model(ds,kind,seed);old=json.loads((original/'run.json').read_text())
    sh,ph,rng=sample;assert run['sample_sha256']==old['sample_sha256']==sh and run['permutation_sha256']==old['permutation_sha256']==ph
    torch.manual_seed(c['model_seed_base']+seed);model=Transition(kind,c['hidden'])
    assert state_digest(model)==run['initial_weights_sha256']==old['initial_weights_sha256']
    artifacts=[folder/'run.json',folder/'metrics.json',HERE/run['checkpoint_path'],folder/'evaluation.pt',folder/'rows.csv']
    if condition=='full':
        assert run['new_training_steps']==0 and run['source_run_sha256']==r.sha(original/'run.json')
        assert (HERE/run['checkpoint_path']).resolve()==(original/'model.pt').resolve()
    else:
        assert run['steps']==run['new_training_steps']==12000
        assert run['training_archive_sha256']==r.sha(world(ds)/'variable_force/train/trajectories.npz')
        op=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
        assert torch.equal(op['sampling_rng'],rng) and all(int(x['step'])==c['steps'] for x in op['optimizer']['state'].values())
        artifacts.append(folder/'optimizer.pt')
    ck=torch.load(HERE/run['checkpoint_path'],map_location='cpu',weights_only=True);assert ck['kind']==kind
    model.load_state_dict(ck['state_dict']);model.eval()
    for f,sha in metrics['files'].items():assert r.sha(folder/f)==sha
    stored=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True);records=list(csv.DictReader((folder/'rows.csv').open()))
    assert set(stored)==set(metrics['groups'])=={m+'__'+s for m,s in data}
    predictions=rows=aggregates=full_matches=0
    for (mode,split),item in data.items():
        name=mode+'__'+split;assert metrics['data_sha256'][name]==r.sha(world(ds)/mode/split/'trajectories.npz')
        context=item['context'].clone()
        if condition in ['no_net','no_context']:context[:,:2]=0
        if condition in ['no_drag','no_context']:context[:,2]=0
        pred=[]
        for i in range(0,len(item['state']),128):pred.append(model(item['state'][i:i+128],item['action'][i:i+128],context[i:i+128]))
        pred=torch.cat(pred);torch.testing.assert_close(pred,stored[name],rtol=0,atol=0);assert torch.isfinite(pred).all();predictions+=len(pred)
        if condition=='full' and mode=='variable_force' and split not in ['train','val']:
            ref=torch.load(original/'evaluation.pt',map_location='cpu',weights_only=True)[split]
            torch.testing.assert_close(pred,ref,rtol=0,atol=0);full_matches+=len(pred)
        alive=item['state'][...,-1].numpy();err=np.abs(pred.numpy()-item['next'].numpy())*np.array([31.5,31.5,3,3])
        assert not pred.numpy()[alive==0].any()
        pos=(err[...,:2].sum(-1)*alive).sum(-1)/(2*alive.sum(-1));vel=(err[...,2:].sum(-1)*alive).sum(-1)/(2*alive.sum(-1))
        pair=item['events'][:,0].numpy()>0;wall=item['events'][:,1].numpy()>0;episode=item['episode'].numpy()
        categories={'all':np.ones(len(pair),bool),'no_contact':~(pair|wall),'contact':pair|wall,'pair':pair,'wall_only':wall&~pair}
        rr=[x for x in records if (x['mode'],x['split'])==(mode,split)]
        expected={(int(i),key) for i in np.unique(episode) for key,selection in categories.items() if ((episode==i)&selection).any()}
        assert len(rr)==len(expected) and {(int(x['episode']),x['category']) for x in rr}==expected
        for row in rr:
            selected=(episode==int(row['episode']))&categories[row['category']];assert selected.sum()==int(row['transitions'])
            np.testing.assert_allclose([pos[selected].mean(),vel[selected].mean()],
                [float(row['position_mae_pixels']),float(row['velocity_mae_pixels_per_frame'])],atol=1e-6,rtol=1e-6);rows+=1
        for category,entry in metrics['groups'][name].items():
            selected=[x for x in rr if x['category']==category];assert len(selected)==entry['episodes']
            for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                if not selected:assert entry[key] is None;continue
                computed=r.bootstrap([float(x[key]) for x in selected]);assert computed==entry[key];aggregates+=1
    return {'all_passed':True,'data_seed':ds,'condition':condition,'kind':kind,'seed':seed,'predictions_replayed':predictions,
        'episode_rows_verified':rows,'aggregate_and_interval_checks':aggregates,'full_context_original_predictions_matched':full_matches,
        'same_initialization_samples_and_mask_contract_verified':True,'artifacts':{str(p.relative_to(HERE)):r.sha(p) for p in artifacts},
        'verifier_sha256':r.sha(__file__)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');args=p.parse_args();c=freeze();torch.set_num_threads(2)
    sample={seed:sampling(c,seed) for seed in c['seeds']};results=[]
    for ds in DATA:
        data={(g['mode'],g['split']):load_split(world(ds),g['mode'],g['split']) for g in c['groups'] if g['data_seed']==ds}
        for split in ['train','val']:data['variable_force',split]=load_split(world(ds),'variable_force',split)
        for condition in MASKS:
            for seed in c['seeds']:
                for kind in KINDS:
                    folder=HERE/f'runs/{NAME}/d{ds}/{condition}/{kind}_s{seed}'
                    if not (folder/'metrics.json').exists():
                        if args.available:continue
                        raise RuntimeError(f'Condition not yet complete: {folder}')
                    receipt=HERE/f'reports/{NAME}/receipts/d{ds}_{condition}_{kind}_s{seed}.json'
                    if receipt.exists():
                        result=json.loads(receipt.read_text());assert result['verifier_sha256']==r.sha(__file__)
                        for path,sha in result['artifacts'].items():assert r.sha(HERE/path)==sha,path
                    else:result=verify(c,ds,condition,kind,seed,data,sample[seed]);dump(receipt,result)
                    results.append(result);print('verified omission',ds,condition,kind,seed,flush=True)
    pairs=[];available={(x['data_seed'],x['condition'],x['kind'],x['seed']) for x in results}
    for ds in DATA:
        for condition in ['no_net','no_context']:
            for seed in c['seeds']:
                if not all((ds,condition,k,seed) in available for k in ['wrong_exact','joint_exact']):continue
                folders=[HERE/f'runs/{NAME}/d{ds}/{condition}/{k}_s{seed}' for k in ['wrong_exact','joint_exact']]
                weights=[torch.load(f/'model.pt',map_location='cpu',weights_only=True)['state_dict'] for f in folders]
                gap=max(float((weights[0][k]-weights[1][k]).abs().max()) for k in weights[0])
                pairs.append({'data_seed':ds,'condition':condition,'seed':seed,'max_weight_difference':gap,'weights_exactly_equal':gap==0})
    complete=len(results)==108
    report={'all_passed':complete,'all_available_checked_passed':True,'conditions_verified':len(results),'expected_conditions':108,
        'predictions_replayed':sum(x['predictions_replayed'] for x in results),'episode_rows_verified':sum(x['episode_rows_verified'] for x in results),
        'aggregate_and_interval_checks':sum(x['aggregate_and_interval_checks'] for x in results),'zero_net_structure_training_comparisons':pairs,
        'results':results,'verifier_sha256':r.sha(__file__),'scope':'Available full-context and identically-masked retrained models. Full completion requires 108 conditions and subsequent across-data analysis.'}
    root=HERE/f'reports/{NAME}';dump(root/'verification_progress.json',report)
    if complete:
        source=root/'summary.json';summary=json.loads(source.read_text());assert summary['conditions']==108 and summary['new_trainings']==81
        for item in summary['results']:
            folder=HERE/f"runs/{NAME}/d{item['data_seed']}/{item['condition']}/{item['kind']}_s{item['seed']}"
            assert item['run']==json.loads((folder/'run.json').read_text()) and item['metrics']==json.loads((folder/'metrics.json').read_text())
        assert len(pairs)==18;report['summary_sha256']=r.sha(source);dump(root/'verification.json',report)
    print('omission audit',len(results),'/108',flush=True)


if __name__=='__main__':main()
