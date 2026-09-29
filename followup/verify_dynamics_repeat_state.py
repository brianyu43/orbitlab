"""Incremental audit of completed immutable state repeats; no training restart."""
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_repeat_data import NAME
from dynamics_repeat_state import CONFIG,freeze
from dynamics_models import Transition,load_split,rotate_state,rotate_vec,rotate_context,rotate_motion


def sampling_reference(c,seed,n):
    rng=torch.Generator().manual_seed(c['sampling_seed_base']+seed);aug=torch.Generator().manual_seed(c['augmentation_seed_base']+seed)
    sh=hashlib.sha256();ph=hashlib.sha256()
    for _ in range(c['steps']):
        idx=torch.cat([torch.randint(n,(32,),generator=rng),torch.randint(n,(32,),generator=rng)])
        order=torch.rand(64,4,generator=rng).argsort(-1);sh.update(idx.numpy().tobytes());ph.update(order.numpy().tobytes())
        torch.randint(4,(),generator=aug)
    return sh.hexdigest(),ph.hexdigest(),rng.get_state(),aug.get_state()


@torch.no_grad()
def verify_model(c,data_seed,mode,kind,seed,data,training,sampling):
    folder=HERE/f'runs/{NAME}/d{data_seed}/state/{mode}/{kind}_s{seed}'
    run=json.loads((folder/'run.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    assert run['data_seed']==data_seed and run['mode']==mode and run['kind']==kind and run['seed']==seed
    assert run['steps']==run['new_training_steps']==12000 and run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
    assert run['training_archive_sha256']==r.sha(HERE/f'data/{NAME}/d{data_seed}/world/{mode}/train/trajectories.npz')
    assert run['checkpoint_sha256']==metrics['checkpoint_sha256']==r.sha(folder/'model.pt')
    assert metrics['evaluation_sha256']==r.sha(folder/'evaluation.pt') and metrics['rows_sha256']==r.sha(folder/'rows.csv')
    sh,ph,rng,aug=sampling
    assert run['sample_sha256']==sh and run['permutation_sha256']==ph
    optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
    assert torch.equal(optimizer['sampling_rng'],rng)
    if kind=='augmented':assert torch.equal(optimizer['augmentation_rng'],aug)
    assert all(int(v['step'])==c['steps'] for v in optimizer['optimizer']['state'].values())
    old=json.loads((HERE/f'runs/dynamics_state_study_v1/{mode}/{kind}_s{seed}/run.json').read_text())
    assert old['sample_sha256']==sh and old['permutation_sha256']==ph
    torch.manual_seed(c['model_seed_base']+seed);model=Transition(kind,c['hidden'])
    assert state_digest(model)==run['initial_weights_sha256']==old['initial_weights_sha256']
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
    assert sum(v.numel() for v in model.parameters())==run['parameters']
    saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True);rows=list(csv.DictReader((folder/'rows.csv').open()))
    assert set(saved)==set(data)==set(metrics['splits'])
    predictions=checks=row_count=0
    for split,item in data.items():
        assert metrics['data_sha256'][split]==r.sha(HERE/f'data/{NAME}/d{data_seed}/world/{mode}/{split}/trajectories.npz')
        pp=[]
        for start in range(0,len(item['state']),128):pp.append(model(item['state'][start:start+128],item['action'][start:start+128],item['context'][start:start+128]))
        torch.testing.assert_close(torch.cat(pp),saved[split],atol=0,rtol=0);assert torch.isfinite(saved[split]).all()
        predictions+=len(item['state']);live=item['state'][...,-1].numpy()
        err=np.abs(saved[split].numpy()-item['next'].numpy())*np.array([31.5,31.5,3,3])
        position=(err[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1));velocity=(err[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1))
        assert not np.any(saved[split].numpy()[live==0])
        pair=item['events'][:,0].numpy()>0;wall=item['events'][:,1].numpy()>0
        categories={'all':np.ones(len(pair),bool),'no_contact':~(pair|wall),'contact':pair|wall,'pair':pair,'wall_only':wall&~pair}
        selected_rows=[v for v in rows if v['split']==split]
        expected={(int(i),category) for i in torch.unique(item['episode']).tolist() for category,selection in categories.items()
                  if ((item['episode'].numpy()==i)&selection).any()}
        assert {(int(v['episode']),v['category']) for v in selected_rows}==expected and len(selected_rows)==len(expected)
        for row in selected_rows:
            select=(item['episode'].numpy()==int(row['episode']))&categories[row['category']]
            assert select.sum()==int(row['transitions'])
            np.testing.assert_allclose([position[select].mean(),velocity[select].mean()],
                [float(row['position_mae_pixels']),float(row['velocity_mae_pixels_per_frame'])],atol=1e-6,rtol=1e-6);row_count+=1
        for category,m in metrics['splits'][split].items():
            selected=[v for v in selected_rows if v['category']==category];assert len(selected)==m['episodes']
            for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                expected=r.bootstrap([float(v[key]) for v in selected]);assert abs(expected['mean']-m[key]['mean'])<1e-10
                np.testing.assert_allclose(expected['base_scene_ci95'],m[key]['base_scene_ci95'],atol=1e-10,rtol=0);checks+=1
        if split=='test':
            ii=torch.arange(32)*64+3;s=item['state'][ii];act=item['action'][ii];ctx=item['context'][ii];base=model(s,act,ctx)
            for row in metrics['equivariance']:
                k=row['rotation'];cc=rotate_context(ctx,k) if row['condition']=='joint' else ctx
                out=model(rotate_state(s,k),rotate_vec(act,k),cc);error=(out-rotate_motion(base,k)).abs()*s.new_tensor([31.5,31.5,3,3])
                assert abs(float(error.max())-row['max_physical_coordinate_error'])<1e-10
                if (kind=='joint_exact' and row['condition']=='joint') or (kind=='wrong_exact' and row['condition']=='state_only'):assert float(error.max())<1e-4
    return {'all_passed':True,'data_seed':data_seed,'mode':mode,'kind':kind,'seed':seed,'predictions_replayed':predictions,
        'episode_rows_checked':row_count,'aggregate_and_interval_checks':checks,
        'artifacts':{str((folder/p).relative_to(HERE)):r.sha(folder/p) for p in ['run.json','model.pt','optimizer.pt','evaluation.pt','rows.csv','metrics.json']},
        'initialization_and_sampling_match_original_protocol':True,'verifier_sha256':r.sha(__file__)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--available',action='store_true');args=parser.parse_args()
    c=freeze();torch.set_num_threads(c['threads']);sampling={seed:sampling_reference(c,seed,256*19) for seed in c['seeds']};results=[]
    for ds in c['data_seeds']:
        for mode in c['modes']:
            root=HERE/f'data/{NAME}/d{ds}/world';training=load_split(root,mode,'train');assert training['states_sequence'].shape[1]==20
            data={split:load_split(root,mode,split) for split in c['evaluation_splits'] if split!='force_ood' or mode in c['force_ood_modes']}
            for seed in c['seeds']:
                for kind in c['kinds']:
                    folder=HERE/f'runs/{NAME}/d{ds}/state/{mode}/{kind}_s{seed}'
                    if not (folder/'metrics.json').exists():
                        if args.available:continue
                        raise RuntimeError(f'Required training/evaluation not complete: {folder}')
                    receipt=HERE/f'reports/{NAME}/state_verification/d{ds}_{mode}_{kind}_s{seed}.json'
                    if receipt.exists():
                        result=json.loads(receipt.read_text());assert result['verifier_sha256']==r.sha(__file__)
                        for path,sha in result['artifacts'].items():assert r.sha(HERE/path)==sha,path
                    else:
                        result=verify_model(c,ds,mode,kind,seed,data,training,sampling[seed]);dump(receipt,result)
                    results.append(result);print('verified state repeat',ds,mode,kind,seed,flush=True)
    complete=len(results)==126
    report={'all_passed':complete,'all_available_checked_passed':True,'models_verified':len(results),'expected_models':126,
        'predictions_replayed':sum(x['predictions_replayed'] for x in results),
        'episode_rows_checked':sum(x['episode_rows_checked'] for x in results),'aggregate_and_interval_checks':sum(x['aggregate_and_interval_checks'] for x in results),
        'original_initialization_and_sampling_matched':True,'results':results,'verifier_sha256':r.sha(__file__)}
    dump(HERE/f'reports/{NAME}/state_verification_progress.json',report)
    if complete:
        summary=HERE/f'reports/{NAME}/state_summary.json';d=json.loads(summary.read_text());assert d['models']==126 and d['new_trainings']==126
        for x in d['results']:
            f=HERE/f"runs/{NAME}/d{x['data_seed']}/state/{x['mode']}/{x['kind']}_s{x['seed']}"
            assert x['run']==json.loads((f/'run.json').read_text()) and x['metrics']==json.loads((f/'metrics.json').read_text())
        report['summary_sha256']=r.sha(summary);dump(HERE/f'reports/{NAME}/state_verification.json',report)
    print('state repeat audit',len(results),'/126 complete',complete,flush=True)


if __name__=='__main__':main()
