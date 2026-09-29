"""Full held-out prediction replay and matching audit for the state study."""
import csv
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_models import Transition,load_split,rotate_state,rotate_vec,rotate_context,rotate_motion


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/dynamics_state_study_v1.json';c=json.loads(cp.read_text())
    root=HERE/'reports/dynamics_state_study_v1';summary=json.loads((root/'summary.json').read_text())
    assert summary['config_sha256']==r.sha(cp) and summary['models']==63 and summary['new_trainings']==54 and summary['reused_models']==9
    assert c['runner_sha256']==r.sha(HERE/'dynamics_state_study.py')
    assert c['model_sha256']==r.sha(HERE/'dynamics_models.py')
    assert c['data_manifest_sha256']==r.sha(HERE/'data/dynamics_world_v1/manifest.json')
    results=[];total_predictions=metric_checks=0
    for mode in c['modes']:
        train=load_split(HERE/'data/dynamics_world_v1',mode,'train');assert train['states_sequence'].shape[1]==20
        split_data={split:load_split(HERE/'data/dynamics_world_v1',mode,split) for split in c['evaluation_splits'] if split!='force_ood' or mode in c['force_ood_modes']}
        for seed in c['seeds']:
            rng=torch.Generator().manual_seed(765100+seed);aug=torch.Generator().manual_seed(765200+seed)
            sh=hashlib.sha256();ph=hashlib.sha256()
            for _ in range(c['steps']):
                idx=torch.cat([torch.randint(len(train['state']),(32,),generator=rng),torch.randint(len(train['state']),(32,),generator=rng)])
                order=torch.rand(64,4,generator=rng).argsort(-1);sh.update(idx.numpy().tobytes());ph.update(order.numpy().tobytes());torch.randint(4,(),generator=aug)
            for kind in c['kinds']:
                folder=HERE/f'runs/dynamics_state_study_v1/{mode}/{kind}_s{seed}';run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
                assert run['steps']==c['steps'] and run['seed']==seed and run['config_sha256']==r.sha(cp)
                assert run['sample_sha256']==sh.hexdigest() and run['permutation_sha256']==ph.hexdigest()
                assert run['training_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz')
                assert run['checkpoint_sha256']==m['checkpoint_sha256']==r.sha(folder/'model.pt')
                assert m['evaluation_sha256']==r.sha(folder/'evaluation.pt')
                optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
                assert torch.equal(optimizer['sampling_rng'],rng.get_state())
                if kind=='augmented':assert torch.equal(optimizer['augmentation_rng'],aug.get_state())
                assert all(int(v['step'])==c['steps'] for v in optimizer['optimizer']['state'].values())
                if run['new_training_steps']==0:
                    parent=HERE/run['reused_from'];assert r.sha(parent/'model.pt')==run['checkpoint_sha256']
                    assert r.sha(parent/'run.json')==run['original_run_sha256']
                    assert r.sha(folder/'model.pt')==c['reused_seed_zero_checkpoints'][f'{mode}/{kind}']
                torch.manual_seed(765000+seed);model=Transition(kind,c['hidden']);assert state_digest(model)==run['initial_weights_sha256']
                ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
                assert sum(p.numel() for p in model.parameters())==run['parameters']
                saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True);records=list(csv.DictReader((folder/'rows.csv').open()))
                assert set(saved)==set(split_data)==set(m['splits'])
                for split,data in split_data.items():
                    assert m['data_sha256'][split]==r.sha(HERE/f'data/dynamics_world_v1/{mode}/{split}/trajectories.npz')
                    pred=[]
                    for i in range(0,len(data['state']),128):pred.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
                    torch.testing.assert_close(torch.cat(pred),saved[split],atol=0,rtol=0)
                    assert torch.isfinite(saved[split]).all();total_predictions+=len(data['state'])
                    live=data['state'][...,-1].numpy();err=np.abs(saved[split].numpy()-data['next'].numpy())*np.array([31.5,31.5,3,3])
                    pos=(err[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1));vel=(err[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1))
                    assert not np.any(saved[split].numpy()[live==0])
                    pair=data['events'][:,0].numpy()>0;wall=data['events'][:,1].numpy()>0
                    conditions={'all':np.ones(len(pair),bool),'no_contact':~(pair|wall),'contact':pair|wall,'pair':pair,'wall_only':wall&~pair}
                    for row in [v for v in records if v['split']==split]:
                        select=(data['episode'].numpy()==int(row['episode']))&conditions[row['category']]
                        assert int(select.sum())==int(row['transitions'])
                        np.testing.assert_allclose([pos[select].mean(),vel[select].mean()],
                            [float(row['position_mae_pixels']),float(row['velocity_mae_pixels_per_frame'])],atol=1e-6,rtol=1e-6)
                    for category,metrics in m['splits'][split].items():
                        selected=[v for v in records if v['split']==split and v['category']==category]
                        assert len(selected)==metrics['episodes']
                        for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                            expected=r.bootstrap([float(v[key]) for v in selected]);np.testing.assert_allclose(expected['base_scene_ci95'],metrics[key]['base_scene_ci95'],atol=1e-10,rtol=0)
                            assert abs(expected['mean']-metrics[key]['mean'])<1e-10;metric_checks+=1
                    if split=='test':
                        idx=torch.arange(32)*64+3;s=data['state'][idx];a=data['action'][idx];ctx=data['context'][idx];base=model(s,a,ctx)
                        for row in m['equivariance']:
                            k=row['rotation'];context=rotate_context(ctx,k) if row['condition']=='joint' else ctx
                            out=model(rotate_state(s,k),rotate_vec(a,k),context);error=(out-rotate_motion(base,k)).abs()*s.new_tensor([31.5,31.5,3,3])
                            assert abs(float(error.max())-row['max_physical_coordinate_error'])<1e-10
                            if (kind=='joint_exact' and row['condition']=='joint') or (kind=='wrong_exact' and row['condition']=='state_only'):
                                assert float(error.max())<1e-4
                results.append({'mode':mode,'kind':kind,'seed':seed});print('verified',mode,kind,seed,flush=True)
    dump(root/'verification.json',{'all_passed':True,'models_verified':len(results),'heldout_transition_predictions_replayed':total_predictions,
        'metric_aggregate_and_interval_checks':metric_checks,'optimizer_samples_initialization_and_reuse_verified':True,
        'results':results,'verifier_sha256':r.sha(__file__),'claim_boundary':c['scope']})


if __name__=='__main__':main()
