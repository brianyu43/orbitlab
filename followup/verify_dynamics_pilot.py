"""Replay all state-oracle pilot predictions and data/weight/sample provenance."""
import csv
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_models import Transition,load_split,inertial_next


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/dynamics_pilot_v1.json';c=json.loads(cp.read_text())
    assert c['model_code_sha256']==r.sha(HERE/'dynamics_models.py')
    assert c['runner_code_sha256']==r.sha(HERE/'dynamics_pilot.py')
    assert c['data_manifest_sha256']==r.sha(HERE/'data/dynamics_world_v1/manifest.json')
    checks=[];aggregates=predictions=0;contact_rates={}
    for mode in c['modes']:
        data=load_split(HERE/'data/dynamics_world_v1',mode,'val');train=load_split(HERE/'data/dynamics_world_v1',mode,'train')
        assert train['states_sequence'].shape[1]==20 and data['states_sequence'].shape[1]==65
        rng=torch.Generator().manual_seed(765100);contact=torch.nonzero(train['events'].sum(-1)>0).flatten()
        sh=hashlib.sha256();ph=hashlib.sha256()
        for _ in range(c['steps']):
            idx=torch.cat([torch.randint(len(train['state']),(c['batch']//2,),generator=rng),
                contact[torch.randint(len(contact),(c['batch']//2,),generator=rng)]])
            order=torch.rand(c['batch'],4,generator=rng).argsort(-1);sh.update(idx.numpy().tobytes());ph.update(order.numpy().tobytes())
        contact_rates[mode]={'train_natural_fraction':len(contact)/len(train['state']),
            'expected_sampled_contact_fraction':.5+.5*len(contact)/len(train['state']),
            'val_natural_fraction':float((data['events'].sum(-1)>0).float().mean())}
        for kind in c['kinds']:
            folder=HERE/f'runs/dynamics_pilot_v1/{mode}/{kind}_s0';run=json.loads((folder/'run.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
            assert run['config_sha256']==r.sha(cp) and run['steps']==c['steps']
            assert run['sample_sha256']==sh.hexdigest() and run['permutation_sha256']==ph.hexdigest()
            assert run['training_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz')
            assert metrics['validation_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/val/trajectories.npz')
            assert metrics['checkpoint_sha256']==run['checkpoint_sha256']==r.sha(folder/'model.pt')
            assert metrics['evaluation_sha256']==r.sha(folder/'evaluation.pt')
            torch.manual_seed(765000);model=Transition(kind,c['hidden']);assert state_digest(model)==run['initial_weights_sha256']
            ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
            assert sum(p.numel() for p in model.parameters())==run['parameters']
            saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True);out=[]
            for i in range(0,len(data['state']),128):out.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
            torch.testing.assert_close(torch.cat(out),saved['prediction'],atol=0,rtol=0)
            torch.testing.assert_close(inertial_next(data['state'],data['action']),saved['inertial'],atol=0,rtol=0)
            rows=list(csv.DictReader((folder/'val_rows.csv').open()))
            for name,key in [('learned','prediction'),('inertial','inertial')]:
                p=saved[key].numpy();target=data['next'].numpy();live=data['state'][...,-1].numpy()
                absolute=np.abs(p-target)*np.array([31.5,31.5,3,3])
                pos=(absolute[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1));vel=(absolute[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1))
                for row in [v for v in rows if v['baseline']==name]:
                    select=data['episode'].numpy()==int(row['episode']);events=data['events'].numpy().sum(-1)>0
                    if row['category']=='contact':select&=events
                    if row['category']=='no_contact':select&=~events
                    assert select.sum()==int(row['transitions'])
                    assert abs(pos[select].mean()-float(row['position_mae_pixels']))<1e-6
                    assert abs(vel[select].mean()-float(row['velocity_mae_pixels_per_frame']))<1e-6
                for cat,m in metrics['summary'][name].items():
                    selected=[v for v in rows if v['baseline']==name and v['category']==cat]
                    assert len(selected)==m['episodes']
                    for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                        assert abs(np.mean([float(v[key]) for v in selected])-m[key]['mean'])<1e-8;aggregates+=1
            predictions+=len(data['state']);checks.append({'mode':mode,'kind':kind,'transitions_replayed':len(data['state'])})
            print('verified',mode,kind,flush=True)
    report={'all_passed':True,'checkpoints_verified':len(checks),'predicted_transitions_replayed':predictions,
        'metric_aggregate_checks':aggregates,'results':checks,'contact_sampling':contact_rates,
        'weights_samples_permutations_reproduced':True,'verifier_sha256':r.sha(__file__),
        'claim_boundary':'Validation-only one-step teacher-forced predictions from true states; not a learned long rollout or image-based result.'}
    dump(HERE/'reports/dynamics_pilot_v1/verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
