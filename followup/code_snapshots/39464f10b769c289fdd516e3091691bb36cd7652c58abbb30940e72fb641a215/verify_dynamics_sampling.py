import csv
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_models import Transition,load_split,inertial_next


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/dynamics_sampling_v1.json';c=json.loads(cp.read_text());results=[]
    assert c['code_sha256']==r.sha(HERE/'dynamics_sampling_study.py')
    assert c['model_code_sha256']==r.sha(HERE/'dynamics_models.py')
    assert c['evaluator_code_sha256']==r.sha(HERE/'dynamics_pilot.py')
    assert c['parent_config_sha256']==r.sha(HERE/'configs/dynamics_pilot_v1.json')
    for mode in c['modes']:
        data=load_split(HERE/'data/dynamics_world_v1',mode,'val');train=load_split(HERE/'data/dynamics_world_v1',mode,'train')
        generator=torch.Generator().manual_seed(765100);other=torch.Generator().manual_seed(765100)
        contact=torch.nonzero(train['events'].sum(-1)>0).flatten();sample=hashlib.sha256();perm=hashlib.sha256()
        for _ in range(c['steps']):
            first=torch.randint(len(train['state']),(32,),generator=generator)
            assert torch.equal(first,torch.randint(len(train['state']),(32,),generator=other))
            second=torch.randint(len(train['state']),(32,),generator=generator)
            torch.randint(len(contact),(32,),generator=other)
            p=torch.rand(64,4,generator=generator).argsort(-1)
            assert torch.equal(p,torch.rand(64,4,generator=other).argsort(-1))
            sample.update(torch.cat([first,second]).numpy().tobytes());perm.update(p.numpy().tobytes())
        for kind in c['kinds']:
            folder=HERE/f'runs/dynamics_sampling_v1/{mode}/{kind}_s0'
            run=json.loads((folder/'run.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
            parent=json.loads((HERE/f'runs/dynamics_pilot_v1/{mode}/{kind}_s0/run.json').read_text())
            for key in ['steps','parameters','initial_weights_sha256','permutation_sha256','training_archive_sha256']:assert run[key]==parent[key]
            assert run['sample_sha256']==sample.hexdigest() and run['permutation_sha256']==perm.hexdigest()
            assert run['config_sha256']==r.sha(cp)
            assert run['checkpoint_sha256']==metrics['checkpoint_sha256']==r.sha(folder/'model.pt')
            assert metrics['evaluation_sha256']==r.sha(folder/'evaluation.pt')
            assert metrics['validation_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/val/trajectories.npz')
            assert run['training_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz')
            torch.manual_seed(765000);model=Transition(kind,c['hidden']);assert state_digest(model)==run['initial_weights_sha256']
            ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
            saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True);pred=[]
            for i in range(0,len(data['state']),128):pred.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
            torch.testing.assert_close(torch.cat(pred),saved['prediction'],atol=0,rtol=0)
            torch.testing.assert_close(inertial_next(data['state'],data['action']),saved['inertial'],atol=0,rtol=0)
            rows=list(csv.DictReader((folder/'val_rows.csv').open()))
            for name,key in [('learned','prediction'),('inertial','inertial')]:
                error=np.abs(saved[key].numpy()-data['next'].numpy())*np.array([31.5,31.5,3.,3.]);live=data['state'][...,-1].numpy()
                pos=(error[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1));vel=(error[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1))
                for row in [v for v in rows if v['baseline']==name]:
                    select=data['episode'].numpy()==int(row['episode']);events=data['events'].numpy().sum(-1)>0
                    if row['category']=='contact':select&=events
                    if row['category']=='no_contact':select&=~events
                    assert abs(pos[select].mean()-float(row['position_mae_pixels']))<1e-6
                    assert abs(vel[select].mean()-float(row['velocity_mae_pixels_per_frame']))<1e-6
                for category,stat in metrics['summary'][name].items():
                    selected=[v for v in rows if v['baseline']==name and v['category']==category]
                    for metric in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                        assert abs(np.mean([float(v[metric]) for v in selected])-stat[metric]['mean'])<1e-8
            results.append({'mode':mode,'kind':kind,'predictions':len(data['state'])})
    dump(HERE/'reports/dynamics_sampling_v1/verification.json',{'all_passed':True,'checkpoints':9,
        'predictions_replayed':sum(v['predictions'] for v in results),'first_half_samples_and_all_slot_permutations_match_parent':True,
        'results':results,'verifier_sha256':r.sha(__file__),'claim_boundary':c['selection']})


if __name__=='__main__':main()
