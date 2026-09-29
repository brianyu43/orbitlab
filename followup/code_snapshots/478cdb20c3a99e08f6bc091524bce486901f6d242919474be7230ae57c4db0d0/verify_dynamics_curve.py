"""Validate all curve checkpoints, their RNG/optimizer states and saved metrics."""
import csv
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_models import Transition,load_split,inertial_next


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/dynamics_curve_v1.json';c=json.loads(cp.read_text())
    assert c['runner_sha256']==r.sha(HERE/'dynamics_curve.py')
    assert c['model_code_sha256']==r.sha(HERE/'dynamics_models.py')
    assert c['evaluator_sha256']==r.sha(HERE/'dynamics_pilot.py')
    assert c['data_manifest_sha256']==r.sha(HERE/'data/dynamics_world_v1/manifest.json')
    results=[];metrics_checked=0
    for mode in c['modes']:
        data=load_split(HERE/'data/dynamics_world_v1',mode,'val');train=load_split(HERE/'data/dynamics_world_v1',mode,'train')
        rg=torch.Generator().manual_seed(765100);sample=hashlib.sha256();permutation=hashlib.sha256();expected={}
        for step in range(1,max(c['checkpoints'])+1):
            idx=torch.cat([torch.randint(len(train['state']),(32,),generator=rg),torch.randint(len(train['state']),(32,),generator=rg)])
            order=torch.rand(64,4,generator=rg).argsort(-1);sample.update(idx.numpy().tobytes());permutation.update(order.numpy().tobytes())
            if step in c['checkpoints']:expected[step]=(sample.hexdigest(),permutation.hexdigest(),rg.get_state().clone())
        for kind in c['kinds']:
            for step in c['checkpoints']:
                folder=HERE/f'runs/dynamics_curve_v1/{mode}/{kind}_s0_step{step}';run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
                assert run['steps']==step and run['config_sha256']==r.sha(cp)
                assert run['sample_sha256']==expected[step][0] and run['permutation_sha256']==expected[step][1]
                opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
                assert torch.equal(opt['sampling_rng'],expected[step][2])
                assert all(int(v['step'])==step for v in opt['optimizer']['state'].values())
                assert r.sha(folder/'model.pt')==run['checkpoint_sha256']==m['checkpoint_sha256']
                assert r.sha(folder/'evaluation.pt')==m['evaluation_sha256']
                assert run['training_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz')
                assert m['validation_archive_sha256']==r.sha(HERE/f'data/dynamics_world_v1/{mode}/val/trajectories.npz')
                torch.manual_seed(765000);model=Transition(kind,c['hidden']);assert state_digest(model)==run['initial_weights_sha256']
                ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
                if step==1000:
                    parent=torch.load(HERE/f'runs/dynamics_sampling_v1/{mode}/{kind}_s0/model.pt',map_location='cpu',weights_only=True)
                    for key,value in ck['state_dict'].items():torch.testing.assert_close(value,parent['state_dict'][key],atol=0,rtol=0)
                saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True);pred=[]
                for i in range(0,len(data['state']),128):pred.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
                torch.testing.assert_close(torch.cat(pred),saved['prediction'],atol=0,rtol=0)
                torch.testing.assert_close(inertial_next(data['state'],data['action']),saved['inertial'],atol=0,rtol=0)
                records=list(csv.DictReader((folder/'val_rows.csv').open()));live=data['state'][...,-1].numpy()
                for name,key in [('learned','prediction'),('inertial','inertial')]:
                    err=np.abs(saved[key].numpy()-data['next'].numpy())*np.array([31.5,31.5,3,3])
                    pos=(err[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1));vel=(err[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1))
                    for row in [v for v in records if v['baseline']==name]:
                        sel=data['episode'].numpy()==int(row['episode']);contact=data['events'].numpy().sum(-1)>0
                        if row['category']=='contact':sel&=contact
                        if row['category']=='no_contact':sel&=~contact
                        assert abs(pos[sel].mean()-float(row['position_mae_pixels']))<1e-6
                        assert abs(vel[sel].mean()-float(row['velocity_mae_pixels_per_frame']))<1e-6
                    for category,stats in m['summary'][name].items():
                        selected=[v for v in records if v['baseline']==name and v['category']==category]
                        for metric in ['position_mae_pixels','velocity_mae_pixels_per_frame']:
                            assert abs(np.mean([float(v[metric]) for v in selected])-stats[metric]['mean'])<1e-8;metrics_checked+=1
                subset=torch.randperm(len(train['state']),generator=torch.Generator().manual_seed(765300))[:512];losses=[]
                for batch in subset.split(128):
                    state=train['state'][batch];p=model(state,train['action'][batch],train['context'][batch]);mask=state[...,-1:]
                    losses.append(((p-train['next'][batch]).square()*mask).sum()/(4*mask.sum()))
                assert abs(float(torch.stack(losses).mean())-run['train_subset_normalized_mse'])<1e-10
                results.append({'mode':mode,'kind':kind,'steps':step});print('verified',mode,kind,step,flush=True)
    dump(HERE/'reports/dynamics_curve_v1/verification.json',{'all_passed':True,'checkpoints_verified':len(results),
        'validation_transitions_replayed':len(results)*4096,'aggregate_checks':metrics_checked,
        'one_thousand_step_parent_weights_exactly_replayed':True,'optimizer_and_rng_checked':True,
        'results':results,'verifier_sha256':r.sha(__file__),'claim_boundary':c['selection']})


if __name__=='__main__':main()
