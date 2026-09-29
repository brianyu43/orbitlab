"""One-seed state-oracle feasibility study; validation only, no final C03 claim."""
import hashlib
import json
import time
import torch
import numpy as np
from common import HERE,dump,fresh_dir,state_digest,digest_tensor,r,update_status
from dynamics_models import Transition,load_split,permute_slots,rotate_state,rotate_vec,rotate_context,rotate_motion,inertial_next


def freeze():
    cp=HERE/'configs/dynamics_pilot_v1.json'
    if not cp.exists():dump(cp,{'modes':['isotropic','fixed_gravity','variable_force'],
        'kinds':['flat','object','interaction','augmented','wrong_exact','joint_exact','relaxed'],
        'seed':0,'steps':1000,'batch':64,'lr':.001,'hidden':48,'device':'cpu','threads':2,
        'data_manifest_sha256':r.sha(HERE/'data/dynamics_world_v1/manifest.json'),
        'model_code_sha256':r.sha(HERE/'dynamics_models.py'),'runner_code_sha256':r.sha(__file__),
        'sampling':'Half uniform train transitions, half train transitions with a pair/wall impulse. Equal draws and slot permutations across models.',
        'augmentation':'Augmented interaction network rotates state/action/context/target together; separate random stream.',
        'information':'Ground-truth states and known actions plus net acceleration/drag. Static radius/color/presence preserved in all models.',
        'loss':'Mean squared error in normalized motion: position /31.5, velocity /3, active objects only.',
        'budgets':'Same steps/batch/optimizer. Different routing parameter counts and time; symmetry variants share raw interaction architecture.',
        'evaluation':'Validation only. Per-episode aggregation of all, no-contact, and contact transitions. Final benchmark and multi-seed study pending.',
        'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def train(c,mode,kind):
    folder=HERE/f'runs/dynamics_pilot_v1/{mode}/{kind}_s0'
    if (folder/'run.json').exists():return folder
    fresh_dir(folder);data=load_split(HERE/'data/dynamics_world_v1',mode,'train')
    torch.manual_seed(765000);model=Transition(kind,c['hidden']);initial=state_digest(model)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(765100)
    aug=torch.Generator().manual_seed(765200);contact=torch.nonzero(data['events'].sum(-1)>0).flatten();assert len(contact)>0
    sample_hash=hashlib.sha256();permutation_hash=hashlib.sha256();logs=[];start=time.perf_counter()
    dump(folder/'started.json',{'config_sha256':r.sha(HERE/'configs/dynamics_pilot_v1.json'),'initial_weights_sha256':initial})
    for step in range(1,c['steps']+1):
        idx=torch.cat([torch.randint(len(data['state']),(c['batch']//2,),generator=rng),
            contact[torch.randint(len(contact),(c['batch']//2,),generator=rng)]])
        order=torch.rand(c['batch'],4,generator=rng).argsort(-1)
        sample_hash.update(idx.numpy().tobytes());permutation_hash.update(order.numpy().tobytes())
        s,a,t=permute_slots(data['state'][idx],data['action'][idx],data['next'][idx],order);ctx=data['context'][idx]
        if kind=='augmented':
            k=int(torch.randint(4,(),generator=aug));s=rotate_state(s,k);a=rotate_vec(a,k);ctx=rotate_context(ctx,k);t=rotate_motion(t,k)
        out=model(s,a,ctx);mask=s[...,-1:];loss=((out-t).square()*mask).sum()/(4*mask.sum())
        if not torch.isfinite(loss):raise FloatingPointError('Dynamics loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Dynamics gradient')
        opt.step()
        if step%250==0:logs.append({'step':step,'loss':float(loss.detach())})
    seconds=time.perf_counter()-start
    torch.save({'state_dict':model.state_dict(),'kind':kind,'hidden':c['hidden']},folder/'model.pt')
    dump(folder/'run.json',{'mode':mode,'kind':kind,'steps':c['steps'],'parameters':sum(v.numel() for v in model.parameters()),
        'train_seconds':seconds,'initial_weights_sha256':initial,'sample_sha256':sample_hash.hexdigest(),'permutation_sha256':permutation_hash.hexdigest(),
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/'configs/dynamics_pilot_v1.json'),
        'training_archive_sha256':r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz'),
        'contact_train_transitions':len(contact),'total_train_transitions':len(data['state']),'loss_log':logs})
    print('trained',mode,kind,round(seconds,2),logs[-1],flush=True);return folder


@torch.no_grad()
def evaluate(folder,c):
    if (folder/'metrics.json').exists():return
    run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval()
    data=load_split(HERE/'data/dynamics_world_v1',run['mode'],'val');predictions=[]
    for i in range(0,len(data['state']),128):predictions.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
    pred=torch.cat(predictions);base=inertial_next(data['state'],data['action']);rows=[]
    for name,p in [('learned',pred),('inertial',base)]:
        err=(p-data['next']).abs()*p.new_tensor([31.5,31.5,3.,3.]);mask=data['state'][...,-1]
        pos=(err[...,:2].sum(-1)*mask).sum(-1)/(2*mask.sum(-1));vel=(err[...,2:].sum(-1)*mask).sum(-1)/(2*mask.sum(-1))
        for epi in torch.unique(data['episode']).tolist():
            scene=data['episode']==epi;contact=data['events'].sum(-1)>0
            for category,select in [('all',scene),('no_contact',scene&~contact),('contact',scene&contact)]:
                if select.any():rows.append({'baseline':name,'episode':epi,'category':category,'transitions':int(select.sum()),
                    'position_mae_pixels':float(pos[select].mean()),'velocity_mae_pixels_per_frame':float(vel[select].mean())})
    summary={}
    for name in ['learned','inertial']:
        summary[name]={}
        for category in ['all','no_contact','contact']:
            selected=[v for v in rows if v['baseline']==name and v['category']==category]
            summary[name][category]={k:r.bootstrap([v[k] for v in selected]) for k in ['position_mae_pixels','velocity_mae_pixels_per_frame']}
            summary[name][category]['episodes']=len(selected)
    r.write_csv(folder/'val_rows.csv',rows);torch.save({'prediction':pred,'inertial':base},folder/'evaluation.pt')
    dump(folder/'metrics.json',{'summary':summary,'checkpoint_sha256':r.sha(folder/'model.pt'),'evaluation_sha256':r.sha(folder/'evaluation.pt'),
        'validation_archive_sha256':r.sha(HERE/f'data/dynamics_world_v1/{run["mode"]}/val/trajectories.npz'),
        'relaxed_raw_fraction':float(1-model.mix_logit.sigmoid()) if run['kind']=='relaxed' else None})
    print('validation',run['mode'],run['kind'],{cat:round(v['velocity_mae_pixels_per_frame']['mean'],4) for cat,v in summary['learned'].items()},flush=True)


def main():
    c=freeze();torch.set_num_threads(c['threads']);results=[]
    update_status('C03','in_progress',['configs/dynamics_pilot_v1.json'],note='One-seed validation-only oracle feasibility study; final multi-seed comparison pending.')
    for mode in c['modes']:
        runs=[]
        for kind in c['kinds']:
            folder=train(c,mode,kind);evaluate(folder,c);run=json.loads((folder/'run.json').read_text());runs.append(run)
            results.append({'mode':mode,'kind':kind,'run':run,'metrics':json.loads((folder/'metrics.json').read_text())})
            dump(HERE/'reports/dynamics_pilot_v1/progress.json',{'completed_cells':len(results),'expected_cells':21,'results':results})
        for key in ['sample_sha256','permutation_sha256']:assert len({v[key] for v in runs})==1
        controls=[v for v in runs if v['kind'] in ['interaction','augmented','wrong_exact','joint_exact']]
        assert len({v['initial_weights_sha256'] for v in controls})==1
    dump(HERE/'reports/dynamics_pilot_v1/summary.json',{'results':results,'trainings':21,'config_sha256':r.sha(HERE/'configs/dynamics_pilot_v1.json'),
        'claim_boundary':'Single initialization, validation-only one-step pilot. No final test, long rollout, image-based inference or independent repetition claims.'})


if __name__=='__main__':main()
