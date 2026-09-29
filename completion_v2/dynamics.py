"""Paired one-step/multistep and bounded-feedback transition experiments."""
from pathlib import Path
import sys,json,argparse,time
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
import torch.nn.functional as F
from p0 import dump,sha
from dynamics_models import Transition,load_split
BASE=HERE/'dynamics';MODES=['isotropic','fixed_gravity','variable_force']

def freeze():
    p=BASE/'protocol.json'
    cfg={'source_sha256':sha(Path(__file__)),'development_data':'followup/data/dynamics_world_v1',
         'confirmation_data_seeds':[882201,882202,882203],'initializations':[0,1,2],
         'training_variants':['one','multi','multi_bounded'],'inference_controls':['one_bounded','multi_projected'],
         'steps':4000,'batch_episodes':32,'multistep_horizon':8,'optimizer':'AdamW .001, weight_decay .0001, grad clip 1',
         'input':'Known states, per-object action and net force/drag. Pooled three environments. RGB initial estimates evaluated separately.',
         'bounds':'Position clipped to physical walls given object radius; velocity norm capped at 1.5 times maximum training velocity norm. Inactive slots zero. No test statistics.',
         'loss':'Mean squared normalized xy/velocity over active slots and rollout steps; no teacher forcing within multistep segment.',
         'evaluation_horizons':[1,16,61],'metrics':['position MAE','velocity MAE','nonfinite path rate','finite blow-up rate','wall penetration','target and nontarget counterfactual errors'],
         'comparison':'Same updates/initialization/sample start indices, different compute per update; both time and step counts reported.',
         'selection':'Validation only; minimum mean 16-step position error among multi and multi_bounded. Report every arm; no test reselection.',
         'dependencies':{'dynamics_models.py':sha(ROOT/'followup/dynamics_models.py'),'dynamics_world.py':sha(ROOT/'followup/dynamics_world.py')}}
    if p.exists():assert json.loads(p.read_text())==cfg
    else:dump(p,cfg)
    return cfg

def data_root(data_seed):
    return ROOT/'followup/data/dynamics_world_v1' if data_seed==640031 else BASE/f'data/d{data_seed}'

def bound(motion,state,limit):
    walls=1-state[...,4:5]*8/31.5
    xy=torch.maximum(torch.minimum(motion[...,:2],walls),-walls)
    v=motion[...,2:];norm=v.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    v=v*(limit/norm).clamp(max=1)
    return torch.cat([xy,v],-1)*state[...,-1:]

def advance(model,state,action,context,limit=None):
    motion=model(state,action,context)
    if limit is not None:motion=bound(motion,state,limit)
    return torch.cat([motion,state[...,4:]],-1)

def training_arrays(seed):
    groups=[load_split(data_root(seed),mode,'train') for mode in MODES]
    return (torch.cat([v['states_sequence'] for v in groups]),torch.cat([v['actions_sequence'] for v in groups]),
            torch.cat([v['contexts_sequence'] for v in groups]))

def train(data_seed,init,kind):
    freeze();folder=BASE/f'runs/d{data_seed}_s{init}/{kind}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():
        rr=json.loads((folder/'run.json').read_text());assert rr['checkpoint_sha256']==sha(folder/'model.pt');return
    states,actions,contexts=training_arrays(data_seed)
    limit=float(states[...,2:4].norm(dim=-1).max())*1.5
    torch.manual_seed(882000+init);model=Transition('joint_exact')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(882100+init);horizon=1 if kind=='one' else 8
    progress=folder/'progress.pt';start_step=0;prior=0.;logs=[]
    if progress.exists():
        ck=torch.load(progress,weights_only=True);assert ck['source_sha256']==sha(Path(__file__))
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);rng.set_state(ck['rng'])
        start_step=ck['step'];prior=ck['seconds'];logs=ck['logs']
    start=time.perf_counter()
    for step in range(start_step+1,4001):
        episode=torch.randint(len(states),(32,),generator=rng)
        # Shared start support across arms makes the one-step samples paired.
        t=torch.randint(states.shape[1]-8,(32,),generator=rng)
        current=states[episode,t];loss=0.
        for k in range(horizon):
            current=advance(model,current,actions[episode,t+k],contexts[episode],limit if kind=='multi_bounded' else None)
            target=states[episode,t+k+1,:,:4];mask=current[...,-1:]
            loss=loss+((current[...,:4]-target).square()*mask).sum()/(4*mask.sum())/horizon
        if not torch.isfinite(loss):
            dump(folder/'failure.json',{'step':step,'reason':'nonfinite training loss','source_sha256':sha(Path(__file__))})
            raise FloatingPointError('Retained nonfinite training failure')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        assert torch.isfinite(norm);optimizer.step()
        if step%250==0:
            seconds=prior+time.perf_counter()-start;logs.append({'step':step,'loss':float(loss.detach()),'seconds':seconds})
            ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.get_state(),'step':step,'seconds':seconds,'logs':logs,'source_sha256':sha(Path(__file__))}
            tmp=progress.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(progress)
            print('dynamics',data_seed,init,kind,logs[-1],flush=True)
    torch.save({'state_dict':model.state_dict(),'velocity_limit':limit,'data_seed':data_seed,'init':init,'kind':kind},folder/'model.pt')
    dump(folder/'run.json',{'checkpoint_sha256':sha(folder/'model.pt'),'source_sha256':sha(Path(__file__)),
                          'data_hashes':{mode:sha(data_root(data_seed)/mode/'train/trajectories.npz') for mode in MODES},
                          'steps':4000,'seconds':prior+time.perf_counter()-start,'logs':logs,'velocity_limit':limit})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['one','multi','multi_bounded'],required=True)
    p.add_argument('--seed',type=int,default=640031);p.add_argument('--init',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(2);train(a.seed,a.init,a.kind)
