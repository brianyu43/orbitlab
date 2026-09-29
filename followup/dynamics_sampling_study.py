"""Paired diagnostic changing only the second half of transition sampling.

The original pilot selected 50% from contact events. Here every transition is
drawn uniformly; architecture, init, steps, batch, loss and slot RNG are held.
"""
import hashlib
import json
import time
import torch
from common import HERE,dump,fresh_dir,state_digest,r
from dynamics_models import Transition,load_split,permute_slots
from dynamics_pilot import evaluate


def freeze():
    cp=HERE/'configs/dynamics_sampling_v1.json'
    if not cp.exists():
        parent=json.loads((HERE/'configs/dynamics_pilot_v1.json').read_text())
        c={k:parent[k] for k in ['modes','seed','steps','batch','lr','hidden','device','threads','data_manifest_sha256','model_code_sha256']}
        c.update(kinds=['interaction','wrong_exact','joint_exact'],
            parent_config_sha256=r.sha(HERE/'configs/dynamics_pilot_v1.json'),
            code_sha256=r.sha(__file__),evaluator_code_sha256=r.sha(HERE/'dynamics_pilot.py'),
            change='The second half of every minibatch is drawn from all train transitions rather than the contact subset.',
            selection='Same 1000 steps, one initialization, validation-only cause diagnostic. No test or long-rollout claim.',frozen_unix=time.time())
        dump(cp,c)
    return json.loads(cp.read_text())


def train(c,mode,kind):
    folder=HERE/f'runs/dynamics_sampling_v1/{mode}/{kind}_s0';fresh_dir(folder)
    data=load_split(HERE/'data/dynamics_world_v1',mode,'train');torch.manual_seed(765000)
    model=Transition(kind,c['hidden']);initial=state_digest(model);opt=torch.optim.Adam(model.parameters(),lr=c['lr'])
    rng=torch.Generator().manual_seed(765100);sh=hashlib.sha256();ph=hashlib.sha256();logs=[];start=time.perf_counter()
    for step in range(1,c['steps']+1):
        idx=torch.cat([torch.randint(len(data['state']),(c['batch']//2,),generator=rng),
                       torch.randint(len(data['state']),(c['batch']//2,),generator=rng)])
        order=torch.rand(c['batch'],4,generator=rng).argsort(-1);sh.update(idx.numpy().tobytes());ph.update(order.numpy().tobytes())
        s,a,t=permute_slots(data['state'][idx],data['action'][idx],data['next'][idx],order)
        out=model(s,a,data['context'][idx]);mask=s[...,-1:];loss=((out-t).square()*mask).sum()/(4*mask.sum())
        if not torch.isfinite(loss):raise FloatingPointError('Uniform dynamics loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Uniform dynamics gradient')
        opt.step()
        if step%250==0:logs.append({'step':step,'loss':float(loss.detach())})
    torch.save({'state_dict':model.state_dict(),'kind':kind,'hidden':c['hidden']},folder/'model.pt')
    run={'mode':mode,'kind':kind,'steps':c['steps'],'parameters':sum(v.numel() for v in model.parameters()),
        'train_seconds':time.perf_counter()-start,'initial_weights_sha256':initial,'sample_sha256':sh.hexdigest(),'permutation_sha256':ph.hexdigest(),
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/'configs/dynamics_sampling_v1.json'),
        'training_archive_sha256':r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz'),'loss_log':logs}
    parent=json.loads((HERE/f'runs/dynamics_pilot_v1/{mode}/{kind}_s0/run.json').read_text())
    for key in ['steps','parameters','initial_weights_sha256','permutation_sha256','training_archive_sha256']:assert run[key]==parent[key]
    dump(folder/'run.json',run);print('trained uniform',mode,kind,flush=True);return folder


def main():
    c=freeze();torch.set_num_threads(c['threads']);results=[]
    for mode in c['modes']:
        for kind in c['kinds']:
            folder=train(c,mode,kind);evaluate(folder,c)
            uniform=json.loads((folder/'metrics.json').read_text());old=json.loads((HERE/f'runs/dynamics_pilot_v1/{mode}/{kind}_s0/metrics.json').read_text())
            results.append({'mode':mode,'kind':kind,'uniform':uniform,'contact_balanced':old})
    dump(HERE/'reports/dynamics_sampling_v1/summary.json',{'results':results,'trainings':9,'config_sha256':r.sha(HERE/'configs/dynamics_sampling_v1.json'),
        'claim_boundary':c['selection']})


if __name__=='__main__':main()
