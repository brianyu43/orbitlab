"""Uniform-sampling learning curves, with saved optimizer and RNG checkpoints."""
import hashlib
import json
import time
import torch
from common import HERE,dump,fresh_dir,state_digest,r
from dynamics_models import Transition,load_split,permute_slots
from dynamics_pilot import evaluate


def freeze():
    cp=HERE/'configs/dynamics_curve_v1.json'
    if not cp.exists():
        p=json.loads((HERE/'configs/dynamics_sampling_v1.json').read_text())
        c={k:p[k] for k in ['modes','kinds','seed','batch','lr','hidden','threads','model_code_sha256','data_manifest_sha256']}
        c.update(checkpoints=[1000,6000,12000],parent_config_sha256=r.sha(HERE/'configs/dynamics_sampling_v1.json'),
            runner_sha256=r.sha(__file__),evaluator_sha256=r.sha(HERE/'dynamics_pilot.py'),
            training='Fresh deterministic replay from the same initial weights. Verify equality to the prior uniform pilot at 1k; continue without reset to 6k/12k.',
            selection='All fixed checkpoints, one initialization, train and validation only; no test selection.',frozen_unix=time.time())
        dump(cp,c)
    return json.loads(cp.read_text())


@torch.no_grad()
def train_diagnostic(model,data):
    idx=torch.randperm(len(data['state']),generator=torch.Generator().manual_seed(765300))[:512]
    errors=[]
    for batch in idx.split(128):
        s=data['state'][batch];out=model(s,data['action'][batch],data['context'][batch]);t=data['next'][batch]
        errors.append(((out-t).square()*s[...,-1:]).sum()/(4*s[...,-1:].sum()))
    return float(torch.stack(errors).mean())


def main():
    c=freeze();torch.set_num_threads(c['threads']);rows=[]
    for mode in c['modes']:
        data=load_split(HERE/'data/dynamics_world_v1',mode,'train')
        for kind in c['kinds']:
            torch.manual_seed(765000);model=Transition(kind,c['hidden']);initial=state_digest(model)
            opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(765100)
            sh=hashlib.sha256();ph=hashlib.sha256();logs=[];start=time.perf_counter();training_seconds=0.;last=0
            for step in range(1,max(c['checkpoints'])+1):
                idx=torch.cat([torch.randint(len(data['state']),(c['batch']//2,),generator=rng),
                    torch.randint(len(data['state']),(c['batch']//2,),generator=rng)])
                order=torch.rand(c['batch'],4,generator=rng).argsort(-1);sh.update(idx.numpy().tobytes());ph.update(order.numpy().tobytes())
                s,a,t=permute_slots(data['state'][idx],data['action'][idx],data['next'][idx],order)
                pred=model(s,a,data['context'][idx]);mask=s[...,-1:];loss=((pred-t).square()*mask).sum()/(4*mask.sum())
                if not torch.isfinite(loss):raise FloatingPointError('Curve loss')
                opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError('Curve gradient')
                opt.step()
                if step%1000==0:logs.append({'step':step,'loss':float(loss.detach())})
                if step in c['checkpoints']:
                    training_seconds+=time.perf_counter()-start;folder=fresh_dir(HERE/f'runs/dynamics_curve_v1/{mode}/{kind}_s0_step{step}')
                    if step==1000:
                        parent=HERE/f'runs/dynamics_sampling_v1/{mode}/{kind}_s0'
                        ck=torch.load(parent/'model.pt',map_location='cpu',weights_only=True)
                        for key,value in model.state_dict().items():torch.testing.assert_close(value,ck['state_dict'][key],atol=0,rtol=0)
                        previous=json.loads((parent/'run.json').read_text())
                        assert previous['sample_sha256']==sh.hexdigest() and previous['permutation_sha256']==ph.hexdigest()
                    torch.save({'kind':kind,'hidden':c['hidden'],'state_dict':model.state_dict()},folder/'model.pt')
                    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
                    run={'mode':mode,'kind':kind,'steps':step,'parameters':sum(p.numel() for p in model.parameters()),
                        'train_seconds':training_seconds,'initial_weights_sha256':initial,'sample_sha256':sh.hexdigest(),
                        'permutation_sha256':ph.hexdigest(),'checkpoint_sha256':r.sha(folder/'model.pt'),
                        'config_sha256':r.sha(HERE/'configs/dynamics_curve_v1.json'),'training_archive_sha256':r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz'),
                        'train_subset_normalized_mse':train_diagnostic(model,data),'loss_log':logs.copy()}
                    dump(folder/'run.json',run);evaluate(folder,c)
                    rows.append({'mode':mode,'kind':kind,'steps':step,'run':run,'metrics':json.loads((folder/'metrics.json').read_text())})
                    dump(HERE/'reports/dynamics_curve_v1/progress.json',{'completed_checkpoints':len(rows),'expected_checkpoints':27,'results':rows})
                    print('curve',mode,kind,step,'train MSE',run['train_subset_normalized_mse'],flush=True);start=time.perf_counter();last=step
    dump(HERE/'reports/dynamics_curve_v1/summary.json',{'results':rows,'models':9,'checkpoints':27,
        'training_steps_executed':9*12000,'config_sha256':r.sha(HERE/'configs/dynamics_curve_v1.json'),
        'claim_boundary':c['selection']})


if __name__=='__main__':main()
