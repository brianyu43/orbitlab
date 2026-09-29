"""Equal update pilot; no test trajectory or test context in training."""
import argparse,json,resource,time,os
import numpy as np
import torch
import r3_core as c
from v3_common import dump,sha,event

def train(seed,init,arm,benchmark=False):
    cfg=c.freeze();torch.set_num_threads(2)
    folder=c.BASE/('benchmark' if benchmark else 'runs')/f'd{seed}_s{init}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():
        rec=json.loads((folder/'run.json').read_text())
        if not benchmark:assert rec['checkpoint_sha256']==sha(folder/'model.pt')
        return rec
    path=c.BASE/f'data/d{seed}/train/n2/trajectories.npz';a=np.load(path)
    states=c.encode_states(a['states']);actions=torch.tensor(a['actions'],dtype=torch.float32)/3;contexts=c.encode_context(a['contexts'])
    limit=float(states[...,2:4].norm(dim=-1).max())*1.5
    torch.manual_seed(888000+init);model=c.model_for(arm);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(888100+init);horizon=1 if arm=='one' else 8;steps=100 if benchmark else cfg['training_steps']
    progress=folder/'progress.pt';start,prior,logs=0,0.,[]
    if progress.exists():
        ck=torch.load(progress,weights_only=True);assert ck['protocol_sha256']==sha(c.BASE/'protocol.json')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.set_state(ck['rng']);start,prior,logs=ck['step'],ck['seconds'],ck['logs']
    begin=time.perf_counter()
    for step in range(start+1,steps+1):
        ep=torch.randint(len(states),(32,),generator=rng);t=torch.randint(states.shape[1]-8,(32,),generator=rng)
        current=states[ep,t];loss=0
        for k in range(horizon):
            current=c.advance(model,current,actions[ep,t+k],contexts[ep],arm,limit)
            loss=loss+(current[...,:4]-states[ep,t+k+1,:,:4]).square().mean()/horizon
        if not torch.isfinite(loss):
            dump(folder/'failure.json',{'step':step,'reason':'nonfinite loss'});raise FloatingPointError('Nonfinite loss preserved')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        assert torch.isfinite(norm);opt.step()
        if step%250==0 or step==steps:
            seconds=prior+time.perf_counter()-begin;logs.append({'step':step,'loss':float(loss.detach()),'seconds':seconds})
            tmp=progress.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng.get_state(),'step':step,'seconds':seconds,'logs':logs,'protocol_sha256':sha(c.BASE/'protocol.json')},tmp);tmp.replace(progress)
            dump(folder/'progress.json',{'pid':os.getpid(),'step':step,'total':steps,'seconds':seconds})
            print('R3 train',seed,init,arm,logs[-1],flush=True)
    result={'steps':steps,'seconds':prior+time.perf_counter()-begin,'logs':logs,'parameters':sum(p.numel() for p in model.parameters()),'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'data_sha256':sha(path),'protocol_sha256':sha(c.BASE/'protocol.json'),'velocity_limit':limit}
    if not benchmark:
        torch.save({'state_dict':model.state_dict(),'velocity_limit':limit,'seed':seed,'init':init,'arm':arm},folder/'model.pt');result['checkpoint_sha256']=sha(folder/'model.pt')
    else:result['estimated_seconds_per_2000_steps']=result['seconds']*20
    dump(folder/'run.json',result);event('R3_training_complete',seed=seed,init=init,arm=arm,benchmark=benchmark);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=c.DEVELOPMENT);p.add_argument('--init',type=int,default=0);p.add_argument('--arm',choices=c.ARMS,required=True);p.add_argument('--benchmark',action='store_true');a=p.parse_args();train(a.seed,a.init,a.arm,a.benchmark)
