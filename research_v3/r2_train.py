"""Resumable supervised R2 training with matched data and current-label budgets."""
import argparse,json,time,resource,os
import numpy as np
import torch
import r2_core as c
import r2_world as w
from v3_common import dump,sha,event

def train(seed,init,arm,benchmark=False):
    cfg=c.freeze();torch.set_num_threads(2);folder=w.BASE/('benchmark' if benchmark else 'runs')/f'd{seed}_s{init}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():
        rr=json.loads((folder/'run.json').read_text())
        if not benchmark:assert rr['checkpoint_sha256']==sha(folder/'model.pt')
        return rr
    data=w.BASE/f'data/d{seed}/train/n2'
    x=torch.from_numpy(np.load(data/'rgb.npz')['images']).permute(0,1,4,2,3)
    masks=torch.from_numpy(np.load(data/'masks.npz')['visible']).float()/255
    y=c.labels_tensor(json.loads((data/'labels.json').read_text()))
    torch.manual_seed(887000+init);model=c.model_for(arm);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001);rng=torch.Generator().manual_seed(887100+init)
    progress=folder/'progress.pt';start,prior,logs=0,0.,[];steps=100 if benchmark else cfg['steps']
    if progress.exists():
        ck=torch.load(progress,weights_only=True);assert ck['protocol_sha256']==sha(w.BASE/'training_protocol.json')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);rng.set_state(ck['rng']);start,prior,logs=ck['step'],ck['seconds'],ck['logs']
    begin=time.perf_counter()
    for step in range(start+1,steps+1):
        idx=torch.randint(len(x),(cfg['batch'],),generator=rng)
        pred=model(x[idx].float()/255);loss=c.training_loss(pred,y[idx],masks[idx],arm)
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if step%250==0 or step==steps:
            sec=prior+time.perf_counter()-begin;logs.append({'step':step,'loss':float(loss.detach()),'seconds':sec})
            tmp=progress.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng.get_state(),'step':step,'seconds':sec,'logs':logs,'protocol_sha256':sha(w.BASE/'training_protocol.json')},tmp);tmp.replace(progress)
            dump(folder/'progress.json',{'pid':os.getpid(),'step':step,'total':steps,'seconds':sec});print('R2 train',seed,init,arm,logs[-1],flush=True)
    result={'arm':arm,'steps':steps,'seconds':prior+time.perf_counter()-begin,'logs':logs,'parameters':sum(p.numel() for p in model.parameters()),'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'protocol_sha256':sha(w.BASE/'training_protocol.json'),'data_sha256':sha(data/'rgb.npz')}
    if not benchmark:
        torch.save({'state_dict':model.state_dict(),'seed':seed,'init':init,'arm':arm},folder/'model.pt');result['checkpoint_sha256']=sha(folder/'model.pt')
    else:result['estimated_seconds_per_4000_steps']=result['seconds']*40
    dump(folder/'run.json',result);event('R2_training_complete',seed=seed,init=init,arm=arm,benchmark=benchmark);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=w.DEVELOPMENT);p.add_argument('--init',type=int,default=0);p.add_argument('--arm',choices=c.ARMS,required=True);p.add_argument('--benchmark',action='store_true');a=p.parse_args();train(a.seed,a.init,a.arm,a.benchmark)
