"""Fixed-budget training; pilot timings are separate from reported final models."""
import argparse,json,time,os,resource
import numpy as np
import torch
import torch.nn.functional as F
import r2_decoder_core as c
import r2_world as w
from v3_common import dump,sha,event

def load_training(kind):
    source=w.BASE/f'data/d{w.DEVELOPMENT}/train/n2';records=json.loads((source/'labels.json').read_text())
    if kind=='patch':
        states=c.states([r['objects'] for r in records])[:,:2];attributes=states[...,:15].reshape(-1,15);targets=[]
        for row in records:
            for v in row['objects']:
                a=c.r1.alpha(v['shape'],v['radius'],v['pose'],'symmetric4_lanczos');rgb=np.array(v['rgb'],np.float32)/255
                targets.append(np.concatenate([rgb[:,None,None]*a[None],a[None]],0))
        return {'attributes':attributes,'targets':torch.from_numpy(np.stack(targets))}
    root=c.BASE/f'data/d{w.DEVELOPMENT}/train/n2';archive=np.load(root/'targets.npz')
    commands=json.loads((root/'commands.json').read_text());scenes=[row['objects'] for row in records for _ in c.OPS]
    changed=[c.edit(objects,cmd)[0] for objects,cmd in zip(scenes,commands)]
    cp=c.BASE/'runs/patch/model.pt';model=c.PatchDecoder();model.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);model.eval()
    cache=root/'learned_foreground_cache.npz';receipt=root/'learned_foreground_cache.json'
    if receipt.exists():
        rr=json.loads(receipt.read_text());assert rr['patch_sha256']==sha(cp) and rr['cache_sha256']==sha(cache)
    else:
        current=[];edited=[]
        with torch.no_grad():
            for i in range(0,len(scenes),32):
                current.append(c.render(model,c.states(scenes[i:i+32])).numpy().astype(np.float16));edited.append(c.render(model,c.states(changed[i:i+32])).numpy().astype(np.float16))
        np.savez_compressed(cache,current=np.concatenate(current),edited=np.concatenate(edited));dump(receipt,{'patch_sha256':sha(cp),'cache_sha256':sha(cache)})
    rendered=np.load(cache);source_images=torch.from_numpy(np.load(source/'rgb.npz')['images'][:,0]).permute(0,3,1,2)
    return {'source':source_images.repeat_interleave(4,0),'targets':torch.from_numpy(archive['targets']).permute(0,3,1,2),'current':torch.from_numpy(rendered['current']),'edited':torch.from_numpy(rendered['edited']),
            'panel':torch.from_numpy(archive['panels'])[:,None],'foreground':torch.from_numpy(archive['foreground'])[:,None],'commands':commands}

def loss_for(kind,model,d,idx):
    if kind=='patch':
        y=d['targets'][idx];p=model(d['attributes'][idx]);weights=1+8*y[:,3:]
        mse=((p[:,:3]-y[:,:3]).square()*weights).sum()/(3*weights.sum())
        bce=(F.binary_cross_entropy(p[:,3:].clamp(1e-6,1-1e-6),y[:,3:],reduction='none')*weights).sum()/weights.sum()
        return 4*mse+bce
    source=d['source'][idx].float()/255;target=d['targets'][idx].float()/255
    changed=(source!=target).any(1,keepdim=True).float();panel=d['panel'][idx].float()/255;fg=d['foreground'][idx].float()/255
    pred,gate=model(source,d['current'][idx].float(),d['edited'][idx].float(),c.command_maps([d['commands'][i] for i in idx.tolist()]))
    weights=1+12*changed+4*fg+4*panel
    return ((pred-target).square()*weights).sum()/(3*weights.sum())+.1*(F.binary_cross_entropy_with_logits(gate,changed,reduction='none')*(1+8*changed)).mean()

def train(kind,benchmark=False):
    cfg=c.freeze();torch.set_num_threads(2);folder=c.BASE/('benchmark' if benchmark else 'runs')/kind;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run.json').exists():
        r=json.loads((folder/'run.json').read_text())
        if not benchmark:assert r['checkpoint_sha256']==sha(folder/'model.pt')
        return r
    d=load_training(kind);steps=100 if benchmark else cfg[kind]['steps'];torch.manual_seed(887701);model=c.PatchDecoder() if kind=='patch' else c.ImageEditor()
    opt=torch.optim.AdamW(model.parameters(),lr=.001 if kind=='patch' else .0005,weight_decay=.0001);rng=torch.Generator().manual_seed(887702)
    progress=folder/'progress.pt';start=0;prior=0.;logs=[]
    if progress.exists():
        p=torch.load(progress,weights_only=True);assert p['protocol_sha256']==sha(c.BASE/'protocol.json')
        model.load_state_dict(p['model']);opt.load_state_dict(p['optimizer']);rng.set_state(p['rng']);start=p['step'];prior=p['seconds'];logs=p['logs']
    begin=time.perf_counter()
    for step in range(start+1,steps+1):
        idx=torch.randint(len(d['targets']),(cfg[kind]['batch'],),generator=rng);loss=loss_for(kind,model,d,idx);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if step%250==0 or step==steps:
            sec=prior+time.perf_counter()-begin;logs.append({'step':step,'loss':float(loss.detach()),'seconds':sec})
            tmp=progress.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng.get_state(),'step':step,'seconds':sec,'logs':logs,'protocol_sha256':sha(c.BASE/'protocol.json')},tmp);tmp.replace(progress)
            dump(folder/'progress.json',{'pid':os.getpid(),'step':step,'total':steps,'seconds':sec});print('R2 decoder',kind,logs[-1],flush=True)
    r={'kind':kind,'steps':steps,'seconds':prior+time.perf_counter()-begin,'parameters':sum(p.numel() for p in model.parameters()),'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'logs':logs,'protocol_sha256':sha(c.BASE/'protocol.json'),'training_manifest_sha256':sha(w.BASE/f'data/d{w.DEVELOPMENT}/train/n2/manifest.json')}
    if kind=='editor':r['target_manifest_sha256']=sha(c.BASE/f'data/d{w.DEVELOPMENT}/train/n2/manifest.json');r['patch_checkpoint_sha256']=sha(c.BASE/'runs/patch/model.pt')
    if benchmark:r['estimated_full_seconds']=r['seconds']*cfg[kind]['steps']/100
    else:
        torch.save({'state_dict':model.state_dict(),'kind':kind},folder/'model.pt');r['checkpoint_sha256']=sha(folder/'model.pt')
    dump(folder/'run.json',r);event('R2_decoder_training_complete',kind=kind,benchmark=benchmark);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['patch','editor'],required=True);p.add_argument('--benchmark',action='store_true');a=p.parse_args();train(a.kind,a.benchmark)
