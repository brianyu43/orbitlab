"""New bounded generation study; historical trainers are reused read-only."""
from pathlib import Path
import argparse, json, sys, time, hashlib
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
import torch.nn.functional as F
import orbitlab as o
import research as r
import p1_train as p1
import p4_prepare as prep
from p4_train import correct_checkpoint
from p0 import sha,dump,THRESHOLD
from models import from_variant
from decoder_study import ControlledDecoder

BASE=HERE/'generation'
DATA=BASE/'data'
RUNS=BASE/'runs'
LOCK=BASE/'protocol.json'
SEEDS=[880101,880201,880202,880203]

def freeze():
    cfg={'development_seed':880101,'confirmation_seeds':SEEDS[1:],'initializations':[0,1,2],
         'ae_steps':6000,'decoder_steps':6000,'flow_steps':16000,'batch_decoder':32,'batch_flow':128,
         'conditions':['uniform_flow/standard_decoder','stratified_flow/standard_decoder',
                       'uniform_flow/area_edge_decoder','stratified_flow/area_edge_decoder'],
         'strata':'shape,color,true renderer radius thirds; sample each nonempty stratum equally',
         'decoder_intervention':'Per-image foreground MSE + 0.1 background MSE + 0.25 foreground-normalized gradient MSE; same samples and initialization as standard.',
         'generation_steps':64,'guidance':1.0,'samples_per_pair':128,'reference_per_pair':384,
         'threshold_sha256':sha(THRESHOLD),'selection':'Development only: require all-image and strict-subset diversity; maximize mean seen/OOD strict pass. If no candidate passes, retain best strict pass as diagnostic-only and report failed milestone; never relax thresholds.',
         'confirmation_primary':'Paired comparison against uniform_flow/standard_decoder at equal steps. Preserve old strict/diversity criteria; report no-success if candidate fails.',
         'new_source_sha256':sha(Path(__file__)),
         'dependencies':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'generation_recovery/p1_train.py',ROOT/'generation_recovery/p4_prepare.py',ROOT/'generation_recovery/models.py',ROOT/'followup/decoder_study.py',ROOT/'work/orbitlab.py',ROOT/'work/research.py']}}
    if LOCK.exists():
        assert json.loads(LOCK.read_text())==cfg,'Frozen generation source/protocol changed'
    else:dump(LOCK,cfg)
    return cfg

def prepare():
    freeze()
    original_prior=prep.prior_orbits
    def excluded():
        seen=original_prior()
        for path in (ROOT/'generation_recovery/confirmation_v1/data').rglob('*metadata.json'):
            seen.update(v['orbit_sha256'] for v in json.loads(path.read_text()))
        return seen
    prep.BASE,prep.LOCK,prep.DATA=BASE,LOCK,DATA
    prep.CONFIRMATION_SEEDS=SEEDS
    prep.locked=lambda:json.loads(LOCK.read_text())
    prep.prior_orbits=excluded
    return prep.prepare()

def context(seed,init):
    return {'protocol_sha256':sha(LOCK),'new_trainer_sha256':sha(Path(__file__)),
            'data_manifest_sha256':sha(DATA/'manifest.json'),'data_seed':seed,'initialization':init}

def setup(seed,init):
    freeze();folder=RUNS/f'd{seed}_s{init}'
    p1.RUN=folder;p1.DATA=DATA/f'd{seed}';p1.source_hashes=lambda:context(seed,init)
    original_seed=o.seed_all
    o.seed_all=lambda n:original_seed(n+1000*init)
    return folder

def edge_loss(pred,target):
    mask=(target.amax(1,keepdim=True)>.05).float()
    error=(pred-target).square().mean(1,keepdim=True)
    fg=(error*mask).sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1)
    bg=(error*(1-mask)).sum((1,2,3))/(1-mask).sum((1,2,3)).clamp_min(1)
    gradients=[]
    for axis in (-1,-2):
        delta=(torch.diff(pred,dim=axis)-torch.diff(target,dim=axis)).square().mean(1,keepdim=True)
        gradients.append(delta.sum((1,2,3))/mask.sum((1,2,3)).clamp_min(1))
    return (fg+.1*bg+.25*(gradients[0]+gradients[1])).mean()

def train_extra(seed,init,kind):
    folder=RUNS/f'd{seed}_s{init}'/kind
    ckpath=folder/'checkpoint.pt'
    if (folder/'run.json').exists():
        report=json.loads((folder/'run.json').read_text())
        assert report['context']==context(seed,init) and sha(ckpath)==report['checkpoint_sha256']
        return report
    folder.mkdir(parents=True,exist_ok=True)
    item=p1.pool();z=item['z'];labels=item['labels']
    decoder=kind=='area_edge_decoder'
    o.seed_all(86000 if decoder else 81000)
    model=ControlledDecoder('logit_mean') if decoder else from_variant('F0')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(87000 if decoder else 82000)
    images=None
    if decoder:
        x,_=p1.data('train');images=torch.cat([o.rotate(x,k) for k in range(4)])
    meta=json.loads((p1.DATA/'train_metadata.json').read_text())
    bins=torch.tensor([int(np.searchsorted([.13+.11/3,.13+2*.11/3],v['radius'])) for v in meta]).repeat(4)
    strata=labels[:,0]*18+labels[:,1]*3+bins
    counts=torch.bincount(strata,minlength=72)
    weights=1/counts[strata].float()
    progress=folder/'progress.pt';start_step=0;prior=0.;logs=[]
    if progress.exists():
        ck=torch.load(progress,map_location='cpu',weights_only=True)
        assert ck['context']==context(seed,init)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);rng.set_state(ck['rng'])
        start_step=ck['step'];prior=ck['seconds'];logs=ck['logs']
    started=time.perf_counter();steps=6000 if decoder else 16000;batch=32 if decoder else 128
    for step in range(start_step+1,steps+1):
        idx=(torch.multinomial(weights,batch,replacement=True,generator=rng)
             if kind=='stratified_flow' else torch.randint(len(z),(batch,),generator=rng))
        if decoder:loss=edge_loss(model(z[idx]),images[idx])
        else:
            target=z[idx];noise=torch.randn(target.shape,generator=rng);t=torch.rand(batch,generator=rng)
            point=(1-t[:,None,None])*noise+t[:,None,None]*target
            loss=F.mse_loss(model(point,t,labels[idx]),target-noise)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm)
        optimizer.step()
        if step%500==0:
            seconds=prior+time.perf_counter()-started
            logs.append({'step':step,'loss':float(loss.detach()),'seconds':seconds})
            saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng.get_state(),
                   'step':step,'seconds':seconds,'logs':logs,'context':context(seed,init)}
            tmp=progress.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(progress)
            print(seed,init,kind,logs[-1],flush=True)
    torch.save({'state_dict':model.state_dict(),'mean':item['mean'],'std':item['std'],
                'ae_sha256':item['ae_sha256'],'steps':steps,'context':context(seed,init),'kind':kind},ckpath)
    report={'context':context(seed,init),'checkpoint_sha256':sha(ckpath),'steps':steps,
            'seconds':prior+time.perf_counter()-started,'logs':logs,'status':'trained',
            'stratum_counts':counts.tolist()}
    dump(folder/'run.json',report);return report

def train(seed,init):
    folder=setup(seed,init);ctx=context(seed,init)
    correct_checkpoint(folder/'ae','ae.pt',seed=seed,init=init,context=ctx)
    result=p1.ae_train()
    correct_checkpoint(folder/'ae','ae.pt',seed=seed,init=init,context=ctx)
    if result['status']!='completed':raise RuntimeError('AE training budget interrupted')
    # A failed validation gate is retained, rather than silently dropping a seed.
    correct_checkpoint(folder/'decoder_logit_mean','decoder.pt',seed=seed,init=init,context=ctx)
    p1.decoder_train('logit_mean')
    correct_checkpoint(folder/'decoder_logit_mean','decoder.pt',seed=seed,init=init,context=ctx)
    for kind in ('uniform_flow','stratified_flow','area_edge_decoder'):train_extra(seed,init,kind)
    end=folder/'training_summary.json'
    if not end.exists():dump(end,{'status':'trained','context':ctx,'ae_gate_passed':result['validation_gate_passed'],
                                  'actual_initialization_seed_offset':1000*init})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train'])
    parser.add_argument('--seed',type=int,default=880101);parser.add_argument('--init',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='prepare':print(prepare(),flush=True)
    else:train(args.seed,args.init)
