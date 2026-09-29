"""Resumable, source-locked R4 development training. No test scoring."""
import argparse, json, os, resource, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from v3_common import ROOT, HERE, sha, dump, lock, event, status
from r4_data import BASE, DEVELOPMENT, freeze as data_freeze
from r4_core import (o, CANDIDATES, ae_model, decoder_model, FactorBridge,
    from_variant, representation, oracle_codes, edge_loss, decoder_loss, state_hash, tensor_hash)

RUNS = BASE/'runs'
SEED = DEVELOPMENT
INIT = 0
ROLES = ('oracle', 'learned')
COMPONENTS = ['ae_base', 'ae_spatial', 'bridge_base', 'bridge_spatial'] + [f'decoder_{role}_{c}' for role in ROLES for c in CANDIDATES] + ['flow_base', 'flow_spatial']

def freeze():
    data_freeze()
    return lock(BASE/'training_protocol.json', {
        'version': 1, 'development_seed': DEVELOPMENT, 'initialization': 0,
        'confirmation_conditional_seeds': [889201, 889202, 889203], 'confirmation_initializations': [0,1,2],
        'candidates': list(CANDIDATES), 'roles': list(ROLES), 'components': COMPONENTS,
        'steps': {'ae': 6000, 'decoder': 6000, 'bridge': 4000, 'flow': 16000},
        'total_development_updates': 100000, 'batch': {'ae':32,'decoder':32,'bridge':128,'flow':128},
        'optimizer': 'AdamW lr=.001 weight_decay=.0001; global grad norm clip1',
        'device': 'CPU float32, 2 torch threads', 'selection': 'Fixed final update, no validation checkpoint selection',
        'initialization_seeds': 'AE:910000+init; decoder:920000+init; bridge:930000+init; flow:940000+init. Same standard/edge initial state and exact sampling stream within role.',
        'sampling_seeds': 'Corresponding model seed+100000. AE uniform base scene plus one global quarter turn per minibatch; others uniform over all train scene/rotation pairs. Flow fresh Gaussian noise and uniform t each update.',
        'base_encoder': 'Historical ResearchAE equivariant16, original pixel-mean decoder and original weighted image MSE; no factor supervision. Frozen before all downstream training.',
        'spatial_encoder': 'Conv3->16->32->32->1 with four stride2 stages; each orientation block is1x4x4. Native spatial decoder and area/edge image loss. Changes encoder, decoder and objective; not an architecture-only intervention.',
        'standard_edge_pair': 'Identical frozen base encoder, normalization, architecture, initialization, sampled scenes and optimizer budget. Standard weighted image MSE versus fg MSE+.1bg MSE+.25area-normalized gradient MSE.',
        'spatial_decoder': 'Four upsampling convolutions from each1x4x4 block; average rotated logits then sigmoid. Oracle factor channels are reshaped4x4 for this diagnostic; they are not a true feature map.',
        'coordinate_decoder': 'Shared base encoder; learned33x33RGBA patch and tanh xy head; bilinear placement on64x64 and rotated RGB branch mean. Area/edge image loss+.1center MSE in8px units. Extra true center labels in training, absent at inference. Other decoder arms have no center labels.',
        'oracle_role': '4x16true factors:shape4,color6,relative xy2,radius1,direction2,constant1. Center(64*normalized_center-31.5)/31.5 fixes pixel-center rotation. Separately trained decoder weights from learned role; not a fixed-weight encoder swap.',
        'bridge_diagnostic': 'One shared-block16->64->16 factor readout per frozen encoder;4000MSE updates to normalized true factor codes. Uses extra labels; feed predicted factors into SAME oracle decoder as diagnostic only, never primary selection.',
        'normalization': 'Per representation and oracle role, train-only mean/std across scene and C4block dimensions; std clamped.02; preserves regular C4action.',
        'flows': 'Historical F0 concat C4-Reynolds velocity width256; base flow shared by standard,edge,coordinate; separate spatial flow. Uniform sampling,16000updates; generation Euler64steps,guidance1,stored paired3072x4x16noise.',
        'information_cost': 'All flow arms receive shape/color training and generation labels. Oracle decoders and bridges additionally receive all true factors; coordinate decoders receive true centers. Report parameter count, all upstream updates, runtime and input-label differences.',
        'preflight': 'Forward/backward, C4, center-action and loss identity before freeze; each component100step timing run reset before full training. Pilots separated from100000full updates.',
        'caps': {'stage_seconds':43200,'stage_bytes':30000000000},
        'sources': {str(p.relative_to(ROOT)):sha(p) for p in [HERE/'r4_core.py',HERE/'r4_train.py',HERE/'r4_data.py',HERE/'r4_preflight.py',ROOT/'work/orbitlab.py',ROOT/'work/research.py',ROOT/'followup/decoder_study.py',ROOT/'generation_recovery/models.py']}})

def data(seed, split):
    p=BASE/f'data/d{seed}'; a=np.load(p/f'{split}.npz')
    return torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255, torch.from_numpy(a['labels'].copy()), json.loads((p/f'{split}_metadata.json').read_text())

def root(seed, init): return RUNS/f'd{seed}_s{init}'

def settings(component):
    if component.startswith('ae_'): return 6000,32,910000
    if component.startswith('decoder_'): return 6000,32,920000
    if component.startswith('bridge_'): return 4000,128,930000
    if component.startswith('flow_'): return 16000,128,940000
    raise ValueError(component)

def make_model(component):
    if component.startswith('ae_'): return ae_model(component[3:])
    if component.startswith('decoder_'): return decoder_model(component.split('_')[2])
    if component.startswith('bridge_'): return FactorBridge()
    if component.startswith('flow_'): return from_variant('F0')
    raise ValueError(component)

def save_torch(path, value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)

def context(seed, init, component):
    return {'protocol_sha256':sha(BASE/'training_protocol.json'),'data_manifest_sha256':sha(BASE/f'data/d{seed}/manifest.json'),'seed':seed,'init':init,'component':component}

def load_model(seed, init, component):
    folder=root(seed,init)/component; run=json.loads((folder/'run.json').read_text())
    assert run['context']==context(seed,init,component) and sha(folder/'checkpoint.pt')==run['checkpoint_sha256']
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=make_model(component);model.load_state_dict(ck['state_dict']);model.eval()
    return model,ck

@torch.no_grad()
def pools(seed, init):
    freeze(); folder=root(seed,init)/'pools'; mp=folder/'manifest.json'
    if mp.exists():
        m=json.loads(mp.read_text());assert m['training_protocol_sha256']==sha(BASE/'training_protocol.json')
        for n,h in m['files'].items():assert sha(folder/n)==h
        for rep,h in m['ae_checkpoints'].items():assert sha(root(seed,init)/f'ae_{rep}/checkpoint.pt')==h
        return m
    folder.mkdir(parents=True,exist_ok=True);normal={};ae_hashes={}
    models={rep:load_model(seed,init,f'ae_{rep}')[0] for rep in ['base','spatial']}
    ae_hashes={rep:sha(root(seed,init)/f'ae_{rep}/checkpoint.pt') for rep in models}
    for split in ['train','val','ood']:
        x,y,meta=data(seed,split);raw={rep:[] for rep in ['base','spatial','oracle']}
        for rotation in range(4):
            raw['oracle'].append(oracle_codes(y,meta,rotation))
            for rep,model in models.items():raw[rep].append(torch.cat([model.encode(o.rotate(x[i:i+64],rotation)) for i in range(0,len(x),64)]))
        raw={k:torch.cat(v) for k,v in raw.items()}
        if split=='train':normal={k:{'mean':v.mean((0,1),keepdim=True),'std':v.std((0,1),unbiased=False,keepdim=True).clamp_min(.02)} for k,v in raw.items()}
        record={'labels':y.repeat(4,1),'base_n':len(x),'raw':raw,'z':{k:(v-normal[k]['mean'])/normal[k]['std'] for k,v in raw.items()},'center':raw['oracle'][...,10:12]}
        save_torch(folder/f'{split}.pt',record)
        print('R4 pool',split,len(record['labels']),flush=True)
    save_torch(folder/'normalization.pt',normal)
    m={'training_protocol_sha256':sha(BASE/'training_protocol.json'),'ae_checkpoints':ae_hashes,'files':{p.name:sha(p) for p in folder.glob('*.pt')},'test_encoded':False}
    dump(mp,m);return m

def cap_check(additional_seconds=0):
    started=BASE/'training_started.json'
    if started.exists():assert time.time()-json.loads(started.read_text())['unix']+additional_seconds<43200,'R4 stage time cap; preserve partial results'
    size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    assert size<30000000000,'R4 storage cap'
    return size

def train(seed, init, component, benchmark=False):
    assert component in COMPONENTS
    assert seed in [DEVELOPMENT,889201,889202,889203] and init in [0,1,2]
    freeze();torch.set_num_threads(2);steps,batch,model_seed=settings(component)
    if not benchmark:
        lock(BASE/'training_started.json', json.loads((BASE/'training_started.json').read_text()) if (BASE/'training_started.json').exists() else {'unix':time.time(),'seed':seed,'init':init})
    folder=root(seed,init)/('benchmarks' if benchmark else '')/component;report_path=folder/'run.json';ctx=context(seed,init,component)
    if report_path.exists():
        info=json.loads(report_path.read_text());assert info['context']==ctx
        if not benchmark:assert sha(folder/'checkpoint.pt')==info['checkpoint_sha256']
        return info
    o.seed_all(model_seed+init);model=make_model(component);initial=state_hash(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);rng=torch.Generator().manual_seed(model_seed+100000+init)
    x,y,meta=data(seed,'train');pool=None;deps={}
    if not component.startswith('ae_'):
        pools(seed,init);pool=torch.load(root(seed,init)/'pools/train.pt',map_location='cpu',weights_only=True)
        deps={'pool_manifest_sha256':sha(root(seed,init)/'pools/manifest.json')}
    progress=folder/'progress.pt';first=0;prior=0.;logs=[]
    if progress.exists():
        ck=torch.load(progress,map_location='cpu',weights_only=True);assert ck['context']==ctx and ck['dependencies']==deps
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);rng.set_state(ck['rng']);torch.set_rng_state(ck['torch_rng'])
        first=ck['step'];prior=ck['seconds'];logs=ck['logs']
    total=100 if benchmark else steps;start=time.perf_counter();model.train()
    def save(step):
        value={'context':ctx,'dependencies':deps,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng.get_state(),'torch_rng':torch.get_rng_state(),'step':step,'seconds':prior+time.perf_counter()-start,'logs':logs}
        save_torch(progress,value)
    for step in range(first+1,total+1):
        details={}
        if component.startswith('ae_'):
            idx=torch.randint(len(x),(batch,),generator=rng);rotation=int(torch.randint(4,(1,),generator=rng));target=o.rotate(x[idx],rotation)
            pred=model(target);loss=edge_loss(pred,target) if component=='ae_spatial' else o.reconstruction_loss(pred,target)
        else:
            idx=torch.randint(len(pool['labels']),(batch,),generator=rng)
            if component.startswith('decoder_'):
                _,role,candidate=component.split('_');rep='oracle' if role=='oracle' else representation(candidate)
                rot=(idx//len(x));base=idx%len(x)
                target=torch.stack([o.rotate(x[int(b)],int(k)) for b,k in zip(base,rot)])
                loss,details=decoder_loss(candidate,model,pool['z'][rep][idx],target,pool['center'][idx])
            elif component.startswith('bridge_'):
                rep=component[7:];loss=F.mse_loss(model(pool['z'][rep][idx]),pool['z']['oracle'][idx])
            else:
                rep=component[5:];target=pool['z'][rep][idx];noise=torch.randn(target.shape,generator=rng);t=torch.rand(batch,generator=rng)
                point=(1-t[:,None,None])*noise+t[:,None,None]*target
                loss=F.mse_loss(model(point,t,pool['labels'][idx]),target-noise)
        assert torch.isfinite(loss),f'Nonfinite loss {component}step{step}'
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if step%500==0 or step==total:
            logs.append({'step':step,'loss':float(loss.detach()),'seconds':prior+time.perf_counter()-start,**details});save(step)
            print('R4',component,'pilot' if benchmark else 'full',logs[-1],flush=True);cap_check()
    seconds=prior+time.perf_counter()-start
    report={'context':ctx,'dependencies':deps,'status':'benchmark_complete' if benchmark else 'trained','steps':total,'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial,'final_state_sha256':state_hash(model),'seconds':seconds,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds_per_update':seconds/total,'predicted_full_seconds':seconds/total*steps,'logs':logs,'pilot_updates_excluded_from_full_budget':benchmark}
    if not benchmark:
        save_torch(folder/'checkpoint.pt',{'state_dict':model.state_dict(),'context':ctx,'dependencies':deps,'steps':total});report['checkpoint_sha256']=sha(folder/'checkpoint.pt')
    dump(report_path,report);event('R4_component_finished',component=component,benchmark=benchmark,steps=total,seconds=seconds)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('component',choices=[*COMPONENTS,'pools']);p.add_argument('--benchmark',action='store_true');p.add_argument('--seed',type=int,default=SEED);p.add_argument('--init',type=int,default=INIT);a=p.parse_args()
    print(pools(a.seed,a.init) if a.component=='pools' else train(a.seed,a.init,a.component,a.benchmark),flush=True)
