"""Intervene on flow equivariance while holding a C4 representation fixed."""
from __future__ import annotations
import argparse
import json
import resource
import time
from pathlib import Path
import torch.nn.functional as F
from common import HERE, ROOT, dump, fresh_dir, digest_tensor, state_digest, update_status, np, torch, o, r
import generation as g
from evaluator_v2 import DetailedEvaluator, accepted

CFG = json.loads((HERE/'configs/stage_a_v1.json').read_text())


def rotate_batch(z, k):
    assert z.ndim == 3 and z.shape[1] == 4 and k.shape == (len(z),)
    slots = (torch.arange(4, device=z.device)[None] - k[:,None]) % 4
    return z.gather(1, slots[:,:,None].expand_as(z))


def flow_batch(z, labels, sampler, augmentation_rng, mode, device):
    indices = torch.randint(len(z), (CFG['flow_batch'],), generator=sampler)
    target = z[indices]
    noise = torch.randn(target.shape, generator=sampler)
    t = torch.rand(len(target), generator=sampler)
    if mode == 'augment':
        k = torch.randint(4, (len(target),), generator=augmentation_rng)
        target, noise = rotate_batch(target,k), rotate_batch(noise,k)
    target, noise, t = target.to(device), noise.to(device), t.to(device)
    return (1-t[:,None,None])*noise+t[:,None,None]*target, target-noise, t, labels[indices].to(device)


@torch.no_grad()
def codes(seed, device):
    ae_path = ROOT/f'runs/repair_equivariant_n1024_s{seed}/ae.pt'
    ae,ack = r.load_model(ae_path,device)
    assert ack['model']=='equivariant'
    ae.requires_grad_(False)
    folder = HERE/'cache'; folder.mkdir(exist_ok=True)
    path = folder/f'codes_c4_s{seed}.pt'
    if path.exists():
        d = torch.load(path,map_location='cpu',weights_only=True)
        assert d['ae_sha256']==r.sha(ae_path) and d['data_sha256']==r.sha(ROOT/'data/manifest.json')
        return ae,ae_path,d
    x,y,_ = r.load_data('train',ack['n']); z=[];labels=[]
    o.sync(device);start=time.perf_counter()
    for k in range(4):
        for i in range(0,len(x),64):
            z.append(ae.encode(o.rotate(x[i:i+64].to(device),k)).cpu());labels.append(y[i:i+64])
    z=torch.cat(z);labels=torch.cat(labels)
    mean=z.mean((0,1),keepdim=True);std=z.std((0,1),keepdim=True,unbiased=False).clamp_min(.02)
    z=(z-mean)/std
    o.sync(device)
    d={'z':z,'labels':labels,'mean':mean,'std':std,'ae_sha256':r.sha(ae_path),
       'data_sha256':r.sha(ROOT/'data/manifest.json'),'encoding_seconds':time.perf_counter()-start,
       'latent_sha256':digest_tensor(z)}
    torch.save(d,path)
    return ae,ae_path,d


def train(seed, mode, budget, device):
    tag='fixed' if budget is None else 'time'
    folder=HERE/f'runs/flow_{tag}_s{seed}_{mode}'
    if (folder/'run.json').exists():
        report=json.loads((folder/'run.json').read_text())
        assert r.sha(folder/'flow.pt')==report['checkpoint_sha256']
        return folder
    fresh_dir(folder)
    dump(folder/'started.json',{'seed':seed,'mode':mode,'budget_seconds':budget,'started_unix':time.time(),'code_sha256':r.sha(__file__)})
    ae,ae_path,d=codes(seed,device);before=state_digest(ae)
    o.seed_all(CFG['flow_init_seed_offset']+seed)
    flow=o.Velocity(16,mode=='equivariant').to(device)
    initial=state_digest(flow)
    opt=torch.optim.AdamW(flow.parameters(),lr=CFG['flow_lr'],weight_decay=1e-4)
    sampler=torch.Generator().manual_seed(CFG['flow_sampling_seed_offset']+seed)
    aug=torch.Generator().manual_seed(CFG['flow_augmentation_seed_offset']+seed)
    # Warm kernels without advancing any training RNG or updating weights.
    dummy=d['z'][:CFG['flow_batch']].to(device);y=d['labels'][:len(dummy)].to(device)
    for _ in range(5):
        loss=flow(dummy,torch.full((len(dummy),),.5,device=device),y).square().mean()
        loss.backward();opt.zero_grad(set_to_none=True)
    assert state_digest(flow)==initial
    o.sync(device);start=time.perf_counter();logs=[]
    max_steps=CFG['flow_steps'] if budget is None else 100000
    for step in range(1,max_steps+1):
        point,velocity,t,y=flow_batch(d['z'],d['labels'],sampler,aug,mode,device)
        loss=F.mse_loss(flow(point,t,y),velocity)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite flow loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(flow.parameters(),1.)
        if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
        opt.step()
        if step%500==0:
            o.sync(device);logs.append({'step':step,'loss':float(loss),'seconds':time.perf_counter()-start})
        if budget is not None:
            o.sync(device)
            if time.perf_counter()-start>=budget:break
    o.sync(device);elapsed=time.perf_counter()-start
    if budget is not None and elapsed<budget:raise RuntimeError('Time budget not reached')
    assert state_digest(ae)==before
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in flow.state_dict().items()},'equivariant':mode=='equivariant',
                'mode':mode,'mean':d['mean'],'std':d['std'],'ae_path':str(ae_path),'ae_sha256':d['ae_sha256'],'seed':seed}
    torch.save(checkpoint,folder/'flow.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':sampler.get_state(),'augmentation_rng':aug.get_state(),'steps':step},folder/'optimizer.pt')
    original_run=json.loads((ae_path.parent/'run.json').read_text())
    dump(folder/'run.json',{'mode':mode,'seed':seed,'budget_type':tag,'budget_seconds':budget,'steps':step,'train_wall_seconds':elapsed,
         'original_ae_train_seconds':original_run['train_wall_seconds'],'ae_plus_flow_train_seconds':original_run['train_wall_seconds']+elapsed,
         'encoding_seconds':d['encoding_seconds'],'ae_frozen_unchanged':True,'ae_sha256':d['ae_sha256'],'latent_sha256':d['latent_sha256'],
         'initial_raw_weights_sha256':initial,'parameters':sum(p.numel() for p in flow.parameters()),
         'internal_evaluations_per_forward':4 if mode=='equivariant' else 1,'loss_log':logs,
         'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'mps_driver_allocation_at_end_bytes':torch.mps.driver_allocated_memory() if device.type=='mps' else None,
         'rss_note':'Process high-water mark; in the matrix process it includes earlier tasks and is not per-run device peak.',
         'augmentation_note':'The base latent pool already contains every C4 orientation. Extra augmentation tests parameterization/optimization effects, not missing orientation coverage.',
         'checkpoint_sha256':r.sha(folder/'flow.pt'),'config_sha256':r.sha(HERE/'configs/stage_a_v1.json'),
         'code_sha256':r.sha(__file__),'environment':o.environment()})
    print('trained',folder.name,'steps',step,'seconds',round(elapsed,3),flush=True)
    return folder


@torch.no_grad()
def evaluate(folder,device):
    out=folder/'samples'
    if (out/'metrics.json').exists():
        d=json.loads((out/'metrics.json').read_text());assert d['flow_sha256']==r.sha(folder/'flow.pt');return d
    fresh_dir(out)
    ck=torch.load(folder/'flow.pt',map_location='cpu',weights_only=True)
    ae,_=r.load_model(ck['ae_path'],device);assert r.sha(ck['ae_path'])==ck['ae_sha256']
    flow=o.Velocity(16,ck['equivariant']).to(device);flow.load_state_dict(ck['state_dict']);flow.eval()
    threshold_path=HERE/'reports/evaluator_calibration_v1/locked_threshold.json'
    screen_threshold=json.loads(threshold_path.read_text())
    ev=DetailedEvaluator();gen=torch.Generator().manual_seed(CFG['flow_eval_seed_offset']+ck['seed'])
    noise=torch.randn(CFG['flow_eval_n'],4,16,generator=gen)
    labels=torch.tensor([[i%4,(i//4)%6] for i in range(len(noise))])
    results={};saved={};rows=[]
    # Same inference warm-up for all modes, excluded from reported sample time.
    flow(noise[:128].to(device),torch.zeros(128,device=device),labels[:128].to(device));o.sync(device)
    for steps in CFG['flow_eval_steps']:
        images=[];zs=[];o.sync(device);start=time.perf_counter()
        for i in range(0,len(noise),128):
            z=noise[i:i+128].to(device).clone();y=labels[i:i+128].to(device)
            for k in range(steps):z=z+flow(z,torch.full((len(z),),k/steps,device=device),y)/steps
            raw=z*ck['std'].to(device)+ck['mean'].to(device)
            images.append(ae.decode(raw).cpu());zs.append(raw.cpu())
        o.sync(device);seconds=time.perf_counter()-start;images=torch.cat(images);saved[steps]=torch.cat(zs)
        predictions=ev.predict(images);sr=[]
        for i,(p,y) in enumerate(zip(predictions,labels.tolist())):
            row={'sample':i,'steps':steps,'shape':y[0],'color':y[1],'ood':y[0]==y[1],**p,
                 'shape_correct':int(p['predicted_shape']==y[0]),'color_correct':int(p['predicted_color']==y[1])}
            row['joint_correct']=row['shape_correct']*row['color_correct']
            row['strict_accepted']=int(accepted(p,screen_threshold))
            row['strict_accepted_and_joint']=int(row['joint_correct'] and accepted(p,screen_threshold))
            for threshold in CFG['template_validity_thresholds']:
                row[f'valid_joint_{threshold}']=int(row['joint_correct'] and p['nonempty'] and p['template_iou']>=threshold)
            rows.append(row);sr.append(row)
        results[str(steps)]={'sample_and_decode_seconds':seconds}
        for split in ['seen','ood']:
            subset=[p for p in sr if p['ood']==(split=='ood')]
            results[str(steps)][split]={'n':len(subset),**{k:float(np.mean([v[k] for v in subset])) for k in ['shape_correct','color_correct','joint_correct','template_iou','strict_accepted','strict_accepted_and_joint']+[f'valid_joint_{t}' for t in CFG['template_validity_thresholds']]}}
        o.grid(images[:48],out/f'samples_{steps}.png',8)
        if steps==64:
            worst=np.argsort([p['template_iou'] for p in sr])[:16].copy()
            o.grid(images[worst],out/'lowest_template_fit.png',8)
    r.write_csv(out/'rows.csv',rows)
    torch.save({'noise':noise,'labels':labels,'latents':saved},out/'samples.pt')
    results.update(flow_sha256=r.sha(folder/'flow.pt'),ae_sha256=ck['ae_sha256'],noise_sha256=digest_tensor(noise),
                   strict_threshold=screen_threshold,strict_threshold_sha256=r.sha(threshold_path))
    dump(out/'metrics.json',results)
    print('evaluated',folder.name,'64-step',results['64'],flush=True)
    return results


def summarize(tag):
    records=[]
    for seed in CFG['ae_seeds']:
        for mode in CFG['flow_modes']:
            folder=HERE/f'runs/flow_{tag}_s{seed}_{mode}'
            run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'samples/metrics.json').read_text())
            records.append({'seed':seed,'mode':mode,'train_seconds':run['train_wall_seconds'],'steps':run['steps'],
                'ae_sha256':run['ae_sha256'],'initial_raw_weights_sha256':run['initial_raw_weights_sha256'],
                'latent_sha256':run['latent_sha256'],'noise_sha256':m['noise_sha256'],'parameters':run['parameters'],
                **{f'{split}_{key}':m['64'][split][key] for split in ['seen','ood'] for key in ['joint_correct','valid_joint_0.65','strict_accepted_and_joint']},
                'sample_64_seconds':m['64']['sample_and_decode_seconds']})
    for seed in CFG['ae_seeds']:
        subset=[v for v in records if v['seed']==seed]
        for key in ['ae_sha256','initial_raw_weights_sha256','latent_sha256','noise_sha256','parameters']:
            assert len({v[key] for v in subset})==1,(seed,key)
    keys=['seen_joint_correct','seen_valid_joint_0.65','seen_strict_accepted_and_joint','ood_joint_correct','ood_valid_joint_0.65','ood_strict_accepted_and_joint','train_seconds','sample_64_seconds']
    summary={mode:{k:{'mean':float(np.mean([v[k] for v in records if v['mode']==mode])),
                      'seed_sd':float(np.std([v[k] for v in records if v['mode']==mode],ddof=1))} for k in keys} for mode in CFG['flow_modes']}
    paired={mode:{k:[next(v[k] for v in records if v['mode']==mode and v['seed']==seed)-next(v[k] for v in records if v['mode']=='plain' and v['seed']==seed) for seed in CFG['ae_seeds']] for k in keys[:6]} for mode in ['augment','equivariant']}
    path=HERE/f'reports/flow_{tag}_summary.json'
    dump(path,{'records':records,'summary':summary,'paired_differences_from_plain':paired,
         'matching_verified':True,'data_seeds':1,'ae_and_flow_initializations':3,
         'limitations':'Same three existing C4 AEs; not a direct B1/B2 representation intervention. Evaluation is automatic, human validation pending.',
         'config_sha256':r.sha(HERE/'configs/stage_a_v1.json')})
    r.write_csv(HERE/f'reports/flow_{tag}_summary.csv',records)
    update_status('A07' if tag=='fixed' else 'A08','complete',[str(path.relative_to(HERE))])
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',choices=['fixed','time'],required=True);p.add_argument('--device',default='mps')
    a=p.parse_args();torch.set_num_threads(4);device=o.choose_device(a.device)
    for seed in CFG['ae_seeds']:
        budget=None
        if a.matrix=='time':budget=json.loads((HERE/f'runs/flow_fixed_s{seed}_equivariant/run.json').read_text())['train_wall_seconds']
        for mode in CFG['flow_modes']:
            folder=train(seed,mode,budget,device);evaluate(folder,device)
    print(json.dumps(summarize(a.matrix),indent=2),flush=True)


if __name__=='__main__':main()
