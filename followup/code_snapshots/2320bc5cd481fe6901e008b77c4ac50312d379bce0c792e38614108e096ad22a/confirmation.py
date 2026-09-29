"""Fresh-data confirmation: 3 data seeds x 3 initialization seeds.

All checkpoints use fixed stopping rules. Failed AE validation gates are retained,
not dropped. Original data-seed-42 results are not counted as confirmation samples.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch.nn.functional as F
from common import HERE, ROOT, dump, fresh_dir, original_orbits, render_fresh, state_digest, digest_tensor, update_status, np, torch, o, r
from evaluator_v2 import DetailedEvaluator, accepted
from flow_ablation import flow_batch

CONFIG = HERE/'configs/confirmation_v1.json'


def freeze():
    config={'data_seeds':[43001,43002,43003],'init_seeds':[0,1,2],'train_n':1024,'evaluation_n':256,
        'ae_steps':6000,'ae_lr':.001,'ae_batch':32,'ae_sampling_seed_offset':100000,
        'flow_steps':4000,'flow_lr':.001,'flow_batch':128,
        'flow_init_seed_offset':81000,'flow_sampling_seed_offset':82000,
        'flow_noise_seed_offset':951000,'sample_n':1920,'ode_steps':64,
        'modes':['plain_fixed','equivariant_fixed','plain_time'],
        'primary':'strict_accepted_and_joint under the locked evaluator screen; not human semantic success',
        'secondary':['shape_correct','color_correct','joint_correct','legacy_valid_and_joint'],
        'threshold_path':'reports/evaluator_calibration_v1/locked_threshold.json',
        'threshold_sha256':r.sha(HERE/'reports/evaluator_calibration_v1/locked_threshold.json'),
        'ae_gate':'Record val foreground MAE<=.10, IoU>=.70, shape/color>=.90. Retain all prespecified runs even if a gate fails; report the gate and interpret as diagnostic.',
        'selection':'Fixed final steps; no tuning or seed replacement after confirmation test results.',
        'time_control':'plain_time uses the corresponding 4000-step equivariant flow wall time. Synchronize each step in all flow modes.',
        'independence':'Three entirely fresh data seeds; original data-seed-42 results are exploratory, not one of these confirmation seeds.',
        'frozen_unix':time.time()}
    if not CONFIG.exists():dump(CONFIG,config)
    config=json.loads(CONFIG.read_text())
    assert r.sha(HERE/config['threshold_path'])==config['threshold_sha256']
    return config


def prepare(c):
    root=HERE/'data/confirmation_v1'
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        for record in manifest['files']:
            assert r.sha(root/record['path'])==record['sha256']
        return root
    fresh_dir(root)
    forbidden=original_orbits()
    for p in (HERE/'data/probe_v1').glob('*_metadata.json'):
        forbidden.update(v['orbit_sha256'] for v in json.loads(p.read_text()))
    for name in ['evaluator_audit_v1/sources.json','evaluator_calibration_v1/calibration_sources.json','evaluator_calibration_v1/test_sources.json']:
        forbidden.update(v['orbit_sha256'] for v in json.loads((HERE/'reports'/name).read_text()))
    before=len(forbidden);files=[];counts={}
    for seed in c['data_seeds']:
        folder=root/str(seed);folder.mkdir()
        for split in ['train','val','test','ood']:
            n=c['train_n'] if split=='train' else c['evaluation_n']
            x,y,m=render_fresh(n,seed,split,forbidden)
            new={v['orbit_sha256'] for v in m};assert len(new)==n and not new&forbidden
            forbidden.update(new)
            np.savez_compressed(folder/f'{split}.npz',images=(x.permute(0,2,3,1).numpy()*255).round().astype(np.uint8),labels=y.numpy())
            dump(folder/f'{split}_metadata.json',m)
            for suffix in [f'{split}.npz',f'{split}_metadata.json']:
                p=folder/suffix;files.append({'path':str(p.relative_to(root)),'sha256':r.sha(p)})
            counts[f'{seed}/{split}']=n
    dump(root/'manifest.json',{'files':files,'counts':counts,'all_new_orbits_unique':True,
         'excluded_previous_orbits':before,'new_orbits':len(forbidden)-before,'config_sha256':r.sha(CONFIG)})
    return root


def data(root,seed,split):
    a=np.load(root/str(seed)/f'{split}.npz')
    return torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255,torch.from_numpy(a['labels'].copy())


def train_ae(c,data_root,data_seed,seed,device):
    folder=HERE/f'runs/confirmation_v1/d{data_seed}_s{seed}/ae'
    if (folder/'run.json').exists():
        report=json.loads((folder/'run.json').read_text());assert r.sha(folder/'ae.pt')==report['checkpoint_sha256']
        ae,_=r.load_model(folder/'ae.pt',device);return ae,folder
    fresh_dir(folder);o.seed_all(seed);x,_=data(data_root,data_seed,'train')
    ae=r.ResearchAE('equivariant',16).to(device)
    opt=torch.optim.AdamW(ae.parameters(),lr=c['ae_lr'],weight_decay=1e-4)
    rng=torch.Generator().manual_seed(c['ae_sampling_seed_offset']+seed)
    dump(folder/'started.json',{'data_seed':data_seed,'init_seed':seed,'config_sha256':r.sha(CONFIG),'code_sha256':r.sha(__file__),'started_unix':time.time()})
    logs=[];o.sync(device);start=time.perf_counter()
    for step in range(1,c['ae_steps']+1):
        idx=torch.randint(len(x),(c['ae_batch'],),generator=rng);k=int(torch.randint(4,(1,),generator=rng))
        target=o.rotate(x[idx].to(device),k);loss=o.reconstruction_loss(ae(target),target)
        if not torch.isfinite(loss):raise FloatingPointError('AE loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(ae.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('AE gradient')
        opt.step()
        if step%1000==0:
            o.sync(device);logs.append({'step':step,'loss':loss.detach().item(),'seconds':time.perf_counter()-start})
            print('AE',data_seed,seed,logs[-1],flush=True)
    o.sync(device);seconds=time.perf_counter()-start;ae.eval()
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in ae.state_dict().items()},'model':'equivariant','width':16,'channels':16,
                'size':64,'seed':seed,'data_seed':data_seed,'n':c['train_n'],'steps':c['ae_steps']},folder/'ae.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    ev=DetailedEvaluator();vx,vy=data(data_root,data_seed,'val');rows=[]
    with torch.no_grad():
        for k in range(4):
            for i in range(0,len(vx),64):
                b=o.rotate(vx[i:i+64].to(device),k);p=ae(b);metrics=r.pixel_metrics(p,b)
                for j,(pred,label) in enumerate(zip(ev.predict(p),vy[i:i+64].tolist())):
                    rows.append({'base_id':i+j,'rotation':k,'shape_correct':pred['predicted_shape']==label[0],
                        'color_correct':pred['predicted_color']==label[1],**{name:float(v[j]) for name,v in metrics.items()}})
        o.grid(torch.cat([vx[:8],ae(vx[:8].to(device)).cpu()]),folder/'validation.png',8)
    r.write_csv(folder/'validation.csv',rows)
    val={key:float(np.mean([v[key] for v in rows])) for key in ['shape_correct','color_correct','foreground_mae','mask_iou']}
    gate=val['shape_correct']>=.9 and val['color_correct']>=.9 and val['foreground_mae']<=.1 and val['mask_iou']>=.7
    dump(folder/'run.json',{'data_seed':data_seed,'init_seed':seed,'steps':c['ae_steps'],'train_wall_seconds':seconds,
         'validation':val,'validation_gate_passed':gate,'loss_log':logs,'checkpoint_sha256':r.sha(folder/'ae.pt'),
         'data_manifest_sha256':r.sha(data_root/'manifest.json'),'config_sha256':r.sha(CONFIG),'code_sha256':r.sha(__file__),
         'note':'Validation gate recorded; failed seeds are retained as prespecified diagnostic comparisons.'})
    return ae,folder


@torch.no_grad()
def encode_pool(ae,root,data_seed,device):
    x,y=data(root,data_seed,'train');zs=[];labels=[]
    for k in range(4):
        for i in range(0,len(x),64):
            zs.append(ae.encode(o.rotate(x[i:i+64].to(device),k)).cpu());labels.append(y[i:i+64])
    z=torch.cat(zs);labels=torch.cat(labels);mean=z.mean((0,1),keepdim=True);std=z.std((0,1),keepdim=True,unbiased=False).clamp_min(.02)
    return (z-mean)/std,labels,mean,std


def train_flow(c,ae,ae_folder,pool,seed,mode,device,budget=None):
    folder=ae_folder.parent/mode
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert r.sha(folder/'flow.pt')==run['checkpoint_sha256'];return folder
    fresh_dir(folder);before=state_digest(ae);ae.requires_grad_(False)
    o.seed_all(c['flow_init_seed_offset']+seed);flow=o.Velocity(16,mode=='equivariant_fixed').to(device);initial=state_digest(flow)
    opt=torch.optim.AdamW(flow.parameters(),lr=c['flow_lr'],weight_decay=1e-4)
    z,y,mean,std=pool
    sampler=torch.Generator().manual_seed(c['flow_sampling_seed_offset']+seed);unused_aug=torch.Generator().manual_seed(83000+seed)
    for _ in range(5):
        loss=flow(z[:128].to(device),torch.full((128,),.5,device=device),y[:128].to(device)).square().mean()
        loss.backward();opt.zero_grad(set_to_none=True)
    assert state_digest(flow)==initial
    dump(folder/'started.json',{'mode':mode,'init_seed':seed,'budget_seconds':budget,'started_unix':time.time(),'config_sha256':r.sha(CONFIG)})
    o.sync(device);start=time.perf_counter();logs=[]
    for step in range(1,(c['flow_steps'] if budget is None else 100000)+1):
        point,target,t,label=flow_batch(z,y,sampler,unused_aug,'plain',device)
        loss=F.mse_loss(flow(point,t,label),target)
        if not torch.isfinite(loss):raise FloatingPointError('Flow loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Flow gradient')
        opt.step();o.sync(device)
        if step%1000==0:logs.append({'step':step,'loss':loss.detach().item(),'seconds':time.perf_counter()-start})
        if budget is not None and time.perf_counter()-start>=budget:break
    seconds=time.perf_counter()-start
    if budget is not None:assert seconds>=budget
    assert state_digest(ae)==before
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in flow.state_dict().items()},'equivariant':mode=='equivariant_fixed',
                'mean':mean,'std':std,'ae_path':str(ae_folder/'ae.pt'),'ae_sha256':r.sha(ae_folder/'ae.pt'),'seed':seed},folder/'flow.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':sampler.get_state(),'steps':step},folder/'optimizer.pt')
    dump(folder/'run.json',{'mode':mode,'seed':seed,'steps':step,'train_wall_seconds':seconds,'requested_seconds':budget,
        'ae_frozen_unchanged':True,'initial_raw_weights_sha256':initial,'latent_sha256':digest_tensor(z),'ae_sha256':r.sha(ae_folder/'ae.pt'),
        'parameters':sum(p.numel() for p in flow.parameters()),'loss_log':logs,'checkpoint_sha256':r.sha(folder/'flow.pt'),
        'config_sha256':r.sha(CONFIG),'code_sha256':r.sha(__file__),'batch_code_sha256':r.sha(HERE/'flow_ablation.py')})
    print('flow',folder.parent.name,mode,step,round(seconds,2),flush=True)
    return folder


@torch.no_grad()
def evaluate(c,folder,data_index,seed,device):
    path=folder/'metrics.json'
    if path.exists():
        result=json.loads(path.read_text());assert result['flow_sha256']==r.sha(folder/'flow.pt');return result
    ck=torch.load(folder/'flow.pt',map_location='cpu',weights_only=True);ae,_=r.load_model(ck['ae_path'],device)
    flow=o.Velocity(16,ck['equivariant']).to(device);flow.load_state_dict(ck['state_dict']);flow.eval()
    gen=torch.Generator().manual_seed(c['flow_noise_seed_offset']+100*data_index+seed)
    noise=torch.randn(c['sample_n'],4,16,generator=gen);labels=torch.tensor([[i%4,(i//4)%6] for i in range(len(noise))])
    images=[];latents=[];o.sync(device);start=time.perf_counter()
    for i in range(0,len(noise),128):
        z=noise[i:i+128].to(device).clone();y=labels[i:i+128].to(device)
        for k in range(c['ode_steps']):z+=flow(z,torch.full((len(z),),k/c['ode_steps'],device=device),y)/c['ode_steps']
        raw=z*ck['std'].to(device)+ck['mean'].to(device);images.append(ae.decode(raw).cpu());latents.append(raw.cpu())
    o.sync(device);seconds=time.perf_counter()-start;images=torch.cat(images)
    ev=DetailedEvaluator();threshold=json.loads((HERE/c['threshold_path']).read_text());rows=[]
    for i,(p,y) in enumerate(zip(ev.predict(images),labels.tolist())):
        joint=p['predicted_shape']==y[0] and p['predicted_color']==y[1]
        rows.append({'sample':i,'shape':y[0],'color':y[1],'ood':y[0]==y[1],**p,
            'shape_correct':p['predicted_shape']==y[0],'color_correct':p['predicted_color']==y[1],'joint_correct':joint,
            'legacy_valid_and_joint':joint and p['nonempty'] and p['template_iou']>=.65,
            'strict_accepted_and_joint':joint and accepted(p,threshold)})
    r.write_csv(folder/'samples.csv',rows);o.grid(images[:48],folder/'samples.png',8)
    torch.save({'noise':noise,'labels':labels,'latents':torch.cat(latents)},folder/'samples.pt')
    result={'flow_sha256':r.sha(folder/'flow.pt'),'noise_sha256':digest_tensor(noise),'sample_and_decode_seconds':seconds,'threshold':threshold}
    for split in ['seen','ood']:
        sr=[v for v in rows if v['ood']==(split=='ood')]
        result[split]={'n':len(sr),**{key:float(np.mean([v[key] for v in sr])) for key in c['secondary']+['strict_accepted_and_joint']}}
    dump(path,result);print('evaluated',folder.parent.name,folder.name,result['seen'],flush=True);return result


def summarize(c):
    rows=[]
    for d in c['data_seeds']:
        for s in c['init_seeds']:
            base=HERE/f'runs/confirmation_v1/d{d}_s{s}';ar=json.loads((base/'ae/run.json').read_text())
            block=[]
            for mode in c['modes']:
                folder=base/mode;run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
                row={'data_seed':d,'init_seed':s,'mode':mode,'ae_gate_passed':ar['validation_gate_passed'],
                    'ae_seconds':ar['train_wall_seconds'],'flow_seconds':run['train_wall_seconds'],'flow_steps':run['steps'],
                    'initial_sha':run['initial_raw_weights_sha256'],'latent_sha':run['latent_sha256'],'noise_sha':m['noise_sha256'],
                    **{f'{split}_{metric}':m[split][metric] for split in ['seen','ood'] for metric in ['joint_correct','strict_accepted_and_joint']}}
                rows.append(row);block.append(row)
            for key in ['initial_sha','latent_sha','noise_sha']:assert len({v[key] for v in block})==1
    summary={mode:{key:{'mean':float(np.mean([v[key] for v in rows if v['mode']==mode])),
                       'per_data_seed':[float(np.mean([v[key] for v in rows if v['mode']==mode and v['data_seed']==d])) for d in c['data_seeds']]}
                    for key in ['seen_joint_correct','seen_strict_accepted_and_joint','ood_joint_correct','ood_strict_accepted_and_joint']} for mode in c['modes']}
    out=HERE/'reports/confirmation_v1';out.mkdir(exist_ok=True)
    r.write_csv(out/'runs.csv',rows)
    dump(out/'summary.json',{'summary':summary,'runs':rows,'ae_runs':9,'flow_runs':27,'new_data_seeds':3,'initializations_per_data_seed':3,
        'gate_failures':[{'data_seed':v['data_seed'],'init_seed':v['init_seed']} for v in rows if v['mode']=='plain_fixed' and not v['ae_gate_passed']],
        'note':'Factorial repetitions; only three independent data-generation seeds. Automatic acceptance is not human-validated quality.',
        'config_sha256':r.sha(CONFIG),'matching_verified':True})
    update_status('A09','complete',['reports/confirmation_v1/summary.json','data/confirmation_v1/manifest.json'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);c=freeze();root=prepare(c)
    if a.prepare_only:print('confirmation protocol and disjoint data prepared',flush=True);return
    device=o.choose_device('mps');update_status('A09','running',['configs/confirmation_v1.json','data/confirmation_v1/manifest.json'])
    for di,d in enumerate(c['data_seeds']):
        for s in c['init_seeds']:
            ae,folder=train_ae(c,root,d,s,device);pool=encode_pool(ae,root,d,device)
            for mode in c['modes']:
                budget=json.loads((folder.parent/'equivariant_fixed/run.json').read_text())['train_wall_seconds'] if mode=='plain_time' else None
                f=train_flow(c,ae,folder,pool,s,mode,device,budget);evaluate(c,f,di,s,device)
    summarize(c)


if __name__=='__main__':main()
