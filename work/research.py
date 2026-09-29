"""Reproducible research additions to the preserved OrbitLab starter.

Run from project root. Data and run directories are immutable after completion.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, resource, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import orbitlab as o

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def orbit_hash(a):
    return min(hashlib.sha256(np.ascontiguousarray(np.rot90(a,k)).tobytes()).hexdigest() for k in range(4))

def dump(path,obj): o.dump_json(Path(path),obj)

def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
    if (DATA/'manifest.json').exists():
        return json.loads((DATA/'manifest.json').read_text())
    DATA.mkdir(exist_ok=True)
    seen=set(); manifest={'data_seed':42,'size':64,'split_order':['train','val','test','ood'],'splits':{}}
    for split,n in [('train',4096),('val',256),('test',256),('ood',256)]:
        arrays=[]; meta=[]; rejected=0; i=0
        while len(arrays)<n:
            a,label,info=o.render_base(i,42,64,split);i+=1;h=orbit_hash(a)
            if h in seen: rejected+=1;continue
            seen.add(h);info.update(canonical_sha256=hashlib.sha256(a.tobytes()).hexdigest(),orbit_sha256=h)
            arrays.append(a);meta.append(info)
        np.savez_compressed(DATA/f'{split}.npz',images=np.stack(arrays),labels=np.array([[m['kind'],m['color']] for m in meta]))
        dump(DATA/f'{split}_metadata.json',meta)
        counts={f'{k},{c}':sum(m['kind']==k and m['color']==c for m in meta) for k in range(4) for c in range(6)}
        manifest['splits'][split]={'n':n,'candidates':i,'duplicates_rejected':rejected,'pair_counts':counts,'archive_sha256':sha(DATA/f'{split}.npz'),'metadata_sha256':sha(DATA/f'{split}_metadata.json')}
    manifest['all_orbits_unique']=len(seen)==4864
    dump(DATA/'manifest.json',manifest)
    x,_,_=load_data('train',32);o.grid(x,DATA/'input_32.png',8)
    o.grid(torch.cat([o.rotate(x[:8],k) for k in range(4)]),DATA/'c4_orbits.png',8)
    return manifest


def load_data(split,n=None):
    a=np.load(DATA/f'{split}.npz'); x=torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255
    y=torch.from_numpy(a['labels'].copy()).long();m=json.loads((DATA/f'{split}_metadata.json').read_text())
    if n is not None: return x[:n],y[:n],m[:n]
    return x,y,m


class ResearchAE(o.Autoencoder):
    def __init__(self,model='equivariant',width=16):
        super().__init__(64,model,16)
        if width!=16:
            latent=64 if model in ('plain','aug') else 16
            self.encoder.net=nn.Sequential(nn.Conv2d(3,width,4,2,1),nn.SiLU(),nn.Conv2d(width,2*width,4,2,1),nn.SiLU(),nn.Conv2d(2*width,2*width,4,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(2*width*8*8,latent))
            self.decoder.fc=nn.Linear(latent,2*width*8*8)
            self.decoder.net=nn.Sequential(nn.ConvTranspose2d(2*width,2*width,4,2,1),nn.SiLU(),nn.ConvTranspose2d(2*width,width,4,2,1),nn.SiLU(),nn.ConvTranspose2d(width,3,4,2,1),nn.Sigmoid())
            # Decoder.forward has its original channel count hard-coded: replace it.
            self.decoder.forward=lambda z:self.decoder.net(self.decoder.fc(z).reshape(-1,2*width,8,8))
        self.width=width


def load_model(path,dev):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    ae=ResearchAE(ck['model'],ck.get('width',16)).to(dev);ae.load_state_dict(ck['state_dict']);ae.eval()
    return ae,ck


def parameter_counts():
    target=sum(p.numel() for p in ResearchAE('aug').parameters())
    options=[(abs(sum(p.numel() for p in ResearchAE('equivariant',w).parameters())-target),w) for w in range(8,65)]
    width=min(options)[1]
    return {'aug':target,'equivariant':sum(p.numel() for p in ResearchAE().parameters()),'matched_eq_width':width,'matched_eq_parameters':sum(p.numel() for p in ResearchAE('equivariant',width).parameters())}


def pixel_metrics(pred,target):
    dims=(1,2,3);mask=target.amax(1,keepdim=True)>.05;pm=pred.amax(1,keepdim=True)>.20;tm=target.amax(1,keepdim=True)>.20
    mse=(pred-target).square().mean(dims)
    fg=((pred-target).abs()*mask).sum(dims)/(mask.sum(dims)*3).clamp_min(1)
    iou=(pm&tm).sum(dims)/(pm|tm).sum(dims).clamp_min(1)
    axis=torch.arange(target.shape[-1],device=target.device,dtype=torch.float32)
    centroids=[]
    for m in [pm,tm]:
        mass=m.sum(dims).clamp_min(1)
        cx=(m*axis[None,None,None,:]).sum(dims)/mass
        cy=(m*axis[None,None,:,None]).sum(dims)/mass
        centroids.append(torch.stack([cx,cy],1))
    centroid=(centroids[0]-centroids[1]).square().sum(1).sqrt()/target.shape[-1]
    return {'mse':mse,'psnr':-10*torch.log10(mse.clamp_min(1e-12)),'foreground_mae':fg,'mask_iou':iou,'centroid_error':centroid,'black_mse':target.square().mean(dims),'black_foreground_mae':(target*mask).sum(dims)/(mask.sum(dims)*3).clamp_min(1)}


def bootstrap(values,seed=9123):
    a=np.asarray(values);rng=np.random.default_rng(seed)
    b=a[rng.integers(len(a),size=(1000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'base_scene_ci95':[float(x) for x in np.quantile(b,[.025,.975])]}


@torch.no_grad()
def evaluate(ae,out,splits=('val','test','ood'),n=256,save_latents=False):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);dev=next(ae.parameters()).device
    summary={}
    for split in splits:
        x,y,meta=load_data(split,n);rows=[];zs=[];max_eq=0.;max_dec=0.
        for k in range(4):
            for i in range(0,len(x),32):
                b=o.rotate(x[i:i+32].to(dev),k);z=ae.encode(b);p=ae.decode(z);zs.append(z.cpu().numpy())
                metrics={key:v.cpu().numpy() for key,v in pixel_metrics(p,b).items()}
                zrot=ae.encode(o.rotate(b,1));max_eq=max(max_eq,float((zrot-o.rho(z,1)).abs().max()))
                max_dec=max(max_dec,float((ae.decode(o.rho(z,1))-o.rotate(p,1)).abs().max()))
                for j in range(len(b)):
                    row={'base_id':meta[i+j]['base_id'],'rotation':k,'shape':int(y[i+j,0]),'color':int(y[i+j,1])}
                    row.update({key:float(v[j]) for key,v in metrics.items()});rows.append(row)
        write_csv(out/f'{split}_samples.csv',rows)
        summary[split]={key:bootstrap(np.array([r[key] for r in rows]).reshape(4,len(x)).mean(0)) for key in metrics}
        summary[split]['encoder_max_abs']=max_eq;summary[split]['decoder_max_abs']=max_dec
        summary[split]['fixed_rho_meaningful']=ae.model in ('equivariant','invariant')
        if save_latents:
            np.savez_compressed(out/f'{split}_latents.npz',z=np.concatenate(zs),labels=np.tile(y.numpy(),(4,1)),rotation=np.repeat(np.arange(4),len(x)))
            dump(out/f'{split}_manifest.json',{'base_scenes':meta})
        if split in ('val','test'):
            b=x[:8].to(dev);p=ae(b)
            o.grid(torch.cat([b,p]),out/f'{split}_reconstruction.png',8)
            # Deterministic highest-error cases, not hand-selected successes.
            avg=np.array([r['mse'] for r in rows]).reshape(4,len(x)).mean(0)
            idx=np.argsort(avg)[-8:][::-1].copy();b=x[idx].to(dev)
            o.grid(torch.cat([b,ae(b)]),out/f'{split}_worst_reconstruction.png',8)
    dump(out/'metrics.json',summary);return summary


def train(args):
    prepare();out=o.new_output(args.out);dev=o.choose_device(args.device);o.seed_all(args.seed)
    x,_,meta=load_data('train',args.n);ae=ResearchAE(args.model,args.width).to(dev)
    opt=torch.optim.AdamW(ae.parameters(),lr=args.lr,weight_decay=1e-4)
    # The sampled scenes and rotations are independent of model parameter initialization.
    rng=torch.Generator().manual_seed(100000+args.seed);logs=[];o.sync(dev);started=time.perf_counter();step=0
    while step<args.steps:
        idx=torch.randint(len(x),(32,),generator=rng);batch=x[idx].to(dev)
        k=int(torch.randint(4,(1,),generator=rng))
        if args.model!='plain':batch=o.rotate(batch,k)
        opt.zero_grad(set_to_none=True);loss=o.reconstruction_loss(ae(batch),batch)
        if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(ae.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('non-finite gradient')
        opt.step();step+=1
        if step%100==0 or step==args.steps:
            o.sync(dev);elapsed=time.perf_counter()-started
            logs.append({'step':step,'loss':loss.item(),'seconds':elapsed})
            print(f'{out.name} step={step} loss={loss.item():.6f} seconds={elapsed:.2f}',flush=True)
        if args.seconds:
            o.sync(dev)
            if time.perf_counter()-started>=args.seconds:break
    o.sync(dev);elapsed=time.perf_counter()-started;ae.eval()
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ck={'state_dict':{k:v.cpu() for k,v in ae.state_dict().items()},'model':args.model,'width':args.width,'channels':16,'size':64,'seed':args.seed,'data_seed':42,'n':args.n,'steps':step,'optimizer':opt.state_dict(),'rng_sampling':rng.get_state(),'rng_torch':torch.get_rng_state(),'data_manifest_sha256':sha(DATA/'manifest.json'),'selection':'fixed final step, no test selection'}
    torch.save(ck,out/'ae.pt.tmp');(out/'ae.pt.tmp').rename(out/'ae.pt')
    run={'environment':o.environment(),'arguments':vars(args),'parameters':sum(p.numel() for p in ae.parameters()),'steps':step,'external_sample_presentations':step*32,'internal_view_factor':1 if args.model in ('plain','aug') else 4,'train_wall_seconds':elapsed,'peak_process_rss_bytes':peak,'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory() if dev.type=='mps' else None,'loss_log':logs,'code_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'checkpoint_sha256':sha(out/'ae.pt'),'data_manifest_sha256':sha(DATA/'manifest.json')}
    dump(out/'run.json',run)
    # Calibration uses only validation; final analysis is a separate command.
    run['validation']=evaluate(ae,out/'validation',splits=('val',));dump(out/'run.json',run)


@torch.no_grad()
def analyze(args):
    ae,ck=load_model(args.ae,o.choose_device(args.device));out=o.new_output(args.out)
    summary=evaluate(ae,out,splits=('train','val','test','ood'),n=args.n,save_latents=True)
    x,_,_=load_data('test',8);dev=next(ae.parameters()).device;b=x.to(dev);z=ae.encode(b)
    rows=[b,ae.decode(z),ae.decode(o.rho(z,1)),o.rotate(b,1)]
    o.grid(torch.cat(rows),out/'intervention.png',8)
    if ae.model in ('equivariant','invariant'):
        interventions={}
        for name,keep in [('m0',[0]),('remove_m0',[1,2,3]),('remove_m13',[0,2]),('remove_m2',[0,1,3]),('double_m13',None)]:
            all_m=[];all_energies=[];pictures=[]
            full,_,_=load_data('test',args.n)
            for i in range(0,len(full),32):
                target=full[i:i+32].to(dev);latent=ae.encode(target);f=np.fft.fft(latent.cpu().numpy(),axis=1,norm='ortho');all_energies.append((abs(f)**2).mean(2))
                if keep is None:f[:,[1,3]]*=2
                else:f[:,[k for k in range(4) if k not in keep]]=0
                inv=np.fft.ifft(f,axis=1,norm='ortho');assert np.max(abs(inv.imag))<1e-5
                edited=torch.from_numpy(inv.real.copy()).to(dev,dtype=torch.float32);pred=ae.decode(edited)
                all_m.extend([{k:float(v[j]) for k,v in pixel_metrics(pred,target).items()} for j in range(len(pred))])
                if i==0:pictures=[target[:8],ae.decode(latent[:8]),pred[:8]]
            interventions[name]={k:bootstrap([r[k] for r in all_m]) for k in all_m[0]}
            o.grid(torch.cat(pictures),out/f'band_{name}.png',8)
        dump(out/'band_interventions.json',interventions)
        dump(out/'band_energy.json',{'test_canonical_mean':np.concatenate(all_energies).mean(0).tolist()})
    return summary


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('prepare');sp.add_parser('counts')
    t=sp.add_parser('train');t.add_argument('--model',choices=['plain','aug','equivariant','invariant'],required=True);t.add_argument('--width',type=int,default=16);t.add_argument('--n',type=int,default=1024);t.add_argument('--steps',type=int,default=1000);t.add_argument('--seed',type=int,default=0);t.add_argument('--lr',type=float,default=.001);t.add_argument('--seconds',type=float,default=0)
    a=sp.add_parser('analyze');a.add_argument('--ae',required=True);a.add_argument('--n',type=int,default=256)
    for sub in [t,a]:sub.add_argument('--device',default='mps',choices=['mps','cpu']);sub.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    if args.command=='prepare':print(json.dumps(prepare(),indent=2))
    elif args.command=='counts':print(json.dumps(parameter_counts(),indent=2))
    elif args.command=='train':train(args)
    elif args.command=='analyze':analyze(args)

if __name__=='__main__':main()
