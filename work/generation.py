"""Frozen-AE conditional flow and independent renderer-state evaluation."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import orbitlab as o
import research as r


def state_features(images):
    a=images.detach().cpu().numpy();features=[]
    for image in a:
        mask=image.max(0)>.20;ys,xs=np.where(mask);mass=len(xs)
        if mass:
            crop=mask[ys.min():ys.max()+1,xs.min():xs.max()+1].astype(np.uint8)*255
            normalized=np.asarray(Image.fromarray(crop).resize((32,32),Image.Resampling.NEAREST))/255
            # Color chromaticity removes brightness without changing hue ordering.
            color=image[:,mask].mean(1);color=color/max(color.sum(),1e-8)
            cx=xs.mean()/64;cy=ys.mean()/64;scale=max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)/64
        else:normalized=np.zeros((32,32));color=np.zeros(3);cx=cy=scale=0.
        features.append((normalized,color,cx,cy,mass/4096,scale))
    return features


class StateEvaluator:
    """Fixed binary silhouette templates from an independent renderer namespace.

    Validated on unseen base scenes. Low template overlap is reported, not hidden.
    """
    def __init__(self):
        self.templates=[];self.kinds=[];counts=[0]*4;i=0
        while min(counts)<16:
            a,(k,c),_=o.render_base(i,99173,64,'train');i+=1
            if counts[k]>=16:continue
            x=torch.from_numpy(a.copy()).permute(2,0,1)[None].float()/255
            for rot in range(4):self.templates.append(state_features(o.rotate(x,rot))[0][0]);self.kinds.append(k)
            counts[k]+=1
        self.templates=np.stack(self.templates);self.kinds=np.array(self.kinds)
        colors=np.array(o.PALETTE,dtype=float);self.palette=colors/colors.sum(1,keepdims=True)

    def predict(self,images):
        features=state_features(images);rows=[]
        for mask,color,cx,cy,occupancy,scale in features:
            intersection=(self.templates*mask).sum((1,2));union=np.maximum(self.templates,mask).sum((1,2));scores=intersection/np.maximum(union,1)
            best=int(scores.argmax());kind=int(self.kinds[best]);col=int(((self.palette-color)**2).sum(1).argmin())
            rows.append({'predicted_shape':kind,'predicted_color':col,'template_iou':float(scores[best]),'cx':cx,'cy':cy,'occupancy':occupancy,'scale':scale,'nonempty':occupancy>0})
        return rows

    def validate(self):
        xs=[];ys=[];hashes=[]
        for split in ['test','ood']:
            for i in range(256):
                a,y,_=o.render_base(i,88217,64,split);xs.append(a);ys.append(y);hashes.append(r.orbit_hash(a))
        x=torch.from_numpy(np.stack(xs)).permute(0,3,1,2).float()/255
        y=np.array(ys);pred=[]
        for k in range(4):pred.extend(self.predict(o.rotate(x,k)))
        labels=np.tile(y,(4,1));confusion=np.zeros((4,4),dtype=int)
        for row,label in zip(pred,labels):confusion[label[0],row['predicted_shape']]+=1
        training={m['orbit_sha256'] for m in json.loads((r.DATA/'train_metadata.json').read_text())}
        report={'method':'independent renderer binary silhouette template IoU; nearest palette chromaticity','template_seed':99173,'evaluation_seed':88217,'base_scenes':512,'rotated_examples':2048,'train_orbit_overlap':len(set(hashes)&training),'shape_accuracy':float(np.mean([p['predicted_shape']==int(y[0]) for p,y in zip(pred,labels)])),'color_accuracy':float(np.mean([p['predicted_color']==int(y[1]) for p,y in zip(pred,labels)])),'shape_confusion':confusion.tolist(),'validity_threshold':.65,'note':'Validation is on clean rendered data; accuracy on blurred generated images is not guaranteed. Report template IoU, visual failure grids and conditional fidelity jointly.'}
        r.dump(r.ROOT/'reports/state_evaluator_validation.json',report)
        assert report['shape_accuracy']>=.95 and report['color_accuracy']>=.99
        assert not report['train_orbit_overlap']
        return report


@torch.no_grad()
def nearest_train(images,train,device):
    # Exact float32 pixel MSE against every train image and all four rotations.
    q=images.to(device).flatten(1);qm=q.square().mean(1);best=torch.full((len(q),),float('inf'),device=device);best_id=torch.zeros(len(q),dtype=torch.long,device=device);best_rot=best_id.clone()
    dim=q.shape[1]
    for k in range(4):
        for i in range(0,len(train),256):
            ref=o.rotate(train[i:i+256].to(device),k).flatten(1)
            distances=(qm[:,None]+ref.square().mean(1)[None,:]-2*q@ref.T/dim).clamp_min(0)
            d,idx=distances.min(1);take=d<best
            best=torch.where(take,d,best);best_id=torch.where(take,idx+i,best_id);best_rot=torch.where(take,torch.full_like(best_rot,k),best_rot)
    return best.cpu().numpy(),best_id.cpu().numpy(),best_rot.cpu().numpy()


def train(args):
    out=o.new_output(args.out);dev=o.choose_device(args.device);o.seed_all(args.seed)
    ae,ack=r.load_model(args.ae,dev);ae.requires_grad_(False);ae.eval()
    before={k:v.detach().cpu().clone() for k,v in ae.state_dict().items()}
    x,label,_=r.load_data('train',ack['n']);codes=[];labels=[]
    with torch.no_grad():
        for k in range(4):
            for i in range(0,len(x),64):
                codes.append(ae.encode(o.rotate(x[i:i+64].to(dev),k)).cpu());labels.append(label[i:i+64])
    z=torch.cat(codes);labels=torch.cat(labels)
    mean=z.mean((0,1),keepdim=True);std=z.std((0,1),keepdim=True,unbiased=False).clamp_min(.02);z=(z-mean)/std
    flow=o.Velocity(16,ack['model']=='equivariant').to(dev);opt=torch.optim.AdamW(flow.parameters(),lr=.001,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(200000+args.seed);logs=[];o.sync(dev);start=time.perf_counter();step=0
    while step<args.steps:
        idx=torch.randint(len(z),(128,),generator=rng);target=z[idx].to(dev);y=labels[idx].to(dev)
        noise=torch.randn(128,4,16,generator=rng).to(dev);t=torch.rand(128,generator=rng).to(dev)
        point=(1-t[:,None,None])*noise+t[:,None,None]*target;loss=F.mse_loss(flow(point,t,y),target-noise)
        if not torch.isfinite(loss):raise FloatingPointError('flow nonfinite')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('flow gradient nonfinite')
        opt.step();step+=1
        if step%200==0:
            o.sync(dev);logs.append({'step':step,'loss':loss.item(),'seconds':time.perf_counter()-start})
            print(out.name,logs[-1],flush=True)
        if args.seconds:
            o.sync(dev)
            if time.perf_counter()-start>=args.seconds:break
    o.sync(dev);elapsed=time.perf_counter()-start
    unchanged=all(torch.equal(v,ae.state_dict()[k].cpu()) for k,v in before.items());assert unchanged
    # No optimizer state is needed for inference; retain it separately for recovery.
    torch.save({'state_dict':flow.cpu().state_dict(),'mean':mean,'std':std,'ae_path':str(Path(args.ae).resolve()),'ae_sha256':r.sha(args.ae),'equivariant':flow.equivariant,'seed':args.seed},out/'flow.pt')
    torch.save({'optimizer':opt.state_dict(),'step':step,'sampling_rng':rng.get_state()},out/'optimizer.pt')
    r.dump(out/'run.json',{'args':vars(args),'environment':o.environment(),'ae_frozen_and_unchanged':unchanged,'ae_parameters':sum(p.numel() for p in ae.parameters()),'flow_parameters':sum(p.numel() for p in flow.parameters()),'internal_velocity_evaluations_per_step':4 if flow.equivariant else 1,'steps':step,'train_wall_seconds':elapsed,'loss_log':logs,'data_manifest_sha256':r.sha(r.DATA/'manifest.json'),'code_sha256':r.sha(__file__)})


@torch.no_grad()
def sample_evaluate(args):
    out=o.new_output(args.out);dev=o.choose_device(args.device);ck=torch.load(args.flow,map_location='cpu',weights_only=True)
    assert r.sha(ck['ae_path'])==ck['ae_sha256'];ae,ack=r.load_model(ck['ae_path'],dev)
    flow=o.Velocity(16,ck['equivariant']).to(dev);flow.load_state_dict(ck['state_dict']);flow.eval()
    ev=StateEvaluator();train,train_labels,_=r.load_data('train',ack['n'])
    rng=torch.Generator().manual_seed(args.seed);noise=torch.randn(args.n,4,16,generator=rng)
    labels=torch.tensor([[i%4,(i//4)%6] for i in range(args.n)])
    results={};states={};all_latents={}
    for steps in [0,16,32,64]:
        o.sync(dev);start=time.perf_counter();images=[];latents=[]
        for start_i in range(0,args.n,128):
            z=noise[start_i:start_i+128].to(dev).clone();y=labels[start_i:start_i+128].to(dev)
            for tstep in range(steps):z=z+flow(z,torch.full((len(z),),tstep/steps,device=dev),y)/steps
            raw=z*ck['std'].to(dev)+ck['mean'].to(dev);images.append(ae.decode(raw).cpu());latents.append(raw.cpu())
        o.sync(dev);elapsed=time.perf_counter()-start;images=torch.cat(images);latents=torch.cat(latents);all_latents[steps]=latents
        predictions=ev.predict(images);distances=[];ids=[];rots=[]
        for i in range(0,len(images),64):
            d,j,k=nearest_train(images[i:i+64],train,dev);distances.extend(d);ids.extend(j);rots.extend(k)
        rows=[]
        for i,(pred,label) in enumerate(zip(predictions,labels.tolist())):
            row={'sample':i,'shape':label[0],'color':label[1],'ood':label[0]==label[1],**pred,'shape_correct':pred['predicted_shape']==label[0],'color_correct':pred['predicted_color']==label[1],'nn_mse':float(distances[i]),'nn_train_index':int(ids[i]),'nn_rotation':int(rots[i])};row['joint_correct']=row['shape_correct'] and row['color_correct'];row['valid']=bool(row['nonempty'] and row['template_iou']>=.65);rows.append(row)
        r.write_csv(out/f'samples_{steps}.csv',rows);o.grid(images[:48],out/f'samples_{steps}.png',8)
        summary={}
        for name,selected in [('all',rows),('seen',[r for r in rows if not r['ood']]),('ood',[r for r in rows if r['ood']])]:
            summary[name]={'n':len(selected),**{key:r.bootstrap([float(row[key]) for row in selected]) for key in ['shape_correct','color_correct','joint_correct','valid','template_iou','nn_mse','occupancy']}}
            valid=[v for v in selected if v['valid'] and v['joint_correct']]
            # Coverage of a fixed 4x4 centroid grid in renderer's [.32,.68] region.
            bins=lambda row:(min(3,max(0,int((row['cx']-.32)/.36*4))),min(3,max(0,int((row['cy']-.32)/.36*4))))
            in_domain=[v for v in valid if .32<=v['cx']<=.68 and .32<=v['cy']<=.68]
            summary[name]['valid_correct_n']=len(valid);summary[name]['valid_correct_centroid_grid_coverage']=len({bins(v) for v in in_domain})/16
            summary[name]['valid_correct_centroid_outside_domain_fraction']=1-len(in_domain)/len(valid) if valid else None
            summary[name]['valid_correct_centroid_std']=[float(np.std([v[k] for v in valid])) if valid else None for k in ['cx','cy']]
            summary[name]['valid_correct_scale_std']=float(np.std([v['scale'] for v in valid])) if valid else None
        summary['sample_and_decode_seconds']=elapsed
        z=latents[:64].to(dev);summary['decoder_commutation_max_abs']=float((ae.decode(o.rho(z,1))-o.rotate(ae.decode(z),1)).abs().max())
        # Flow + normalization + ODE equivariance tested with coupled, rotated noise.
        eps=noise[:16].to(dev);y=labels[:16].to(dev);zz=eps.clone();zr=o.rho(eps,1).clone()
        for j in range(steps):
            t=torch.full((16,),j/steps,device=dev);zz+=flow(zz,t,y)/steps;zr+=flow(zr,t,y)/steps
        summary['coupled_ode_latent_max_abs']=float((zr-o.rho(zz,1)).abs().max());summary['fixed_rho_meaningful']=ck['equivariant']
        results[str(steps)]=summary;states[steps]=rows
        if steps==64:
            worst=np.argsort([v['template_iou'] for v in rows])[:16].copy();o.grid(images[worst],out/'lowest_template_fit.png',8)
            near=np.argsort(distances)[:8].copy();refs=torch.stack([o.rotate(train[int(ids[i])][None],int(rots[i]))[0] for i in near]);o.grid(torch.cat([images[near],refs]),out/'nearest_train_pairs.png',8)
            for i in range(4):
                frames=[Image.fromarray((ae.decode(o.rho(latents[i:i+1].to(dev),k))[0].cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)) for k in range(4)]
                frames[0].save(out/f'latent_rotation_{i}.gif',save_all=True,append_images=frames[1:],duration=500,loop=0)
    results['paired_latent_convergence_mse']={f'{a}_to_{b}':float((all_latents[a]-all_latents[b]).square().mean()) for a,b in [(16,32),(32,64)]}
    results['args']=vars(args);results['ae_sha256']=ck['ae_sha256'];results['noise_sha256']=__import__('hashlib').sha256(noise.numpy().tobytes()).hexdigest();results['flow_sha256']=r.sha(args.flow)
    r.dump(out/'metrics.json',results)
    torch.save({'noise':noise,'labels':labels,'latents':all_latents},out/'samples.pt')
    print('completed',out,flush=True)


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('validate-evaluator')
    t=sp.add_parser('train');t.add_argument('--ae',required=True);t.add_argument('--steps',type=int,default=4000);t.add_argument('--seconds',type=float,default=0)
    s=sp.add_parser('sample');s.add_argument('--flow',required=True);s.add_argument('--n',type=int,default=576)
    for q in [t,s]:q.add_argument('--device',default='mps');q.add_argument('--seed',type=int,default=0);q.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    if args.command=='validate-evaluator':print(StateEvaluator().validate())
    elif args.command=='train':train(args)
    else:sample_evaluate(args)
if __name__=='__main__':main()
