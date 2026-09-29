"""Second development pilot: linear RGB decoder and balanced RGB-derived loss.

Preserves failed v1 checkpoints and source. Two changes are bundled as a
stabilization attempt; this pilot does not isolate their individual effects.
"""
import json
import time
import torch.nn.functional as F
from torch import nn
from common import HERE, dump, fresh_dir, digest_tensor, state_digest, update_status, np, torch, o, r
from object_perception import ImageSlots, images_only, segmentation_metrics


class LinearImageSlots(ImageSlots):
    def forward(self,x,epsilon):
        feature=self.encoder(x).permute(0,2,3,1)+self.encoder_position(self.encoder_grid)
        feature=self.feature_mlp(self.feature_norm(feature))
        if self.kind=='slot':z,attention=self.router(feature.reshape(len(x),-1,self.dim),epsilon)
        else:
            pooled=F.adaptive_avg_pool2d(feature.permute(0,3,1,2),(4,4))
            z=self.router(pooled.flatten(1)).reshape(len(x),self.slots,self.dim);attention=None
        broadcast=z.reshape(-1,1,1,self.dim)+self.decoder_position(self.decoder_grid)[None]
        raw=self.decoder(broadcast.permute(0,3,1,2)).reshape(len(x),self.slots,4,64,64)
        rgb=raw[:,:,:3];masks=raw[:,:,3:4].softmax(1)
        return {'image':(rgb*masks).sum(1),'rgb':rgb,'masks':masks,'slots':z,'attention':attention}


def balanced_loss(pred,target):
    fg=(target.amax(1,keepdim=True)>.05).to(target.dtype);error=(pred-target).square();dims=(1,2,3)
    foreground=(error*fg).sum(dims)/(3*fg.sum(dims)).clamp_min(1)
    background=(error*(1-fg)).sum(dims)/(3*(1-fg).sum(dims)).clamp_min(1)
    return ((foreground+background)/2).mean()


def freeze():
    cp=HERE/'configs/object_perception_pilot_v2.json'
    if not cp.exists():dump(cp,{'kinds':['flat','slot'],'seed':0,'steps':3000,'batch':16,'lr':.0004,'warmup_steps':100,'slots':5,'dim':32,
        'data_manifest_sha256':r.sha(HERE/'data/object_world_v1/manifest.json'),
        'previous_pilot_sha256':r.sha(HERE/'reports/object_perception_pilot_v1/summary.json'),
        'changes':['Linear RGB outputs before mask mixture, as in reference model.','Mean foreground and background RGB error receive equal weight per image.'],
        'rationale':'V1 converged to all-black images in both models: validation foreground IoU zero. Preserve and report those results.',
        'information':'Training sees RGB only; foreground weight is derived from input brightness, not object labels/masks.',
        'comparison':'Same fixed steps, images, sampling and encoder/decoder per model; parameter and wall time budgets differ.',
        'boundary':'One-seed validation-only development; two stabilization changes together, not isolated causal effects.',
        'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def train(c,kind,device):
    folder=HERE/f'runs/object_perception_pilot_v2/{kind}_s0'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert run['checkpoint_sha256']==r.sha(folder/'model.pt');return folder
    fresh_dir(folder);x=images_only('train');o.seed_all(985000);model=LinearImageSlots(kind,c['dim'],c['slots']).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(986000);noise_rng=torch.Generator().manual_seed(987000)
    initial=state_digest(model);dump(folder/'started.json',{'kind':kind,'code_sha256':r.sha(__file__),'config_sha256':r.sha(HERE/'configs/object_perception_pilot_v2.json')})
    logs=[];o.sync(device);start=time.perf_counter()
    for step in range(1,c['steps']+1):
        idx=torch.randint(len(x),(c['batch'],),generator=rng);target=x[idx].to(device)
        eps=torch.randn(c['batch'],c['slots'],c['dim'],generator=noise_rng).to(device)
        for group in opt.param_groups:group['lr']=c['lr']*min(1,step/c['warmup_steps'])
        pred=model(target,eps);loss=balanced_loss(pred['image'],target)
        if not torch.isfinite(loss):raise FloatingPointError('Perception v2 loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Perception v2 gradient')
        opt.step()
        if step%250==0:
            o.sync(device);logs.append({'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-start});print(folder.name,logs[-1],flush=True)
    o.sync(device);seconds=time.perf_counter()-start
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'kind':kind,'dim':c['dim'],'slots':c['slots'],'rgb':'linear'},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state(),'noise_rng':noise_rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'kind':kind,'steps':c['steps'],'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':seconds,
        'training_image_sha256':digest_tensor(x),'initial_weights_sha256':initial,'sampling_rng_sha256':digest_tensor(rng.get_state()),
        'checkpoint_sha256':r.sha(folder/'model.pt'),'code_sha256':r.sha(__file__),'base_module_sha256':r.sha(HERE/'object_perception.py'),
        'config_sha256':r.sha(HERE/'configs/object_perception_pilot_v2.json'),'loss_log':logs})
    return folder


@torch.no_grad()
def evaluate(folder,c,device):
    if (folder/'metrics.json').exists():return
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=LinearImageSlots(ck['kind'],ck['dim'],ck['slots']).to(device)
    model.load_state_dict(ck['state_dict']);model.eval();x=images_only('val')
    with np.load(HERE/'data/object_world_v1/val/scenes.npz') as archive:gt=archive['segmentation'].copy()
    eps=torch.randn(len(x),c['slots'],c['dim'],generator=torch.Generator().manual_seed(988000));rows=[];images=[];segments=[];states=[]
    for i in range(0,len(x),16):
        b=x[i:i+16].to(device);out=model(b,eps[i:i+16].to(device));pm=r.pixel_metrics(out['image'],b)
        seg=out['masks'][:,:,0].argmax(1).cpu();images.append(out['image'].cpu());segments.append(seg);states.append(out['slots'].cpu())
        for j,labels in enumerate(seg.numpy()):rows.append({'base_id':i+j,**segmentation_metrics(gt[i+j],labels),**{k:float(v[j]) for k,v in pm.items()}})
    pred=torch.cat(images);seg=torch.cat(segments);keys=['mse','foreground_mae','mask_iou','foreground_ari','all_pixel_ari','matched_visible_iou']
    summary={k:r.bootstrap([v[k] for v in rows]) for k in keys};r.write_csv(folder/'val_rows.csv',rows)
    palette=torch.tensor([[.2,.2,.2],[1.,.2,.2],[.2,1.,.2],[.2,.3,1.],[1.,1.,.2]]);examples=[]
    for i in range(12):examples.extend([x[i],pred[i],palette[seg[i]].permute(2,0,1)])
    o.grid(torch.stack(examples),folder/'val_examples.png',6)
    torch.save({'val':{'images':pred,'segmentation':seg,'slots':torch.cat(states),'epsilon':eps}},folder/'evaluation.pt')
    dump(folder/'metrics.json',{'splits':{'val':summary},'raw_prediction_range':[float(pred.min()),float(pred.max())],
        'checkpoint_sha256':r.sha(folder/'model.pt'),'evaluation_sha256':r.sha(folder/'evaluation.pt')})
    print('perception v2',folder.name,{k:round(v['mean'],4) for k,v in summary.items()},flush=True)


def main():
    c=freeze();torch.set_num_threads(4);device=o.choose_device('mps')
    update_status('B04','in_progress',['configs/object_perception_pilot_v2.json','reports/object_perception_pilot_v1/summary.json'],note='V1 collapsed to black outputs. V2 stabilization pilot running; full evaluation remains pending.')
    for kind in c['kinds']:evaluate(train(c,kind,device),c,device)
    rows={kind:{'run':json.loads((HERE/f'runs/object_perception_pilot_v2/{kind}_s0/run.json').read_text()),
        'metrics':json.loads((HERE/f'runs/object_perception_pilot_v2/{kind}_s0/metrics.json').read_text())} for kind in c['kinds']}
    for key in ['training_image_sha256','sampling_rng_sha256','steps']:assert len({v['run'][key] for v in rows.values()})==1
    dump(HERE/'reports/object_perception_pilot_v2/summary.json',{'results':rows,'claim_boundary':c['boundary']})


if __name__=='__main__':main()
