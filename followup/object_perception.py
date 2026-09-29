"""Small image-only Slot Attention/flat-slot pilot for OrbitLab.

Algorithm reference: Locatello et al., arXiv:2006.15055v2, Algorithm 1 and 2.2.
An independent PyTorch implementation for this 64px renderer, not a reproduction
of the paper's benchmark/architecture/budget. Ground-truth masks are evaluation
inputs only; optimization receives RGB reconstruction pairs exclusively.
"""
from __future__ import annotations
import argparse
import json
import time
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from common import HERE, dump, fresh_dir, digest_tensor, state_digest, update_status, np, torch, o, r


def position_grid(side):
    yy,xx=torch.meshgrid(torch.linspace(0,1,side),torch.linspace(0,1,side),indexing='ij')
    return torch.stack([xx,yy,1-xx,1-yy],-1)


class SlotAttention(nn.Module):
    def __init__(self,dim=32,iterations=3):
        super().__init__();self.dim=dim;self.iterations=iterations
        self.mu=nn.Parameter(torch.randn(1,1,dim)*.1);self.log_std=nn.Parameter(torch.zeros(1,1,dim))
        self.input_norm=nn.LayerNorm(dim);self.slot_norm=nn.LayerNorm(dim);self.mlp_norm=nn.LayerNorm(dim)
        self.key=nn.Linear(dim,dim,bias=False);self.value=nn.Linear(dim,dim,bias=False);self.query=nn.Linear(dim,dim,bias=False)
        self.gru=nn.GRUCell(dim,dim);self.mlp=nn.Sequential(nn.Linear(dim,dim*2),nn.ReLU(),nn.Linear(dim*2,dim))

    def forward(self,features,epsilon):
        features=self.input_norm(features);key=self.key(features);value=self.value(features)
        slots=self.mu+self.log_std.exp()*epsilon
        for _ in range(self.iterations):
            previous=slots;query=self.query(self.slot_norm(slots))*self.dim**-.5
            attention=(key@query.transpose(-1,-2)).softmax(-1)+1e-8
            weights=attention/attention.sum(1,keepdim=True)
            update=weights.transpose(-1,-2)@value
            slots=self.gru(update.reshape(-1,self.dim),previous.reshape(-1,self.dim)).reshape_as(previous)
            slots=slots+self.mlp(self.mlp_norm(slots))
        return slots,attention


class ImageSlots(nn.Module):
    def __init__(self,kind,dim=32,slots=5):
        super().__init__();self.kind=kind;self.dim=dim;self.slots=slots
        self.encoder=nn.Sequential(nn.Conv2d(3,dim,5,2,2),nn.ReLU(),nn.Conv2d(dim,dim,5,1,2),nn.ReLU(),nn.Conv2d(dim,dim,5,1,2),nn.ReLU())
        self.register_buffer('encoder_grid',position_grid(32));self.register_buffer('decoder_grid',position_grid(8))
        self.encoder_position=nn.Linear(4,dim);self.decoder_position=nn.Linear(4,dim)
        self.feature_norm=nn.LayerNorm(dim);self.feature_mlp=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,dim))
        if kind=='slot':self.router=SlotAttention(dim)
        elif kind=='flat':self.router=nn.Linear(dim*4*4,slots*dim)
        else:raise ValueError(kind)
        self.decoder=nn.Sequential(nn.ConvTranspose2d(dim,dim,5,2,2,output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(dim,dim,5,2,2,output_padding=1),nn.ReLU(),
            nn.ConvTranspose2d(dim,dim,5,2,2,output_padding=1),nn.ReLU(),nn.Conv2d(dim,4,3,1,1))

    def forward(self,x,epsilon):
        feature=self.encoder(x).permute(0,2,3,1)+self.encoder_position(self.encoder_grid)
        feature=self.feature_mlp(self.feature_norm(feature))
        if self.kind=='slot':z,attention=self.router(feature.reshape(len(x),-1,self.dim),epsilon)
        else:
            pooled=F.adaptive_avg_pool2d(feature.permute(0,3,1,2),(4,4)).flatten(1)
            z=self.router(pooled).reshape(len(x),self.slots,self.dim);attention=None
        broadcast=z.reshape(-1,1,1,self.dim)+self.decoder_position(self.decoder_grid)[None]
        raw=self.decoder(broadcast.permute(0,3,1,2)).reshape(len(x),self.slots,4,64,64)
        rgb=raw[:,:,:3].sigmoid();masks=raw[:,:,3:4].softmax(1);reconstruction=(rgb*masks).sum(1)
        return {'image':reconstruction,'rgb':rgb,'masks':masks,'slots':z,'attention':attention}


def segmentation_metrics(labels,predicted,n_slots=5):
    fg=labels>0;true_ids=np.unique(labels[fg]);iou=np.zeros((len(true_ids),n_slots))
    for i,g in enumerate(true_ids):
        a=labels==g
        for k in range(n_slots):
            b=predicted==k;iou[i,k]=(a&b).sum()/max(1,(a|b).sum())
    ri,ci=linear_sum_assignment(-iou)
    return {'foreground_ari':float(adjusted_rand_score(labels[fg],predicted[fg])),
        'all_pixel_ari':float(adjusted_rand_score(labels.reshape(-1),predicted.reshape(-1))),
        'matched_visible_iou':float(iou[ri,ci].mean()) if len(ri) else 1.,
        'visible_object_count':len(true_ids),'assignment':{str(int(true_ids[i])):int(j) for i,j in zip(ri,ci)}}


def images_only(split):
    with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as archive:
        # Do not materialize labels or masks on the training path.
        return torch.from_numpy(archive['images'].copy()).permute(0,3,1,2).float()/255


def freeze():
    cp=HERE/'configs/object_perception_pilot_v1.json'
    if not cp.exists():dump(cp,{'kinds':['flat','slot'],'seed':0,'steps':1000,'batch':16,'lr':.0004,'warmup_steps':100,
        'slots':5,'dim':32,'attention_iterations':3,'evaluation_splits':['val'],
        'data_manifest_sha256':r.sha(HERE/'data/object_world_v1/manifest.json'),
        'purpose':'Development pilot to measure runtime, optimize feasibility and inspect collapse. No B04 completion from this pilot.',
        'training_information':'Only RGB input as reconstruction target. Foreground loss weight derived from RGB intensity; no renderer masks or factors.',
        'loss':'Original OrbitLab image-weighted MSE: 1+4*(max target channel > .05). Not the unweighted paper loss.',
        'comparison':'Same encoder/decoder, slot capacity, training images and steps; routing differs. Flat model has more parameters; not matched time or parameter budget.',
        'selection':'Fixed final pilot step, validation only. Test and count/occlusion splits reserved for locked full study.',
        'reference':'https://arxiv.org/html/2006.15055v2','frozen_unix':time.time()})
    return json.loads(cp.read_text())


def train(c,kind,device):
    folder=HERE/f'runs/object_perception_pilot_v1/{kind}_s{c["seed"]}'
    if (folder/'run.json').exists():
        rr=json.loads((folder/'run.json').read_text());assert rr['checkpoint_sha256']==r.sha(folder/'model.pt');return folder
    fresh_dir(folder);x=images_only('train');o.seed_all(985000+c['seed']);model=ImageSlots(kind,c['dim'],c['slots']).to(device)
    initial=state_digest(model);opt=torch.optim.Adam(model.parameters(),lr=c['lr'])
    rng=torch.Generator().manual_seed(986000+c['seed']);noise_rng=torch.Generator().manual_seed(987000+c['seed'])
    dump(folder/'started.json',{'kind':kind,'config_sha256':r.sha(HERE/'configs/object_perception_pilot_v1.json'),'code_sha256':r.sha(__file__)})
    o.sync(device);start=time.perf_counter();logs=[]
    for step in range(1,c['steps']+1):
        idx=torch.randint(len(x),(c['batch'],),generator=rng);target=x[idx].to(device)
        epsilon=torch.randn(c['batch'],c['slots'],c['dim'],generator=noise_rng).to(device)
        for group in opt.param_groups:group['lr']=c['lr']*min(1,step/c['warmup_steps'])
        pred=model(target,epsilon);loss=o.reconstruction_loss(pred['image'],target)
        if not torch.isfinite(loss):raise FloatingPointError('Perception loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Perception gradient')
        opt.step()
        if step%100==0:
            o.sync(device);logs.append({'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-start});print(folder.name,logs[-1],flush=True)
    o.sync(device);seconds=time.perf_counter()-start
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'kind':kind,'dim':c['dim'],'slots':c['slots']},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state(),'noise_rng':noise_rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),'steps':c['steps'],
        'train_seconds':seconds,'training_image_sha256':digest_tensor(x),'initial_weights_sha256':initial,
        'sampling_rng_sha256':digest_tensor(rng.get_state()),'checkpoint_sha256':r.sha(folder/'model.pt'),
        'training_supervision':'RGB reconstruction only','code_sha256':r.sha(__file__),'config_sha256':r.sha(HERE/'configs/object_perception_pilot_v1.json'),'loss_log':logs})
    return folder


@torch.no_grad()
def evaluate(folder,c,device):
    if (folder/'metrics.json').exists():return
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=ImageSlots(ck['kind'],ck['dim'],ck['slots']).to(device)
    model.load_state_dict(ck['state_dict']);model.eval();results={};all_outputs={}
    palette=torch.tensor([[.2,.2,.2],[1.,.2,.2],[.2,1.,.2],[.2,.3,1.],[1.,1.,.2]])
    for split in c['evaluation_splits']:
        x=images_only(split)
        with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as archive:gt=archive['segmentation'].copy()
        rng=torch.Generator().manual_seed(988000);epsilon=torch.randn(len(x),c['slots'],c['dim'],generator=rng);rows=[];predictions=[];segmentations=[];slot_states=[]
        for i in range(0,len(x),16):
            b=x[i:i+16].to(device);out=model(b,epsilon[i:i+16].to(device));pm=r.pixel_metrics(out['image'],b)
            seg=out['masks'][:,:,0].argmax(1).cpu();predictions.append(out['image'].cpu());segmentations.append(seg);slot_states.append(out['slots'].cpu())
            for j,labels in enumerate(seg.numpy()):rows.append({'base_id':i+j,**segmentation_metrics(gt[i+j],labels),**{k:float(v[j]) for k,v in pm.items()}})
        pred=torch.cat(predictions);seg=torch.cat(segmentations)
        keys=['mse','foreground_mae','mask_iou','foreground_ari','all_pixel_ari','matched_visible_iou']
        results[split]={k:r.bootstrap([v[k] for v in rows]) for k in keys};r.write_csv(folder/f'{split}_rows.csv',rows)
        imgs=[]
        for i in range(12):imgs.extend([x[i],pred[i],palette[seg[i]].permute(2,0,1)])
        o.grid(torch.stack(imgs),folder/f'{split}_examples.png',6)
        all_outputs[split]={'images':pred,'segmentation':seg,'slots':torch.cat(slot_states),'epsilon':epsilon}
        print('perception',folder.name,split,{k:round(v['mean'],4) for k,v in results[split].items()},flush=True)
    torch.save(all_outputs,folder/'evaluation.pt')
    dump(folder/'metrics.json',{'splits':results,'checkpoint_sha256':r.sha(folder/'model.pt'),'evaluation_sha256':r.sha(folder/'evaluation.pt')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();c=freeze()
    if args.prepare_only:return
    torch.set_num_threads(4);device=o.choose_device('mps');update_status('B04','in_progress',['configs/object_perception_pilot_v1.json'],note='Image-only development pilot; full locked evaluation and intervention readout pending.')
    for kind in c['kinds']:evaluate(train(c,kind,device),c,device)
    rows={kind:{'run':json.loads((HERE/f'runs/object_perception_pilot_v1/{kind}_s0/run.json').read_text()),
                'metrics':json.loads((HERE/f'runs/object_perception_pilot_v1/{kind}_s0/metrics.json').read_text())} for kind in c['kinds']}
    for key in ['training_image_sha256','sampling_rng_sha256','steps']:assert len({v['run'][key] for v in rows.values()})==1
    dump(HERE/'reports/object_perception_pilot_v1/summary.json',{'results':rows,'claim_boundary':'Validation-only pilot, one seed, not final object discovery or downstream intervention result.'})


if __name__=='__main__':main()
