"""Supervised RGB object readers; masks are optional training labels only."""
import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import r2_world as w
from v3_common import HERE,sha,lock

ARMS=[f'{kind}_mask{mask}' for kind in ['detector','slot','recurrent'] for mask in [0,1]]

def freeze():
    return lock(w.BASE/'training_protocol.json',{
        'sources':{p:sha(HERE/p) for p in ['r2_core.py','r2_train.py']},'data_protocol_sha256':sha(w.BASE/'data_protocol.json'),
        'arms':ARMS,'steps':4000,'batch':16,'optimizer':'AdamW lr=.0005 weight_decay=.0001 clip1','initializations':[0,1,2],
        'capacity':'Six object slots; a separate background attention logit is not an object slot. No GT count, positions, colors, masks or IDs at inference.',
        'input':'Static readers: current window[0]. Recurrent reader: same four images in past-to-current order. Supervision only for current reference state and, in mask1 arms, current visible masks. No future images.',
        'supervision':'All arms supervised by current two-object states. Detector also has a center heatmap target derived from the same coordinate labels. mask1 adds visible-mask supervision. Therefore these are supervised perceptual readers, not unsupervised object discovery.',
        'matching':'Exhaustive ordered assignment of2 ground-truth objects to6 predictions, minimizing state+presence training loss. No true labels enter test inference.',
        'loss':'shapeCE + quotient poseCE for symmetric shape3 + radiusCE +4*RGBMSE +4*MSE(xy_pixels/8), plus presence BCE. Mask1 adds2*foreground-weighted visible-mask BCE. Detector adds center heatmap MSE weighted toward true centers.',
        'parameters_and_compute':'Same slot and recurrent parameters, but recurrent processes4 frames and costs more compute. Detector topology differs. Report parameters, updates, seconds,100-step timings; no equal-FLOP or isolated architecture claim.',
        'state_tolerances':'Each xy axis <=1px; radius exact5..8; shape exact; pose modulo2 only for shape3 else modulo4; RGB each channel <=20/255; presence probability>=.5.',
        'editing':'Click nearest predicted present center; same assignment retained through recolor,+90degree rotation,+5px x translation. Score target and non-target factors, count and joint success. Known renderer diagnostics separate from any learned pixel editing claims.',
        'uncertainty':'Shape/pose/radius smallest probability sets with cumulative mass>=.9; position andRGB symmetric intervals calibrated by95th percentile matched-object absolute errors on separate calibration split. Report set sizes,widths,marginal andjoint coverage plus misses. No formal OOD coverage guarantee; these are predictive sets, not exact identifiable-state sets.',
        'selection':'Compare recurrent to the better detector or slot static model with the SAME mask supervision. Choose recurrent arm maximizing50%-occlusion all-count joint edit improvement; tie mask0 first. Development gate: improvement>=.15, non-target preservation drop<=.02, count6 missing fraction<=.05. If fails, diagnose misses versus attribute errors without automatic3x3 expansion. If passes, fix baseline/candidate and confirm3fresh data x3initializations; every data seed must meet gate.',
        'decoder_followup':'Regardless of candidate gate, build and evaluate a learned decoder from matched current states to canonical foreground and contrast truth-state and inferred-state decoding. Full background/occluder image editing is a separate unresolved requirement until implemented and verified.',
        'caps':{'local_seconds':43200,'disk_bytes':30000000000},'device':'cpu','threads':2})

def grid(side):
    y,x=torch.meshgrid(torch.arange(side),torch.arange(side),indexing='ij')
    return torch.stack([(x+.5)*64/side-.5,(y+.5)*64/side-.5],-1).float()

class Backbone(nn.Module):
    def __init__(self,dim=48):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.SiLU(),nn.Conv2d(32,dim,5,2,2),nn.SiLU(),nn.Conv2d(dim,dim,3,1,1),nn.SiLU())
    def forward(self,x):return self.net(x)

class SlotReader(nn.Module):
    def __init__(self,recurrent=False,dim=48):
        super().__init__();self.recurrent=recurrent;self.dim=dim;self.backbone=Backbone(dim)
        self.position=nn.Linear(2,dim);self.register_buffer('xy_grid',grid(16).reshape(256,2))
        self.norm=nn.LayerNorm(dim);self.qnorm=nn.LayerNorm(dim)
        self.key=nn.Linear(dim,dim);self.value=nn.Linear(dim,dim);self.query=nn.Linear(dim,dim)
        self.bg=nn.Linear(dim,1);self.gru=nn.GRUCell(dim,dim);self.mlp=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,96),nn.SiLU(),nn.Linear(96,dim))
        self.initial=nn.Parameter(torch.randn(1,6,dim)*.1);self.head=nn.Sequential(nn.Linear(dim,96),nn.SiLU(),nn.Linear(96,18))
    def forward(self,window):
        order=[3,2,1,0] if self.recurrent else [0];b=len(window);slots=self.initial.expand(b,-1,-1)
        for t in order:
            features=self.backbone(window[:,t]).flatten(2).transpose(1,2)
            features=self.norm(features+self.position(self.xy_grid/63))
            key=self.key(features);value=self.value(features);bg=self.bg(features)
            for _ in range(3):
                query=self.query(self.qnorm(slots));logits=key@query.transpose(-1,-2)/self.dim**.5
                probabilities=torch.cat([logits,bg],-1).softmax(-1);attention=probabilities[...,:6]
                weights=(attention+1e-8)/(attention+1e-8).sum(1,keepdim=True)
                updates=weights.transpose(1,2)@value
                slots=self.gru(updates.reshape(-1,self.dim),slots.reshape(-1,self.dim)).reshape(b,6,self.dim)
                slots=slots+self.mlp(slots)
        raw=self.head(slots);xy=weights.transpose(1,2)@self.xy_grid
        raw=torch.cat([raw[...,:12],(xy+4*raw[...,12:14].tanh()).clamp(0,63),raw[...,14:]],-1)
        masks=F.interpolate(attention.transpose(1,2).reshape(b,6,16,16),size=64,mode='bilinear',align_corners=False)
        return {'raw':raw,'masks':masks}

class DetectorReader(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=Backbone();self.heatmap=nn.Conv2d(48,1,1);self.segment=nn.Conv2d(48,1,1)
        self.crop=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),nn.Conv2d(32,32,3,2,1),nn.SiLU(),nn.Flatten(),nn.Linear(800,96),nn.SiLU(),nn.Linear(96,17))
        yy,xx=torch.meshgrid(torch.arange(33)-16,torch.arange(33)-16,indexing='ij')
        self.register_buffer('crop_grid',torch.stack([xx,yy],-1).float())
        self.register_buffer('full_grid',grid(64))
    def forward(self,window):
        image=window[:,0];b=len(image);features=self.backbone(image)
        logits=F.interpolate(self.heatmap(features),size=64,mode='bilinear',align_corners=False)[:,0]
        peaks=logits.detach().clone();positions=[];scores=[]
        for _ in range(6):
            index=peaks.flatten(1).argmax(-1);point=torch.stack([index%64,index//64],-1).float()
            scores.append(logits.flatten(1).gather(1,index[:,None])[:,0]);positions.append(point)
            suppress=(self.full_grid[None]-point[:,None,None]).square().sum(-1)<=36
            peaks=peaks.masked_fill(suppress,-1000)
        xy=torch.stack(positions,1);values=torch.stack(scores,1)
        sampling=(self.crop_grid[None,None]+xy[:,:,None,None])/31.5-1
        patches=F.grid_sample(image[:,None].expand(-1,6,-1,-1,-1).reshape(b*6,3,64,64),sampling.reshape(b*6,33,33,2),align_corners=True,padding_mode='zeros')
        out=self.crop(patches).reshape(b,6,17)
        raw=torch.cat([out[...,:12],(xy+4*out[...,12:14].tanh()).clamp(0,63),out[...,14:17],values[...,None]],-1)
        # Auxiliary segmentation is class-agnostic; assigned visible masks below
        # are cropped by the model's predicted positions, never GT positions.
        fg=F.interpolate(self.segment(features),size=64,mode='bilinear',align_corners=False).sigmoid()
        full=grid(64).to(image.device);distance=(full[None,None]-xy[:,:,None,None]).square().sum(-1)
        masks=fg*(distance/(-2*8**2)).exp()
        return {'raw':raw,'masks':masks,'heatmap_logits':logits}

def model_for(arm):
    kind=arm.split('_')[0]
    return DetectorReader() if kind=='detector' else SlotReader(kind=='recurrent')

def attributes(raw,truth):
    # truth B,G,9: shape,pose,radius_index,x,y,r,g,b,presence.
    shape=truth[...,0].long();pose=truth[...,1].long();radius=truth[...,2].long()
    shape_cost=-raw[...,:4].log_softmax(-1)[:,None].expand(-1,len(truth[0]),-1,-1).gather(-1,shape[:,:,None,None].expand(-1,-1,6,1))[...,0]
    lp=raw[...,4:8].log_softmax(-1)[:,None].expand(-1,truth.shape[1],-1,-1)
    pose_cost=-lp.gather(-1,pose[:,:,None,None].expand(-1,-1,6,1))[...,0]
    a=lp.gather(-1,(pose%2)[:,:,None,None].expand(-1,-1,6,1))[...,0];b=lp.gather(-1,(pose%2+2)[:,:,None,None].expand(-1,-1,6,1))[...,0]
    pose_cost=torch.where((shape==3)[:,:,None],-torch.logaddexp(a,b),pose_cost)
    rad_cost=-raw[...,8:12].log_softmax(-1)[:,None].expand(-1,truth.shape[1],-1,-1).gather(-1,radius[:,:,None,None].expand(-1,-1,6,1))[...,0]
    xy=4*((raw[:,None,:,12:14]-truth[:,:,None,3:5])/8).square().mean(-1)
    rgb=4*(raw[:,None,:,14:17].sigmoid()-truth[:,:,None,5:8]).square().mean(-1)
    return shape_cost+pose_cost+rad_cost+xy+rgb

def training_loss(output,truth,masks,arm):
    raw=output['raw'];attrs=attributes(raw,truth);cost=attrs-(2/6)*raw[:,None,:,17]
    combos=torch.tensor(list(itertools.permutations(range(6),2)),device=raw.device)
    assignment=combos[(cost[:,0,combos[:,0]]+cost[:,1,combos[:,1]]).detach().argmin(-1)]
    presence=torch.zeros_like(raw[...,17]).scatter(1,assignment,1)
    loss=attrs.gather(2,assignment[...,None]).mean()+F.binary_cross_entropy_with_logits(raw[...,17],presence)
    if arm.endswith('mask1'):
        target=torch.zeros_like(output['masks'])
        target.scatter_(1,assignment[...,None,None].expand(-1,-1,64,64),masks)
        weights=1+8*target
        loss=loss+2*(F.binary_cross_entropy(output['masks'].clamp(1e-5,1-1e-5),target,reduction='none')*weights).sum()/weights.sum()
    if 'heatmap_logits' in output:
        xy=grid(64).to(raw.device)
        delta=xy[None,None]-truth[:,:,None,None,3:5]
        target=torch.exp(-delta.square().sum(-1)/(2*1.5**2)).amax(1)
        weights=1+50*target
        loss=loss+10*((output['heatmap_logits'].sigmoid()-target).square()*weights).mean()
    return loss

def labels_tensor(records):
    return torch.tensor([[[v['shape'],v['pose'],v['radius']-5,v['x'],v['y'],*(np.array(v['rgb'])/255),1.] for v in row['objects']] for row in records],dtype=torch.float32)
