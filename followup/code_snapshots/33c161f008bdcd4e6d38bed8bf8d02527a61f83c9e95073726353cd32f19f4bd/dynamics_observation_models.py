"""Supervised four-frame RGB and RGB-measurement state/environment estimators."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import HERE
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_rgb_baseline import NAME as BASELINE


def inverse_softplus(x):return torch.log(torch.expm1(x.clamp_min(1e-4)))


class ObservationEstimator(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='rgb_cnn':
            self.encoder=nn.Sequential(nn.Conv2d(12,32,5,stride=2,padding=2),nn.SiLU(),
                nn.Conv2d(32,32,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),
                nn.Flatten(),nn.Linear(64*8*8,256),nn.SiLU())
            self.state_head=nn.Linear(256,6*6);self.context_head=nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,3))
            with torch.no_grad():
                self.state_head.bias.zero_();self.state_head.bias.reshape(6,6)[:,4]=inverse_softplus(torch.tensor(5/8))
                self.state_head.bias.reshape(6,6)[:,5]=float(np.log(.5))
                self.context_head[-1].bias.zero_();self.context_head[-1].bias[2]=inverse_softplus(torch.tensor(1/3))
        elif kind=='measurement_mlp':
            self.encoder=nn.Sequential(nn.Linear(43,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
            self.state_head=nn.Linear(128,6);self.context_head=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))
            for layer in [self.state_head,self.context_head[-1]]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        else:raise ValueError(kind)

    def forward(self,inputs):
        if self.kind=='rgb_cnn':
            rgb=inputs['images'];h=self.encoder(rgb.reshape(len(rgb),12,64,64))
            state=self.state_head(h).reshape(-1,6,6);context=self.context_head(h)
        else:
            h=self.encoder(inputs['features']);base=inputs['base_state'];state=self.state_head(h)
            state=state+torch.cat([base[...,:4],inverse_softplus(base[...,4:5]),base[...,5:6]],-1)
            presence=(base[...,5]>0).to(h.dtype);pooled=(h*presence[...,None]).sum(1)/presence.sum(1,keepdim=True).clamp_min(1)
            baseline=inputs['base_context'];context=self.context_head(pooled)+torch.cat([baseline[:,:2],inverse_softplus(baseline[:,2:3])],-1)
        state=torch.cat([state[...,:4],F.softplus(state[...,4:5]),state[...,5:6]],-1)
        context=torch.cat([context[:,:2],F.softplus(context[:,2:3])],-1)
        return {'state':state,'context':context}


def measurement_inputs(features,states,contexts):
    f=torch.tensor(features,dtype=torch.float32);s=torch.tensor(states,dtype=torch.float32);c=torch.tensor(contexts,dtype=torch.float32)
    # Fixed physical scales; no held-out or label-derived normalization.
    normalized=f/f.new_tensor([31.5,31.5,8.,1.,128.,.02,1.])
    delta=(f[:,1:,:,:2]-f[:,:-1,:,:2])/3
    context=(c/c.new_tensor([.1,.1,.01]));context=torch.cat([context[:,:2].clamp(-2,2),context[:,2:3].clamp(0,5)],-1)
    own=torch.cat([normalized.permute(0,2,1,3).flatten(2).clamp(-5,5),delta.permute(0,2,1,3).flatten(2).clamp(-5,5),
        context[:,None,:].expand(-1,6,-1),torch.eye(6)[None].expand(len(f),-1,-1)],-1)
    alive=s[...,4]>0;base=s[...,:5]/s.new_tensor([31.5,31.5,3,3,8]);base[...,2:4]=base[...,2:4].clamp(-2,2)
    logit=torch.where(alive,torch.full_like(s[...,0],float(np.log(99))),torch.full_like(s[...,0],-float(np.log(99))))
    return {'features':own,'base_state':torch.cat([base,logit[...,None]],-1),'base_context':context}


def load_inputs(mode,split,kind):
    if kind=='rgb_cnn':
        with np.load(HERE/f'data/{INPUTS}/{mode}/{split}/observations.npz') as a:
            assert a.files==['images'];x=torch.from_numpy(a['images'].copy()).permute(0,1,4,2,3).float()/255
        return {'images':x}
    with np.load(HERE/f'data/{BASELINE}/{mode}/{split}/predictions.npz') as a:
        return measurement_inputs(a['features'],a['states'],a['contexts'])


def load_labels(mode,split):
    with np.load(HERE/f'data/{INPUTS}/{mode}/{split}/labels.npz') as a:
        state=a['past_states'][:,-1].copy();context=np.concatenate([a['net_acceleration'],a['drag'][:,None]],-1)
    out=np.zeros((len(state),6,6),np.float32)
    for i,ss in enumerate(state):
        for s in ss:
            if s[4]>0:out[i,int(s[5])]=[*s[:5],1]
    target=torch.from_numpy(out);target[...,:5]/=target.new_tensor([31.5,31.5,3.,3.,8.])
    return {'state':target,'context':torch.tensor(context,dtype=torch.float32)/torch.tensor([.1,.1,.01])}


def supervised_loss(pred,labels):
    present=labels['state'][...,5];state=pred['state']
    factors=((state[...,:5]-labels['state'][...,:5]).square()*present[...,None]).sum()/(5*present.sum())
    presence=F.binary_cross_entropy_with_logits(state[...,5],present)
    context=F.mse_loss(pred['context'],labels['context'])
    return factors+.2*presence+.2*context,{'factor_mse':factors.detach(),'presence_bce':presence.detach(),'context_mse':context.detach()}


def decode(pred):
    normalized=pred['state'];presence=normalized[...,5]>=0
    factors=normalized[...,:5]*normalized.new_tensor([31.5,31.5,3.,3.,8.]);colors=torch.arange(6,device=factors.device)[None,:,None].expand(len(factors),-1,-1)
    state=torch.cat([factors,colors,presence[...,None].to(factors.dtype)],-1)*presence[...,None]
    context=pred['context']*normalized.new_tensor([.1,.1,.01])
    return state,context
