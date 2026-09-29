"""Fixed-capacity eight-position estimators, hiding pixels before preprocessing."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import HERE
from dynamics_observation_length_data import NAME
from dynamics_observation_length_reference import LENGTHS,infer_window,select_visible
from dynamics_observation_models import ObservationEstimator


class LengthEstimator(ObservationEstimator):
    def __init__(self,kind):
        super().__init__(kind)
        if kind=='rgb_cnn':self.encoder[0]=nn.Conv2d(32,32,5,stride=2,padding=2)
        else:self.encoder[0]=nn.Linear(87,128)

    def forward(self,inputs):
        if self.kind!='rgb_cnn':return super().forward(inputs)
        x=inputs['images'];assert x.shape[1:]==(8,4,64,64)
        h=self.encoder(x.reshape(len(x),32,64,64));state=self.state_head(h).reshape(-1,6,6);context=self.context_head(h)
        return {'state':torch.cat([state[...,:4],F.softplus(state[...,4:5]),state[...,5:6]],-1),
                'context':torch.cat([context[:,:2],F.softplus(context[:,2:3])],-1)}


def rgb_inputs(images,length):
    if length not in LENGTHS or images.shape[1:]!=(8,64,64,3):raise ValueError('Eight-frame image carrier expected')
    # Copy only observed RGB. Prefix pixels cannot influence a tensor operation.
    visible=np.zeros(images.shape,dtype=images.dtype);visible[:,8-length:]=images[:,8-length:]
    rgb=torch.from_numpy(visible).permute(0,1,4,2,3).float()/255
    available=rgb.new_zeros((len(rgb),8,1,64,64));available[:,8-length:]=1
    return {'images':torch.cat([rgb,available],dim=2)}


def measurement_inputs(features,length,states=None,contexts=None):
    if length not in LENGTHS or features.shape[1:]!=(8,6,7):raise ValueError('Eight-position feature carrier expected')
    if states is None or contexts is None:
        predictions=[infer_window(f) for f in select_visible(features,length)]
        states=np.stack([p[0] for p in predictions]);contexts=np.stack([p[1] for p in predictions])
    selected=np.zeros_like(features);selected[:,8-length:]=features[:,8-length:]
    f=torch.tensor(selected,dtype=torch.float32);s=torch.tensor(states,dtype=torch.float32);c=torch.tensor(contexts,dtype=torch.float32)
    available=f.new_zeros((len(f),8));available[:,8-length:]=1
    normalized=f/f.new_tensor([31.5,31.5,8.,1.,128.,.02,1.])
    delta=(f[:,1:,:,:2]-f[:,:-1,:,:2])/3;delta*=((available[:,1:]*available[:,:-1])[:,:,None,None])
    ctx=c/c.new_tensor([.1,.1,.01]);ctx=torch.cat([ctx[:,:2].clamp(-2,2),ctx[:,2:3].clamp(0,5)],-1)
    own=torch.cat([normalized.permute(0,2,1,3).flatten(2).clamp(-5,5),delta.permute(0,2,1,3).flatten(2).clamp(-5,5),
        available[:,None,:].expand(-1,6,-1),ctx[:,None,:].expand(-1,6,-1),torch.eye(6)[None].expand(len(f),-1,-1)],-1)
    assert own.shape[1:]==(6,87)
    alive=s[...,4]>0;base=s[...,:5]/s.new_tensor([31.5,31.5,3,3,8]);base[...,2:4]=base[...,2:4].clamp(-2,2)
    logit=torch.where(alive,torch.full_like(s[...,0],float(np.log(99))),torch.full_like(s[...,0],-float(np.log(99))))
    return {'features':own,'base_state':torch.cat([base,logit[...,None]],-1),'base_context':ctx}


def load_inputs(ds,mode,split,kind,length):
    root=HERE/f'data/{NAME}/d{ds}'
    if kind=='rgb_cnn':
        with np.load(root/f'inputs/{mode}/{split}/observations.npz') as a:
            assert a.files==['images'];return rgb_inputs(a['images'],length)
    with np.load(root/f'measurements/{mode}/{split}/predictions.npz') as a:
        return measurement_inputs(a['features'],length,a[f'states_L{length}'],a[f'contexts_L{length}'])


def load_labels(ds,mode,split):
    with np.load(HERE/f'data/{NAME}/d{ds}/inputs/{mode}/{split}/labels.npz') as a:
        state=a['past_states'][:,-1].copy();context=np.c_[a['net_acceleration'],a['drag']]
    out=np.zeros((len(state),6,6),np.float32)
    for i,scene in enumerate(state):
        for s in scene:
            if s[4]>0:out[i,int(s[5])]=[*s[:5],1]
    target=torch.from_numpy(out);target[...,:5]/=target.new_tensor([31.5,31.5,3,3,8])
    return {'state':target,'context':torch.tensor(context,dtype=torch.float32)/torch.tensor([.1,.1,.01])}


def pooled(ds,kind,split,length):
    xs=[load_inputs(ds,m,split,kind,length) for m in ['isotropic','fixed_gravity','variable_force']]
    ys=[load_labels(ds,m,split) for m in ['isotropic','fixed_gravity','variable_force']]
    return {k:torch.cat([x[k] for x in xs]) for k in xs[0]},{k:torch.cat([y[k] for y in ys]) for k in ys[0]}
