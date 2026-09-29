"""State-oracle transition controls with explicit input/context rotation actions.

All predictors receive the same four padded object states, per-object actions,
and net acceleration plus drag. Object-only routing deliberately cannot use
neighbors. The three symmetry variants share the interaction network topology.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

STATE_DIM=12  # normalized xy, velocity, radius, six color indicators, presence


def encode_states(raw):
    raw=torch.as_tensor(raw,dtype=torch.float32)
    alive=(raw[...,4]>0).to(raw.dtype)
    color=F.one_hot(raw[...,5].long().clamp(0,5),6).to(raw.dtype)*alive[...,None]
    scale=raw.new_tensor([31.5,31.5,3.,3.,8.])
    return torch.cat([raw[...,:5]/scale,color,alive[...,None]],-1)


def encode_context(raw):
    raw=torch.as_tensor(raw,dtype=torch.float32)
    return torch.cat([(raw[...,:2]+raw[...,2:4])/.1,raw[...,4:5]/.01],-1)


def rotate_vec(v,k):
    k%=4
    if k==0:return v
    if k==1:return torch.stack([v[...,1],-v[...,0]],-1)
    if k==2:return -v
    return torch.stack([-v[...,1],v[...,0]],-1)


def rotate_state(s,k):
    return torch.cat([rotate_vec(s[...,:2],k),rotate_vec(s[...,2:4],k),s[...,4:]],-1)


def rotate_context(c,k):
    return torch.cat([rotate_vec(c[...,:2],k),c[...,2:]],-1)


def rotate_motion(delta,k):
    return torch.cat([rotate_vec(delta[...,:2],k),rotate_vec(delta[...,2:],k)],-1)


def inertial_next(s,action):
    velocity=s[...,2:4]+action
    motion=torch.cat([s[...,:2]+velocity*(3/31.5),velocity],-1)
    return motion*s[...,-1:]


class Transition(nn.Module):
    def __init__(self,kind,hidden=48):
        super().__init__();self.kind=kind;self.hidden=hidden
        if kind=='flat':
            self.net=nn.Sequential(nn.Linear(4*(STATE_DIM+2)+3,hidden*2),nn.SiLU(),
                nn.Linear(hidden*2,hidden*2),nn.SiLU(),nn.Linear(hidden*2,16))
        elif kind=='object':
            self.net=nn.Sequential(nn.Linear(STATE_DIM+5,hidden*2),nn.SiLU(),
                nn.Linear(hidden*2,hidden*2),nn.SiLU(),nn.Linear(hidden*2,4))
        elif kind in ['interaction','augmented','wrong_exact','joint_exact','relaxed']:
            self.edge=nn.Sequential(nn.Linear(2*(STATE_DIM+2)+3,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
            self.net=nn.Sequential(nn.Linear(STATE_DIM+5+hidden,hidden*2),nn.SiLU(),
                nn.Linear(hidden*2,hidden),nn.SiLU(),nn.Linear(hidden,4))
            if kind=='relaxed':self.mix_logit=nn.Parameter(torch.tensor(0.))
        else:raise ValueError(kind)
        # Every trainable model starts at the same inertial motion predictor.
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def raw(self,s,action,context):
        b,n,_=s.shape;sa=torch.cat([s,action],-1);expanded=context[:,None,:].expand(-1,n,-1)
        if self.kind=='flat':return self.net(torch.cat([sa.flatten(1),context],-1)).reshape(b,n,4)
        if self.kind=='object':return self.net(torch.cat([sa,expanded],-1))
        left=sa[:,:,None,:].expand(-1,-1,n,-1);right=sa[:,None,:,:].expand(-1,n,-1,-1)
        ctx=context[:,None,None,:].expand(-1,n,n,-1)
        edges=self.edge(torch.cat([left,right,ctx],-1))
        live=s[...,-1];mask=live[:,:,None]*live[:,None,:]*(1-torch.eye(n,device=s.device)[None])
        messages=(edges*mask[...,None]).sum(2)
        return self.net(torch.cat([sa,expanded,messages],-1))

    def symmetrized(self,s,action,context,joint):
        values=[]
        for k in range(4):
            rs=rotate_state(s,-k);ra=rotate_vec(action,-k);rc=rotate_context(context,-k) if joint else context
            values.append(rotate_motion(self.raw(rs,ra,rc),k))
        return torch.stack(values).mean(0)

    def forward(self,s,action,context):
        if self.kind=='wrong_exact':correction=self.symmetrized(s,action,context,False)
        elif self.kind=='joint_exact':correction=self.symmetrized(s,action,context,True)
        elif self.kind=='relaxed':
            strength=self.mix_logit.sigmoid();correction=strength*self.symmetrized(s,action,context,False)+(1-strength)*self.raw(s,action,context)
        else:correction=self.raw(s,action,context)
        return (inertial_next(s,action)+correction)*s[...,-1:]


def load_split(root,mode,split):
    # The requested split is explicit; train archives physically stop after 20 frames.
    with np.load(root/mode/split/'trajectories.npz',allow_pickle=False) as a:
        states=encode_states(a['states']);actions=torch.tensor(a['actions'],dtype=torch.float32)/3
        context=encode_context(a['contexts']);events=torch.tensor(a['events']);counts=torch.tensor(a['n_objects'])
    transitions=states.shape[1]-1
    return {'state':states[:,:-1].reshape(-1,4,STATE_DIM),'next':states[:,1:,:,:4].reshape(-1,4,4),
        'action':actions.reshape(-1,4,2),'context':context[:,None,:].expand(-1,transitions,-1).reshape(-1,3),
        'events':events.reshape(-1,2),'episode':torch.arange(len(states))[:,None].expand(-1,transitions).reshape(-1),
        'counts':counts,'states_sequence':states,'actions_sequence':actions,'contexts_sequence':context}


def permute_slots(state,action,target,order):
    def take(a):return a.gather(1,order[...,None].expand(-1,-1,a.shape[-1]))
    return take(state),take(action),take(target)
