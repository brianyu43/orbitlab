"""Count-normalized interactions and a separately labeled physics-residual model."""
from v3_common import ROOT,HERE,sha,lock
import torch
from torch import nn
import torch.nn.functional as F
from dynamics_models import Transition,encode_states,encode_context,rotate_state,rotate_vec,rotate_context,rotate_motion

BASE=HERE/'r3';DEVELOPMENT=888101;CONFIRMATION=[888201,888202,888203]
ARMS=['one','multi','multi_bounded','mean','local_mean','contact_residual']

def freeze():
    return lock(BASE/'protocol.json',{
        'sources':{p:sha(HERE/p) for p in ['r3_physics.py','r3_core.py','r3_data.py','r3_train.py','r3_evaluate.py']},
        'dependencies':{p:sha(ROOT/p) for p in ['followup/dynamics_world.py','followup/dynamics_models.py','followup/dynamics_rgb_baseline.py']},
        'development_seed':DEVELOPMENT,'confirmation_seeds':CONFIRMATION,'initializations':[0,1,2],
        'training_steps':2000,'batch_episodes':32,'train_episodes':384,'train_frames':20,'training_count':2,
        'evaluation_episodes_per_count':96,'counts':[2,3,4,6],'horizons':[1,16,61,128],
        'stratification':'Equal quotas of no contact / wall only / pair contact within the first 16 forecast frames. Report actual per-horizon contact categories as well. Training is similarly stratified within first16 frames.',
        'sampling_amendment':'Version2 before any R3 model training: explicitly proposed free-flight (velocity .1x, force .01x, impulse .1x), wall (velocity .05x except outward1px/frame disk near wall, force .1x, impulse .1x), and random pair-contact families. Actual events must match the requested stratum. These are controlled mechanism strata with different force/action distributions, not population-representative samples. Version1 sampling exhaustion is archived.',
        'arms':ARMS,'optimizer':'AdamW .001 weight_decay .0001 clip1','multistep_horizon':8,
        'inputs':'True initial state at observed frame3 and supplied action; net force+drag either true or estimated from four RGB observations. Learned transitions train with true force. RGB context estimator knows circle shape, palette, and discrete physics. Initial state remains true in the estimated-force condition to isolate force error.',
        'architecture':'one/multi/multi_bounded reproduce previous joint-C4 sum-message architecture with no count hardcoding. mean changes only message sum to degree mean. local_mean additionally uses a proximity gate. All these models share parameter counts; contact_residual has a declared different physical prior and parameter count.',
        'local_gate':'Pair center distance <= radius_i+radius_j+6 pixels at current input, multiplied before degree averaging. No truth contact events.',
        'contact_prior':'Equal masses, known force/drag, reflective walls and pair elastic impulse/projection; a coarse one-substep differentiable solver plus learned bounded per-object velocity residual applied before solver. Exact eight-substep simulator is an oracle, not a learned competitor. Coarse solver without learned residual is mandatory control.',
        'counterfactuals':'Same observed initial state with negated target impulse (target/nontarget response and zero-response control), and same impulse with x net external force increment .04. Force changes supplied for known-force; estimated baseline force receives the same known intervention delta.',
        'bounds':'multi_bounded clips xy to radius-specific walls and velocity to 1.5*maximum train norm; other neural arms have no feedback clamp. contact residual amplitude .15px/frame per axis and post-step physical projection; stability must not be credited to learning.',
        'development_selection':'Compare candidate mean/local_mean/contact_residual at 16 and61 steps, including all failures. Choose smallest mean failure-penalized position MAE across counts. Advance baseline+candidate only if candidate beats bounded mean61-step error by at least20% at count4, has <1% failed paths, <1% wall paths, <5% overlap paths at every count, and non-target impulse-response error <=1.05*bounded count4. Candidate must also have <=1px mean16-step error in each count2 contact stratum. Otherwise diagnose and stop expansion.',
        'confirmation_gate':'All three dataset seeds must meet the same count4 61-step criteria after averaging three initializations; no failed seeds discarded.',
        'failure_definition':'Nonfinite path or any center coordinate absolute value>64px is failure. Mean position error across all paths uses128px for failure; velocity uses16px/frame. Raw finite-only metrics also reported. All denominators include all96episodes.',
        'contact_metric':'Near contact if distance <= radii_sum+.5px; first contact timing capped at horizon+1 if absent. Compare near-contact traces at frame endpoints; this is not simulator substep impulse timing.',
        'caps':{'local_seconds':43200,'disk_bytes':30000000000}})

class NormalizedTransition(Transition):
    def __init__(self,local=False):
        super().__init__('joint_exact');self.local=local
    def raw(self,s,action,context):
        b,n,_=s.shape;sa=torch.cat([s,action],-1);ctx=context[:,None,None,:].expand(-1,n,n,-1)
        edges=self.edge(torch.cat([sa[:,:,None,:].expand(-1,-1,n,-1),sa[:,None,:,:].expand(-1,n,-1,-1),ctx],-1))
        live=s[...,-1];mask=live[:,:,None]*live[:,None,:]*(1-torch.eye(n,device=s.device)[None])
        if self.local:
            distance=(s[:,:,None,:2]-s[:,None,:,:2]).norm(dim=-1)*31.5
            mask=mask*(distance<=(s[:,:,None,4]+s[:,None,:,4])*8+6)
        messages=(edges*mask[...,None]).sum(2)/mask.sum(2,keepdim=True).clamp_min(1)
        return self.net(torch.cat([sa,context[:,None,:].expand(-1,n,-1),messages],-1))

def coarse(s,action,context,correction=None):
    """One-substep physical prior; never described as learned collision discovery."""
    p=s[...,:2]*31.5;v=s[...,2:4]*3+action*3
    if correction is not None:v=v+correction
    radius=s[...,4]*8;limit=31.5-radius;drag=context[:,2]*.01;net=context[:,:2]*.1
    decay=torch.exp(-drag);force=-torch.expm1(-drag)/drag.clamp_min(1e-8)
    force=torch.where(drag>1e-8,force,torch.ones_like(force))
    v=v*decay[:,None,None]+net[:,None,:]*force[:,None,None];p=p+v
    outside=p.abs()>limit[...,None];sign=p.sign()
    p=torch.where(outside,sign*(2*limit[...,None]-p.abs()),p)
    v=torch.where(outside&(v*sign>0),-v,v)
    n=s.shape[1]
    # Fixed eight sweeps, no label-dependent stopping. All state updates are functional for autograd.
    for _ in range(8):
        for i in range(n):
            for j in range(i+1,n):
                delta=p[:,j]-p[:,i];distance=delta.norm(dim=-1).clamp_min(1e-8);normal=delta/distance[:,None]
                overlap=(radius[:,i]+radius[:,j]-distance).clamp_min(0)
                hit=distance<=radius[:,i]+radius[:,j]+1e-7
                approach=((v[:,j]-v[:,i])*normal).sum(-1)
                impulse=approach.clamp_max(0)*hit
                selector=F.one_hot(torch.tensor(i,device=s.device),n)-F.one_hot(torch.tensor(j,device=s.device),n)
                p=p-selector[None,:,None]*(normal*overlap[:,None]/2)[:,None,:]
                v=v+selector[None,:,None]*(normal*impulse[:,None])[:,None,:]
        outside=p.abs()>limit[...,None];sign=p.sign()
        p=torch.maximum(torch.minimum(p,limit[...,None]),-limit[...,None])
        v=torch.where(outside&(v*sign>0),-v,v)
    return torch.cat([p/31.5,v/3],-1)

class ContactResidual(nn.Module):
    def __init__(self):
        super().__init__();self.core=NormalizedTransition(local=True)
    def forward(self,s,action,context):
        correction=self.core.symmetrized(s,action,context,True)[...,2:]
        return coarse(s,action,context,.15*torch.tanh(correction))

def model_for(arm):
    if arm=='contact_residual':return ContactResidual()
    if arm in ['mean','local_mean']:return NormalizedTransition(local=arm=='local_mean')
    return Transition('joint_exact')

def advance(model,s,action,context,arm,limit):
    motion=coarse(s,action,context) if arm=='coarse_physics' else model(s,action,context)
    if arm=='multi_bounded':
        wall=1-s[...,4:5]*8/31.5;xy=torch.maximum(torch.minimum(motion[...,:2],wall),-wall)
        v=motion[...,2:];v=v*(limit/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
        motion=torch.cat([xy,v],-1)
    return torch.cat([motion,s[...,4:]],-1)
