"""Supervised set readout of states from frozen RGB-only slot representations.

Ground-truth states supervise the readout during training only. No discovery
without labels is claimed. Matching in the inference evaluator never supplies
states, counts or identities to the model itself.
"""
import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class StateReadout(nn.Module):
    def __init__(self,mean,std):
        super().__init__();self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.net=nn.Sequential(nn.Linear(32,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,16))

    def forward(self,slots):return self.net((slots-self.mean)/self.std)


def attribute_cost(raw,truth):
    # B x G x S pair costs, with all weights fixed before optimization.
    shape=-(truth[:,:,None,:4]*F.log_softmax(raw[...,:4],-1)[:,None]).sum(-1)
    color=-(truth[:,:,None,4:10]*F.log_softmax(raw[...,4:10],-1)[:,None]).sum(-1)
    xy=(raw[:,None,:,10:12]-truth[:,:,None,10:12]).square().mean(-1)
    radius=(raw[:,None,:,12]-truth[:,:,None,12]).square()
    pose=(raw[:,None,:,13:15]-truth[:,:,None,13:15]).square().mean(-1)
    return shape+color+20*xy+5*radius+pose


def set_loss(raw,labels):
    if not torch.all((labels[...,15]>.5).sum(1)==2):raise ValueError('Training protocol requires two labeled objects')
    truth=labels[labels[...,15]>.5].reshape(len(labels),2,16)
    attrs=attribute_cost(raw,truth);slots=raw.shape[1]
    # This assignment minimizes the full mean-attribute + mean-presence loss:
    # changing an empty slot to present changes BCE by -presence_logit / S.
    cost=attrs-(2/slots)*raw[:,None,:,15]
    combinations=torch.tensor(list(itertools.permutations(range(slots),2)),device=raw.device)
    candidate=cost[:,0,combinations[:,0]]+cost[:,1,combinations[:,1]]
    match=combinations[candidate.detach().argmin(-1)]
    selected=attrs.gather(2,match[:,:,None]).mean()
    presence=torch.zeros_like(raw[...,15]).scatter(1,match,1)
    return selected+F.binary_cross_entropy_with_logits(raw[...,15],presence),match


def discrete_states(raw):
    """Public inference interface: decode own predictions, without a true count."""
    out=torch.zeros_like(raw)
    out[...,:4]=F.one_hot(raw[...,:4].argmax(-1),4)
    out[...,4:10]=F.one_hot(raw[...,4:10].argmax(-1),6)
    out[...,10:13]=raw[...,10:13]
    directions=raw.new_tensor([[1,0],[0,-1],[-1,0],[0,1]])
    out[...,13:15]=directions[(raw[...,13:15]@directions.T).argmax(-1)]
    out[...,15]=(raw[...,15].sigmoid()>=.5).to(raw.dtype)
    return out*out[...,15:16]


def geometric_assignment(predicted,truth):
    p=np.flatnonzero(predicted[:,15]>.5);g=np.flatnonzero(truth[:,15]>.5)
    if not len(p):return {},p,g
    distance=((truth[g,None,10:12]-predicted[None,p,10:12])**2).sum(-1)
    gi,pi=linear_sum_assignment(distance)
    return {int(g[a]):int(p[b]) for a,b in zip(gi,pi)},p,g


def measure(raw,truth):
    predictions=discrete_states(raw).detach().cpu().numpy();truth=truth.detach().cpu().numpy();rows=[];assignments=[]
    for i,(pred,target) in enumerate(zip(predictions,truth)):
        matched,p,g=geometric_assignment(pred,target);assignments.append(matched)
        correct=[];errors=[]
        for gid in g:
            if gid not in matched:correct.append(np.zeros(6,bool));continue
            v=pred[matched[gid]];t=target[gid]
            checks=np.array([v[:4].argmax()==t[:4].argmax(),v[4:10].argmax()==t[4:10].argmax(),
                np.all(np.abs(v[10:12]-t[10:12])*31.5<=1),abs(v[12]-t[12])*8<=.5,
                np.all(v[13:15]==t[13:15]),v[15]>.5])
            correct.append(checks);errors.append(float(np.abs(v[10:12]-t[10:12]).mean()*31.5))
        correct=np.stack(correct)
        rows.append({'base_id':i,'true_objects':len(g),'predicted_objects':len(p),'count_correct':len(g)==len(p),
            'matched_fraction':len(matched)/len(g),'shape_correct':float(correct[:,0].mean()),'color_correct':float(correct[:,1].mean()),
            'position_within_one_pixel':float(correct[:,2].mean()),'radius_within_half_pixel':float(correct[:,3].mean()),
            'pose_correct':float(correct[:,4].mean()),'object_all_factors_correct':float(correct.all(-1).mean()),
            'full_scene_success':bool(len(g)==len(p) and correct.all()),
            'matched_position_mae_pixels':float(np.mean(errors)) if errors else None})
    return rows,assignments
