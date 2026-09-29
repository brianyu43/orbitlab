"""Learned RGBA patches plus a source-image-conditioned neural edit decoder.

State coordinates determine patch placement, an explicit geometric prior.
Background and occluder pixels enter inference only through the source image.
"""
import copy,json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import r2_world as w
import r1_core as r1
from v3_common import HERE,sha,lock
BASE=w.BASE/'decoder'
OPS=['identity','color','rotate','translate']

def freeze():
    return lock(BASE/'protocol.json',{
        'version':1,'sources':{p:sha(HERE/p) for p in ['r2_decoder_core.py','r2_decoder_data.py','r2_decoder_train.py']},
        'reader_protocol_sha256':sha(w.BASE/'training_protocol.json'),'data_protocol_sha256':sha(w.BASE/'data_protocol.json'),
        'development_seed':w.DEVELOPMENT,'initialization':0,'device':'cpu','threads':2,
        'patch':{'steps':2000,'batch':64,'optimizer':'AdamW .001 weight_decay .0001, clip1','input':'Shape/pose/radius one-hot and continuous RGB. Output33x33 RGBA; alpha and RGB are learned. Coordinates place patches using bilinear grid_sample; predicted slot order is used, never GT painter-order repair.','loss':'4 premultiplied-RGB weighted MSE + weighted alpha BCE; weight1+8*true_alpha'},
        'editor':{'steps':4000,'batch':16,'optimizer':'AdamW .0005 weight_decay .0001, clip1','input':'Current sourceRGB, learned current/edited premultiplied RGBA foregrounds, external click heatmap, operation one-hot, requested RGB and translation. No GT background, occluder mask, object IDs, true count or target image at inferred-state evaluation.','training':'Shared editor trained on true current states from1024two-object training clips with identity/recolor/rotate/translate commands. Ground-truth target pixels and masks supervise losses only. Oracle-state to inferred-state shift is explicit; no end-to-end reader fine-tuning.','loss':'Weighted MSE weights1+12*changed+4*source foreground+4*panel, plus .1 weighted change-mask BCE. Targets preserve the original background and fixed original panel.','cache':'Frozen patch renders quantized to float16 at both train and test inputs.'},
        'evaluation':'All512validation scenes at counts2/3/4/6, all3edits, each six-reader inferred state and ground-truth-state control. Copy-input baseline. Identity also evaluated for oracle state. Canonical foreground reconstruction/edited foreground are separate diagnostics and do not imply background preservation.',
        'pixel_metrics':'Whole/changed/preserved/non-target/panel/background MAE, known foreground-alpha IoU for neural patch diagnostic, changed-area fraction. Pixel edit gate: changed MAE<=.05, preserved MAE<=.01, non-target MAE<=.02, panel MAE<=.01. Absent-region errors are0 with zero area recorded; identity excluded from edit aggregates. These pixel tolerances are not semantic state accuracy.',
        'end_to_end':'Report pixel gate and joint pixel-plus-existing-analytic-state success separately. No perceptual or human-quality claim. Reader state metrics retain their fixed earlier thresholds.',
        'selection':'No decoder architecture selection or hyperparameter tuning on validation. One fixed development decoder; continue only with explicit new protocol if changed.',
        'resource_cap':'Stage-local12hours/30GB includes reader work; benchmark100steps per new network before full training.'})

def command(index,op,click):
    return {'op':op,'click':list(click),'rgb':[45+((index*37+71)%201),45+((index*53+127)%201),45+((index*19+29)%201)],'dx':5,'dy':0}

def edit(objects,cmd,target=None):
    result=copy.deepcopy(objects)
    if not result:return result,-1
    selected=int(np.argmin([(v['x']-cmd['click'][0])**2+(v['y']-cmd['click'][1])**2 for v in result])) if target is None else target
    v=result[selected]
    if cmd['op']=='color':v['rgb']=cmd['rgb']
    elif cmd['op']=='rotate':v['pose']=(v['pose']+1)%4
    elif cmd['op']=='translate':v['x']+=cmd['dx'];v['y']+=cmd['dy']
    return result,selected

def states(scenes):
    out=torch.zeros(len(scenes),6,18)
    for b,objects in enumerate(scenes):
        assert len(objects)<=6
        for j,v in enumerate(objects):
            out[b,j,v['shape']]=1;out[b,j,4+v['pose']%4]=1;out[b,j,8+v['radius']-5]=1
            out[b,j,12:15]=torch.tensor(v['rgb'])/255;out[b,j,15:17]=torch.tensor([v['x'],v['y']]);out[b,j,17]=1
    return out

class PatchDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(15,96),nn.SiLU(),nn.Linear(96,128),nn.SiLU(),nn.Linear(128,4*33*33))
    def forward(self,attributes):
        v=self.net(attributes).reshape(-1,4,33,33).sigmoid();return torch.cat([v[:,:3]*v[:,3:],v[:,3:]],1)

def render(model,state):
    b,s,_=state.shape;patch=model(state[...,:15].reshape(-1,15))
    yy,xx=torch.meshgrid(torch.arange(64,device=state.device),torch.arange(64,device=state.device),indexing='ij')
    grid=torch.stack([xx,yy],-1).float()[None,None]-state[:,:,None,None,15:17]
    sampled=F.grid_sample(patch,(grid/16).reshape(b*s,64,64,2),align_corners=True,padding_mode='zeros').reshape(b,s,4,64,64)
    sampled=sampled*state[:,:,17,None,None,None];out=state.new_zeros(b,4,64,64)
    for j in range(s):out=out*(1-sampled[:,j,3:])+sampled[:,j]
    return out

def command_maps(commands):
    out=torch.zeros(len(commands),10,64,64);yy,xx=torch.meshgrid(torch.arange(64),torch.arange(64),indexing='ij')
    for i,c in enumerate(commands):
        out[i,0]=torch.exp(-((xx-c['click'][0])**2+(yy-c['click'][1])**2)/18)
        out[i,1+OPS.index(c['op'])]=1
        if c['op']=='color':out[i,5:8]=torch.tensor(c['rgb'])[:,None,None]/255
        if c['op']=='translate':out[i,8]=c['dx']/8;out[i,9]=c['dy']/8
    return out

class Conv(nn.Sequential):
    def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.SiLU())

class ImageEditor(nn.Module):
    def __init__(self):
        super().__init__();self.a=Conv(21,24);self.b=Conv(24,40);self.c=Conv(40,64);self.upb=Conv(104,40);self.upa=Conv(64,24);self.final=nn.Conv2d(24,4,1)
        nn.init.zeros_(self.final.weight);nn.init.zeros_(self.final.bias)
        with torch.no_grad():self.final.bias[3]=-2
    def forward(self,source,current,edited,commands):
        a=self.a(torch.cat([source,current,edited,commands],1));b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        y=self.upb(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        y=self.upa(torch.cat([F.interpolate(y,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1));v=self.final(y)
        return (source+v[:,:3].tanh()*v[:,3:].sigmoid()).clamp(0,1),v[:,3:]

def truth_render(objects,bg,panel):
    layers=w.layers(objects);image=bg.copy();foreground=np.zeros((64,64,4),np.float32)
    for v,a in zip(objects,layers):
        image=image*(1-a[...,None])+np.array(v['rgb'],np.float32)*a[...,None]
        patch=np.concatenate([np.array(v['rgb'],np.float32)[None,None]*a[...,None]/255,a[...,None]],-1)
        foreground=foreground*(1-a[...,None])+patch
    image=image*(1-panel[...,None])+np.array([116,120,124],np.float32)*panel[...,None]
    return np.rint(image).clip(0,255).astype(np.uint8),foreground,layers
