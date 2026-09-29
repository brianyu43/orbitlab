"""Native128px RGB-to-RGB comparison: global, C4 and learned slots.

Every primary model receives only source RGB and image-to-image supervision.
The slot arm has a grouping bias; slots are not assumed to correspond to objects
without measurement. Parameters and compute differ and must be reported.
"""
import torch
from torch import nn
import torch.nn.functional as F
from v3_common import ROOT
from svib_preview_models import PreviewPredictor

class ObjectPredictor(nn.Module):
    def __init__(self):
        super().__init__();self.slot_count=3;self.width=128
        self.encoder=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,5,2,2),nn.SiLU(),
            nn.Conv2d(32,64,5,2,2),nn.SiLU(),nn.Conv2d(64,64,5,2,2),nn.SiLU())
        axis=torch.linspace(-1,1,8);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('positions',torch.stack([xx,yy],-1).reshape(1,64,2),persistent=False)
        self.tokens=nn.Sequential(nn.Linear(66,128),nn.LayerNorm(128),nn.Linear(128,128),nn.SiLU())
        self.slots=nn.Parameter(torch.randn(1,3,128)*.02)
        self.slot_norm=nn.LayerNorm(128);self.key=nn.Linear(128,128,bias=False);self.value=nn.Linear(128,128,bias=False);self.query=nn.Linear(128,128,bias=False)
        self.update=nn.GRUCell(128,128);self.refine=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,128))
        self.interaction=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128))
        self.expand=nn.Sequential(nn.Linear(128,64*8*8),nn.SiLU())
        self.decoder=nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1),nn.SiLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.SiLU(),
            nn.ConvTranspose2d(16,16,4,2,1),nn.SiLU(),nn.ConvTranspose2d(16,4,4,2,1))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,image,return_details=False):
        assert image.shape[1:]==(3,128,128)
        features=self.encoder(image).flatten(2).transpose(1,2)
        tokens=self.tokens(torch.cat([features,self.positions.expand(len(image),-1,-1)],-1))
        keys=self.key(tokens);values=self.value(tokens);slots=self.slots.expand(len(image),-1,-1)
        for _ in range(3):
            queries=self.query(self.slot_norm(slots));attention=torch.einsum('bsd,bnd->bsn',queries,keys)*(128**-.5)
            attention=attention.softmax(1)+1e-8;attention=attention/attention.sum(-1,keepdim=True)
            updates=torch.einsum('bsn,bnd->bsd',attention,values)
            slots=self.update(updates.reshape(-1,128),slots.reshape(-1,128)).reshape(len(image),3,128)
            slots=slots+self.refine(slots)
        others=(slots.sum(1,keepdim=True)-slots)/2
        predicted_slots=slots+self.interaction(torch.cat([slots,others],-1))
        outputs=self.decoder(self.expand(predicted_slots.reshape(-1,128)).reshape(-1,64,8,8)).reshape(len(image),3,4,128,128)
        masks=outputs[:,:,3:4].softmax(1);residual=(masks*outputs[:,:,:3].tanh()).sum(1)
        result=(image+residual).clamp(0,1)
        if return_details:return result,{'input_attention':attention,'output_masks':masks,'slots':slots,'predicted_slots':predicted_slots}
        return result

def make(kind):
    return ObjectPredictor() if kind=='object' else PreviewPredictor(kind)

def pixel_metrics(pred,source,target):
    error=(pred-target).square().mean(1);changed=(source!=target).any(1)
    def masked(mask):return (error*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    return torch.stack([error.mean((1,2)),masked(changed),masked(~changed),changed.sum((1,2)),
        (source-target).square().mean((1,2,3)),(pred-source).square().mean((1,2,3))],1)

METRIC_NAMES=['image_mse','changed_mse','preserved_mse','changed_pixels','copy_mse','input_damage_mse']
