"""Explicit device amendment: accelerate AE/decoder, retain CPU sampler/flow.

Historical source and protocol are untouched. This adapter records its own
hash in every training context. Both paired arms use the same accelerated AE.
"""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import generation_run as g
import torch
from p0 import sha,dump

BaseAE=g.r.ResearchAE
BaseDecoder=g.ControlledDecoder

class BridgeAE(BaseAE):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.to('mps')
    def encode(self,x):
        return super().encode(x.to(next(self.parameters()).device)).cpu()
    def decode(self,z):
        return super().decode(z.to(next(self.parameters()).device)).cpu()
    def state_dict(self,*args,**kwargs):
        return {k:v.detach().cpu() for k,v in super().state_dict(*args,**kwargs).items()}

class BridgeDecoder(BaseDecoder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.to('mps')
    def forward(self,z):return super().forward(z.to(next(self.parameters()).device)).cpu()
    def state_dict(self,*args,**kwargs):
        return {k:v.detach().cpu() for k,v in super().state_dict(*args,**kwargs).items()}

def main(seed,init):
    assert torch.backends.mps.is_available()
    probe=HERE/'generation_accelerator_probe.json';assert json.loads(probe.read_text())['all_passed']
    amendment=g.BASE/'device_amendment.json'
    cfg={'adapter_sha256':sha(Path(__file__)),'probe_sha256':sha(probe),'base_protocol_sha256':sha(g.LOCK),
         'scope':'New confirmation training only. Development stays CPU. AE/decoder MPS FP32 with differentiable CPU output bridge; sampling, flow, evaluation CPU.',
         'reason':'Measured AE step approximately 0.051s vs observed CPU 0.12s under concurrent workload. Forward and gradient parity passed; no convergence equivalence claim.',
         'fairness':'Baseline and candidate share same AE/normalization; both decoder arms trained on MPS. Flow arms remain CPU. Time measurements reflect concurrent workloads.',
         'confirmation_scores_seen_before_amendment':False}
    if amendment.exists():assert json.loads(amendment.read_text())==cfg
    else:dump(amendment,cfg)
    original_context=g.context
    g.context=lambda s,i:{**original_context(s,i),'device_amendment_sha256':sha(amendment)}
    g.r.ResearchAE=BridgeAE;g.p1.ControlledDecoder=BridgeDecoder;g.ControlledDecoder=BridgeDecoder
    torch.set_num_threads(4);g.train(seed,init)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--init',type=int,required=True)
    a=p.parse_args();main(a.seed,a.init)
