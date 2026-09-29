"""CPU/MPS forward/gradient parity and small training timing probe."""
from pathlib import Path
import sys,time,json
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'work'),str(ROOT/'followup'),str(ROOT/'generation_recovery')]
import torch
import research as r
import orbitlab as o
from decoder_study import ControlledDecoder
from p0 import dump,sha

def main():
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available()
    torch.manual_seed(991);checks={}
    for name,factory,x in [('ae',lambda:r.ResearchAE('equivariant',16),torch.rand(4,3,64,64)),
                           ('decoder',lambda:ControlledDecoder('logit_mean'),torch.rand(4,4,16))]:
        cpu=factory();gpu=factory().to('mps');gpu.load_state_dict(cpu.state_dict())
        a=cpu(x);b=gpu(x.to('mps')).cpu();forward=float((a-b).abs().max())
        target=torch.rand_like(a);la=o.reconstruction_loss(a,target);lb=o.reconstruction_loss(b,target)
        la.backward();lb.backward()
        absolute=max(float((p.grad-q.grad.cpu()).abs().max()) for p,q in zip(cpu.parameters(),gpu.parameters()))
        gradscale=max(float(p.grad.abs().max()) for p in cpu.parameters())
        checks[name]={'forward_max_abs':forward,'gradient_max_abs':absolute,'gradient_relative_to_global_max':absolute/max(gradscale,1e-12)}
        assert forward<1e-4 and absolute<1e-4 and absolute/max(gradscale,1e-12)<.01
    model=r.ResearchAE('equivariant',16).to('mps');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    x=torch.rand(32,3,64,64);start=time.perf_counter()
    for _ in range(30):
        y=model(x.to('mps')).cpu();loss=o.reconstruction_loss(y,x);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    torch.mps.synchronize();checks['bridge_seconds_per_ae_step']=(time.perf_counter()-start)/30
    dest=HERE/'generation_accelerator_probe.json'
    if not dest.exists():dump(dest,{'checks':checks,'all_passed':True,'torch_version':str(torch.__version__),'source_sha256':sha(Path(__file__))})
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
