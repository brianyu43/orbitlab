"""Native128px CPU/MPS parity and100update timing; reset before research training."""
import json,platform,time,resource
import numpy as np
import torch
from v3_common import HERE,sha,dump,lock
from r5_intake import BASE
from r5_data import Pairs
from r5_models import make,pixel_metrics
from r4_core import state_hash

def main():
    assert torch.backends.mps.is_available(),'MPS unavailable; no silent device substitution'
    torch.set_num_threads(2);device=torch.device('mps');data=Pairs('Single_Atomic');folder=BASE/'model_preflight_v1';folder.mkdir(parents=True,exist_ok=True)
    protocol={'model_source_sha256':sha(HERE/'r5_models.py'),'data_source_sha256':sha(HERE/'r5_data.py'),'source_sha256':sha(__file__),
        'official_intake_sha256':sha(BASE/'data/dsprites_hard/manifest.json'),'kinds':['plain','c4','object'],'batch':32,'steps':100,
        'optimizer':'Adam lr=.0005, grad clip1, full image MSE','device':'MPSfloat32;CPU2threads;actual native128px PNG loading included',
        'timing_only':'100pilot updates per model, discarded for full experiments; no model selection or scientific scoring. Uses deterministic random official training indices, not test.',
        'parity':'Five CPUupdates on16officialtrainingpairs to obtain nonidentityweights, then same checkpoint CPUvsMPSforward on8pairs; absolute tolerance2e-4. Separate from timing100steps.',
        'torch':torch.__version__,'platform':platform.platform()}
    lock(folder/'protocol.json',protocol);indices=np.random.default_rng(951100).integers(64000,size=(100,32))
    for kind in ['plain','c4','object']:
        dest=folder/(kind+'.json')
        if dest.exists():assert json.loads(dest.read_text())['protocol_sha256']==sha(folder/'protocol.json');continue
        torch.manual_seed(950000);cpu=make(kind);x,y=data.batch('train',np.arange(16),torch.device('cpu'));opt=torch.optim.Adam(cpu.parameters(),lr=.0005)
        for _ in range(5):
            loss=(cpu(x)-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(cpu.parameters(),1);opt.step()
        cpu.eval();gpu=make(kind).to(device);gpu.load_state_dict(cpu.state_dict());gpu.eval()
        with torch.no_grad():
            expected=cpu(x[:8]);actual=gpu(x[:8].to(device)).cpu();maxerr=float((actual-expected).abs().max());assert maxerr<2e-4
            eq=None
            if kind=='c4':eq=float((gpu(torch.rot90(x[:8].to(device),1,(-2,-1)))-torch.rot90(actual.to(device),1,(-2,-1))).abs().max());assert eq<2e-4
        del cpu,gpu,opt;torch.mps.empty_cache();torch.manual_seed(950000);model=make(kind).to(device);initial=state_hash(model);model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
        loading=0.;compute=0.;torch.mps.synchronize();start=time.perf_counter();logs=[]
        for step,idx in enumerate(indices,1):
            t=time.perf_counter();x,y=data.batch('train',idx,device);torch.mps.synchronize();loading+=time.perf_counter()-t
            t=time.perf_counter();pred=model(x);loss=(pred-y).square().mean();assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);optimizer.step();torch.mps.synchronize();compute+=time.perf_counter()-t
            if step%25==0:logs.append({'step':step,'loss':float(loss.detach().cpu())});print('R5 timing',kind,step,round(time.perf_counter()-start,1),flush=True)
        seconds=time.perf_counter()-start
        dump(dest,{'protocol_sha256':sha(folder/'protocol.json'),'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),'initial_sha256':initial,'CPU_MPS_max_abs':maxerr,'c4_commutation_max_abs':eq,'steps':100,'batch':32,'seconds':seconds,'PNG_load_and_transfer_seconds':loading,'compute_seconds':compute,'seconds_per_update':seconds/100,'projected_20epoch57600_seconds':seconds/100*36000,'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'logs':logs,'benchmark_weights_discarded':True})
        del model,optimizer,x,y,pred,loss;torch.mps.empty_cache()
    data.close();print('R5 CPU/MPS parity and native PNG timing passed',flush=True)

if __name__=='__main__':main()
