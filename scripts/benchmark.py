import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import torch,numpy as np
import research as r,orbitlab as o

torch.set_num_threads(4);dev=o.choose_device('mps');x,_,_=r.load_data('train',32);x=x.to(dev);results={}
for model,width in [('aug',16),('equivariant',16),('equivariant',31)]:
 o.seed_all(19);ae=r.ResearchAE(model,width).to(dev);opt=torch.optim.AdamW(ae.parameters(),lr=.001)
 def step():
  opt.zero_grad(set_to_none=True);loss=o.reconstruction_loss(ae(x),x);loss.backward();torch.nn.utils.clip_grad_norm_(ae.parameters(),1.);opt.step()
 for _ in range(20):step()
 timings=[]
 for _ in range(3):
  o.sync(dev);start=time.perf_counter()
  for _ in range(100):step()
  o.sync(dev);timings.append((time.perf_counter()-start)/100)
 results[f'{model}_w{width}']={'seconds_per_step_3_trials':timings,'median_seconds_per_step':float(np.median(timings)),'warmup':20,'batch':32,'training_only':True}
r.dump(r.ROOT/'reports/benchmark.json',results);print(json.dumps(results,indent=2))
