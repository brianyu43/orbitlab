import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import torch,numpy as np
import research as r,generation as g

torch.set_num_threads(4);train,_,_=r.load_data('train',1024);report={}
for split in ['test','ood']:
 x,_,_=r.load_data(split,256);dist=[]
 for i in range(0,len(x),64):
  d,_,_=g.nearest_train(x[i:i+64],train,torch.device('mps'));dist.extend(d)
 report[split]={'n':len(x),'mean':float(np.mean(dist)),'quantiles':{str(q):float(np.quantile(dist,q)) for q in [0,.05,.5,.95,1]},'definition':'exact RGB MSE at 64x64 versus all 1024 train scenes and their 4 rotations'}
ev=g.StateEvaluator();x,_,_=r.load_data('test',256);states=ev.predict(x);report['real_test_state_reference']={k:{'mean':float(np.mean([v[k] for v in states])),'sd':float(np.std([v[k] for v in states]))} for k in ['cx','cy','occupancy','scale','template_iou']}
r.dump(r.ROOT/'reports/nn_reference.json',report);print(json.dumps(report,indent=2))
