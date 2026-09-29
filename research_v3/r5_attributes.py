"""Independent image-attribute measurement with disclosed privileged localization.

Readers use a fixed112px RGB crop centered at the SOURCE object's true location.
No true shape/size/color/material or segmentation mask is an input. Labels train
the measurement instrument only; primary RGB predictors never receive them.
Calibration accuracy is always reported, not assumed perfect.
"""
import argparse,json,time,resource
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from v3_common import HERE,ROOT,sha,dump,lock
from r5_intake import BASE,TASKS
from r5_data import Pairs
from r5_3d_data import Pairs3D
from r5_3d_audit import schema as schema3d
from r5_train import save_torch,resource_check
from r4_core import state_hash

FAMILIES=('dsprites','clevr','clevrtex');STEPS=6000;BATCH=64

def folder(family,task):return BASE/'attributes'/family/task
def audit_folder(family,task):return BASE/'audits'/f'{family}_hard'/task
def store_for(family,task):return Pairs(task) if family=='dsprites' else Pairs3D(family,task)
def schema(family):
    if family=='dsprites':return ['shape','color','size'],[4,4,4]
    names,vocab=schema3d(family);return names,[len(vocab[n]) for n in names]

class Reader(nn.Module):
    def __init__(self,counts):
        super().__init__();self.counts=counts
        self.features=nn.Sequential(nn.Conv2d(3,16,5,2,2),nn.SiLU(),nn.Conv2d(16,32,5,2,2),nn.SiLU(),
            nn.Conv2d(32,48,3,2,1),nn.SiLU(),nn.Conv2d(48,64,3,2,1),nn.SiLU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.SiLU(),nn.Linear(128,sum(counts)))
    def forward(self,x):return self.head(self.features(x)).split(self.counts,dim=1)

def crops(images,centers):
    """One image and source pixel-center per requested object; output native spacing."""
    axis=torch.arange(112,device=images.device,dtype=images.dtype)-55.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');grid=torch.stack([xx,yy],-1)[None]+centers[:,None,None,:]
    grid=grid*2/127-1
    return F.grid_sample(images,grid,mode='bilinear',padding_mode='zeros',align_corners=True)

def load_factors(family,task):
    z=np.load(audit_folder(family,task)/'factors.npz');labels=np.stack([z['source'],z['target']],1)
    if family=='dsprites':xy=z['centers'];centers=np.stack([128*xy[...,0],127-128*xy[...,1]],-1)
    else:centers=z['centers'][:,0]
    return labels,centers

def batch(store,family,task,factors,indices,roles,objects):
    labels,centers=factors;images=[]
    for i,role in zip(indices,roles):
        split='train' if i<64000 else 'test';images.append(store.image(split,int(i)%64000,['source','target'][int(role)]))
    images=torch.from_numpy(np.stack(images).copy()).permute(0,3,1,2).float()/255
    xy=torch.from_numpy(centers[indices,objects].astype(np.float32));target=torch.from_numpy(labels[indices,roles,objects].astype(np.int64))
    return crops(images,xy),target

def freeze(family,task):
    names,counts=schema(family);audit=audit_folder(family,task);assert (audit/'summary.json').exists()
    return lock(folder(family,task)/'protocol.json',{'source_sha256':sha(__file__),'family':family,'task':task,'attribute_names':names,'classes':counts,
        'audit_summary_sha256':sha(audit/'summary.json'),'factor_sha256':sha(audit/'factors.npz'),'partition_sha256':sha(audit/'partition.npz'),
        'localization':'Privileged SOURCE metadata center, unchanged between source,target and predicted images. Fixed112x112crop, no size-dependent scale, bilinear fractional center, zero padding. For3D source projectedcenter is used even iftargetmeshheight changes projection. Not a detector or an unprivileged semantic evaluator.',
        'inputs':'CropRGB only; no true attribute, segmentation mask, object size or target center. True training attribute labels are extra supervision for the independent measurement instrument only.',
        'training':'6000Adam updates,lr.001,batch64,mean attribute crossentropy,gradclip1; no model/checkpoint/threshold choice; finalweights. CPUfloat32,2threads. Train-only random scene,source/target andobject with replacement.',
        'train_partition':'Use the development57600training families only. No cleanvalidation,officialtest,or primary model prediction is used in fitting. Same frozen reader later scores allprimary models and conditionalconfirmation; those resplits are not independentreadertraining.',
        'calibration':'All6400validation and8000officialtest scenes, both source/target roles and bothobjects; report perattribute,allattributes andbothobjects accuracy, changed/unchanged semantic attributes. No assumption of perfect readout or correction that manufactures accuracy.',
        'budget':{'steps':STEPS,'batch':BATCH,'preflight_updates_discarded':100},
        'source_dependencies':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'r5_data.py',HERE/'r5_3d_data.py',HERE/'r5_3d_audit.py']}})

def summary(pred,true,names):
    equal=pred==true
    return {'object_count':len(true),'per_attribute_accuracy':{n:float(equal[:,i].mean()) for i,n in enumerate(names)},'all_attributes_accuracy':float(equal.all(1).mean())}

@torch.no_grad()
def calibrate(family,task,split,reader=None,verify=False):
    torch.set_num_threads(2);freeze(family,task);out=folder(family,task);names,counts=schema(family)
    ck=out/'reader.pt';dest=out/f'calibration_{split}.json'
    if dest.exists() and not verify:
        d=json.loads(dest.read_text());assert d['checkpoint_sha256']==sha(ck) and d['predictions_sha256']==sha(out/f'calibration_{split}.npz');return d
    if reader is None:
        state=torch.load(ck,weights_only=True);assert state['context']['protocol_sha256']==sha(out/'protocol.json') and state['steps']==STEPS
        reader=Reader(counts);reader.load_state_dict(state['state_dict'])
    reader.eval();partition=np.load(audit_folder(family,task)/'partition.npz');ids=partition['val'] if split=='val' else np.arange(64000,72000)
    store=store_for(family,task);factors=load_factors(family,task);preds=[];truth=[];coordinates=[];started=time.perf_counter()
    # Every scene appears in stable source0,source1,target0,target1 order.
    items=np.array([(i,r,o) for i in ids for r in (0,1) for o in (0,1)],np.int64)
    for first in range(0,len(items),BATCH):
        q=items[first:first+BATCH];x,y=batch(store,family,task,factors,*q.T);p=torch.stack([v.argmax(1) for v in reader(x)],1)
        preds.append(p.numpy());truth.append(y.numpy())
    p=np.concatenate(preds);t=np.concatenate(truth)
    if verify:
        saved=json.loads(dest.read_text());z=np.load(out/f'calibration_{split}.npz')
        assert saved['checkpoint_sha256']==sha(ck) and saved['predictions_sha256']==sha(out/f'calibration_{split}.npz')
        for key,value in [('items',items),('prediction',p),('truth',t)]:assert np.array_equal(z[key],value)
        assert saved['all']==summary(p,t,names)
        dump(out/f'calibration_{split}_verification.json',{'summary_sha256':sha(dest),'all_object_predictions_replayed':len(items),'all_rows_bitexact':True,'seconds':time.perf_counter()-started})
        store.close();print('R5 attribute calibration replay',family,task,split,len(items),flush=True);return
    np.savez_compressed(out/f'calibration_{split}.npz',items=items,prediction=p,truth=t)
    result={'protocol_sha256':sha(out/'protocol.json'),'checkpoint_sha256':sha(ck),'split':split,'scenes':len(ids),'predictions_sha256':sha(out/f'calibration_{split}.npz'),
        'all':summary(p,t,names),'source':summary(p[items[:,1]==0],t[items[:,1]==0],names),'target':summary(p[items[:,1]==1],t[items[:,1]==1],names),
        'target_both_objects_all_attributes_accuracy':float((p.reshape(-1,2,2,len(names))[:,1]==t.reshape(-1,2,2,len(names))[:,1]).all((1,2)).mean()),
        'seconds':time.perf_counter()-started,'reader_is_perfect':False}
    dump(dest,result);store.close();print('R5 attribute calibration',family,task,split,result['target'],flush=True);return result

def train(family,task):
    assert family in FAMILIES and (family=='dsprites' and task in TASKS or task=='Single_Atomic');freeze(family,task);torch.set_num_threads(2)
    out=folder(family,task);names,counts=schema(family);names=list(names);store=store_for(family,task);factors=load_factors(family,task)
    ids=np.load(audit_folder(family,task)/'partition.npz')['train'];seed=960000+FAMILIES.index(family)*100+list(TASKS).index(task);torch.manual_seed(seed)
    reader=Reader(counts);initial=state_hash(reader);opt=torch.optim.Adam(reader.parameters(),lr=.001);rng=np.random.default_rng(seed+900)
    context={'protocol_sha256':sha(out/'protocol.json'),'initial_state_sha256':initial};logs=[];done=0;prior=0.
    if (out/'run.json').exists():
        run=json.loads((out/'run.json').read_text());assert run['context']==context and run['checkpoint_sha256']==sha(out/'reader.pt');store.close();return run
    if (out/'progress.pt').exists():
        state=torch.load(out/'progress.pt',weights_only=True);assert state['context']==context
        reader.load_state_dict(state['state_dict']);opt.load_state_dict(state['optimizer']);rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng']);done=state['step'];prior=state['seconds'];logs=state['logs']
    else:
        # Separate timing run, reset all scientific weights and random streams.
        started=time.perf_counter();pilot=[]
        for step in range(100):
            selected=rng.choice(ids,BATCH);roles=rng.integers(0,2,BATCH);objects=rng.integers(0,2,BATCH);x,y=batch(store,family,task,factors,selected,roles,objects)
            logits=reader(x);loss=sum(F.cross_entropy(v,y[:,i]) for i,v in enumerate(logits))/len(counts);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),1);assert torch.isfinite(norm);opt.step();pilot.append(float(loss.detach()))
        dump(out/'preflight.json',{'protocol_sha256':sha(out/'protocol.json'),'100step_seconds':time.perf_counter()-started,'loss_first':pilot[0],'loss_last':pilot[-1],'parameters':sum(p.numel() for p in reader.parameters()),'weights_discarded':True})
        torch.manual_seed(seed);reader=Reader(counts);assert state_hash(reader)==initial;opt=torch.optim.Adam(reader.parameters(),lr=.001);rng=np.random.default_rng(seed+900)
    started=time.perf_counter();reader.train()
    for step in range(done+1,STEPS+1):
        selected=rng.choice(ids,BATCH);roles=rng.integers(0,2,BATCH);objects=rng.integers(0,2,BATCH);x,y=batch(store,family,task,factors,selected,roles,objects)
        logits=reader(x);loss=sum(F.cross_entropy(v,y[:,i]) for i,v in enumerate(logits))/len(counts);assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(reader.parameters(),1);assert torch.isfinite(norm);opt.step()
        if step%500==0:
            seconds=prior+time.perf_counter()-started;logs.append({'step':step,'loss':float(loss.detach()),'seconds':seconds})
            save_torch(out/'progress.pt',{'context':context,'state_dict':reader.state_dict(),'optimizer':opt.state_dict(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'step':step,'seconds':seconds,'logs':logs})
            dump(out/'progress.json',logs[-1]);print('R5 attribute reader',family,task,logs[-1],flush=True);resource_check()
    save_torch(out/'reader.pt',{'state_dict':reader.state_dict(),'context':context,'steps':STEPS})
    run={'context':context,'steps':STEPS,'batch':BATCH,'seconds':prior+time.perf_counter()-started,'checkpoint_sha256':sha(out/'reader.pt'),
        'parameters':sum(p.numel() for p in reader.parameters()),'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'no_model_predictions_used':True,'logs':logs}
    dump(out/'run.json',run);store.close();return run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','calibrate','verify_calibration']);p.add_argument('--family',choices=FAMILIES,default='dsprites');p.add_argument('--task',choices=TASKS,default='Single_Atomic');p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
    if a.stage=='train':train(a.family,a.task)
    else:calibrate(a.family,a.task,a.split,verify=a.stage=='verify_calibration')
