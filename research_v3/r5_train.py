"""Validation-selected5/10/20epoch native128px dSprites study.

Training and validation do not read the official test. All three arms run the
same20epoch budget; selected checkpoint epochs can differ and are disclosed.
"""
import argparse,json,os,time,resource
from pathlib import Path
import numpy as np
import torch
from v3_common import HERE,ROOT,sha,dump,lock,event
from r5_intake import BASE,TASKS
from r5_data import Pairs
from r5_models import make,pixel_metrics,METRIC_NAMES
from r4_core import state_hash

KINDS=('plain','c4','object');EPOCHS=(5,10,20);TRAIN_N=57600;BATCH=32;STEPS_EPOCH=1800
RUNS=BASE/'external/dsprites_hard/runs';SEEDS=(0,1,2)

def freeze():
    return lock(BASE/'dsprites_training_protocol.json',{
        'version':1,'task_order':list(TASKS),'kinds':list(KINDS),'development_initialization':0,'conditional_confirmation_initializations':[1,2],
        'official_data_split':'Hard64000training divided57600train/6400validation by r5_audit.py family partition. Official8000test retained and only scored after validation selection. OriginalSingleAtomic test was already seen by prior project.',
        'epochs_evaluated':list(EPOCHS),'full_epochs':20,'batch':BATCH,'updates_each_full_run':36000,'device':'MPSfloat32,CPU2threads',
        'optimizer':'Adam lr=.0005; global grad norm clip1; full-image MSE; no augmentation; no learning-rate tuning',
        'initialization':'950000+init; plain/C4exactsameparametersandinitialweights. Object1081892vsplain1266467parameters; not parameter-matched.',
        'sampling':'Whole training scenes shuffled each complete epoch, NumPyseed952000+init. All arms share exact20orders; no replacement within an epoch.',
        'model_information':'Only source RGB input and target RGB image loss for all primary models. No true masks, factors, positions or rule IDs at inference/training. Object arm3learned slots includes background capacity; object correspondence is not assumed.',
        'selection':'Per-model smallest validation full-image MSE among5/10/20epochs; ties choose earlier epoch. Also report fixed20epoch test as matched-training-budget secondary comparison. Never select by official test.',
        'development_gate':'Compare validation-selectedC4andobject againstvalidation-selectedplain and inputcopy: fullimageMSE and changed-scenechangedpixelMSE each lower thanboth; preservedpixelMSE<=1.05*plain. Maximizefullimageimprovement; tieskindsorder. Only selectedcandidate+plain get two additionalinit runs onthe same officialsplit; these are not new datasets. Gatefailureprecludesautomaticconfirmation.',
        'reporting':'Full image, changed pixels on changed scenes, preserved pixels, unchanged scene damage, source copy baseline. Attribute readout and integrated disk experiment are additional required work, not satisfied by pixel evaluation alone.',
        'preflight':'CPU/MPSforwardafter5CPUupdates,100GPUtimingupdatesperarm, allreset beforetraining. Timing evidence in model_preflight_v1; modelsourcesmustmatch.',
        'caps':{'initial_R5_stage_seconds':43200,'R5_bytes':30000000000},
        'resume':'Atomic optimizer/RNG/checkpointprogress every900updates; exactsavedorders anddata/sourcehashes. Preserve partial runs onfailures.',
        'sources':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'r5_train.py',HERE/'r5_models.py',HERE/'r5_data.py',HERE/'r5_audit.py',ROOT/'followup/svib_preview_models.py']},
        'audit_protocol_sha256':sha(BASE/'dsprites_audit_protocol.json'),'intake_manifest_sha256':sha(BASE/'data/dsprites_hard/manifest.json')})

def location(task,kind,init):return RUNS/task/f'{kind}_s{init}'

def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)

def context(task,kind,init):
    audit=BASE/'audits/dsprites_hard'/task
    return {'training_protocol_sha256':sha(BASE/'dsprites_training_protocol.json'),'audit_sha256':sha(audit/'summary.json'),'partition_sha256':sha(audit/'partition.npz'),'task':task,'kind':kind,'initialization':init}

def partition(task):return np.load(BASE/'audits/dsprites_hard'/task/'partition.npz')

def aggregate_rows(values):
    v=values.astype(np.float64);changed=v[:,3]>0
    return {'n':len(v),'image_mse':float(v[:,0].mean()),'changed_scene_changed_pixel_mse':float(v[changed,1].mean()) if changed.any() else None,
        'preserved_pixel_mse':float(v[:,2].mean()),'unchanged_scene_mse':float(v[~changed,0].mean()) if (~changed).any() else None,
        'changed_scenes':int(changed.sum()),'unchanged_scenes':int((~changed).sum()),'copy_image_mse':float(v[:,4].mean()),'input_damage_mse':float(v[:,5].mean())}

@torch.no_grad()
def score(model,store,ids,device,split='train',copy=False):
    model.eval();rows=[]
    for first in range(0,len(ids),BATCH):
        x,y=store.batch(split,ids[first:first+BATCH],device);pred=x if copy else model(x);assert torch.isfinite(pred).all()
        rows.append(pixel_metrics(pred,x,y).cpu().numpy())
    return np.concatenate(rows)

def validate_checkpoint(task,kind,init,epoch,store,device):
    out=location(task,kind,init);cp=out/f'epoch{epoch}.pt';dest=out/f'val_epoch{epoch}.json'
    if dest.exists():
        d=json.loads(dest.read_text());assert d['context']==context(task,kind,init) and d['checkpoint_sha256']==sha(cp) and d['rows_sha256']==sha(out/f'val_epoch{epoch}.npy');return d
    model=make(kind).to(device);ck=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);ids=partition(task)['val']
    started=time.perf_counter();values=score(model,store,ids,device);np.save(out/f'val_epoch{epoch}.npy',values)
    result={'context':context(task,kind,init),'epoch':epoch,'checkpoint_sha256':sha(cp),'rows_sha256':sha(out/f'val_epoch{epoch}.npy'),'metrics':aggregate_rows(values),'seconds':time.perf_counter()-started,'official_test_used':False}
    dump(dest,result);del model;torch.mps.empty_cache();print('R5 validation',task,kind,init,epoch,result['metrics'],flush=True);return result

def resource_check():
    started=BASE/'training_started.json'
    if started.exists():assert time.time()-json.loads(started.read_text())['unix']<43200,'R5 initial stage time cap; partial results retained'
    size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file());assert size<30000000000,'R5 storage cap; partial results retained'
    return size

def train(task,kind,init):
    assert task in TASKS and kind in KINDS and init in SEEDS;freeze();assert torch.backends.mps.is_available(),'MPS required, no silentfallback'
    pre=json.loads((BASE/'model_preflight_v1/protocol.json').read_text());assert pre['model_source_sha256']==sha(HERE/'r5_models.py') and pre['data_source_sha256']==sha(HERE/'r5_data.py')
    torch.set_num_threads(2);device=torch.device('mps');out=location(task,kind,init);ctx=context(task,kind,init)
    if (out/'run.json').exists():
        d=json.loads((out/'run.json').read_text());assert d['context']==ctx
        for epoch,h in d['checkpoints'].items():assert sha(out/f'epoch{epoch}.pt')==h
        return d
    out.mkdir(parents=True,exist_ok=True)
    lock(BASE/'training_started.json',json.loads((BASE/'training_started.json').read_text()) if (BASE/'training_started.json').exists() else {'unix':time.time(),'task':task,'note':'Initial12hourR5primarytrainingwallclock; preparationtimingreportedseparately'})
    resource_check();torch.manual_seed(950000+init);model=make(kind).to(device);initial=state_hash(model);opt=torch.optim.Adam(model.parameters(),lr=.0005)
    ids=partition(task)['train'];assert len(ids)==57600;generator=np.random.default_rng(952000+init);orders=np.stack([generator.permutation(ids) for _ in range(20)])
    order_hash=__import__('hashlib').sha256(orders.tobytes()).hexdigest();store=Pairs(task);logs=[];completed=0;prior=0.;progress=out/'progress.pt'
    if progress.exists():
        ck=torch.load(progress,map_location='cpu',weights_only=True);assert ck['context']==ctx and ck['order_sha256']==order_hash
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng']);completed=ck['step'];prior=ck['seconds'];logs=ck['logs']
        for epoch in EPOCHS:
            if epoch*STEPS_EPOCH<=completed:validate_checkpoint(task,kind,init,epoch,store,device)
    torch.mps.synchronize();started=time.perf_counter();validation_seconds=0.;model.train()
    for step in range(completed+1,36001):
        epoch,within=divmod(step-1,STEPS_EPOCH);idx=orders[epoch,within*BATCH:(within+1)*BATCH]
        x,y=store.batch('train',idx,device);pred=model(x);loss=(pred-y).square().mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm);opt.step()
        if step%900==0:
            torch.mps.synchronize();seconds=prior+time.perf_counter()-started-validation_seconds
            logs.append({'step':step,'epoch':step/STEPS_EPOCH,'loss':float(loss.detach().cpu()),'train_seconds':seconds})
            if step in [v*STEPS_EPOCH for v in EPOCHS]:
                cp=out/f'epoch{step//STEPS_EPOCH}.pt';weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
                if cp.exists():
                    old=torch.load(cp,weights_only=True)
                    for k,v in weights.items():torch.testing.assert_close(v,old['state_dict'][k],rtol=0,atol=0)
                else:save_torch(cp,{'state_dict':weights,'context':ctx,'step':step,'epoch':step//STEPS_EPOCH})
            save_torch(progress,{'context':ctx,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'step':step,'seconds':seconds,'logs':logs,'order_sha256':order_hash})
            dump(out/'progress.json',{'step':step,'of':36000,'epoch':step/STEPS_EPOCH,'train_seconds':seconds,'pid':os.getpid(),'task':task,'kind':kind,'init':init});print('R5',task,kind,init,logs[-1],flush=True);resource_check()
            if step in [v*STEPS_EPOCH for v in EPOCHS]:
                t=time.perf_counter();validate_checkpoint(task,kind,init,step//STEPS_EPOCH,store,device);validation_seconds+=time.perf_counter()-t;model.train()
    torch.mps.synchronize();seconds=prior+time.perf_counter()-started-validation_seconds;vals={epoch:validate_checkpoint(task,kind,init,epoch,store,device) for epoch in EPOCHS}
    chosen=min(EPOCHS,key=lambda e:(vals[e]['metrics']['image_mse'],e))
    result={'context':ctx,'status':'trained','epochs':20,'steps':36000,'train_n':57600,'val_n':6400,'order_sha256':order_hash,'initial_state_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),
        'seconds':seconds,'validation_seconds':sum(v['seconds'] for v in vals.values()),'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'checkpoints':{str(e):sha(out/f'epoch{e}.pt') for e in EPOCHS},
        'validation':{str(e):sha(out/f'val_epoch{e}.json') for e in EPOCHS},'selected_epoch':chosen,'selected_only_by_validation':True,'official_test_used':False,'logs':logs}
    dump(out/'run.json',result);store.close();event('R5_dsprites_model_trained',task=task,kind=kind,init=init,selected_epoch=chosen,steps=36000);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('--init',type=int,default=0);a=p.parse_args();print(train(a.task,a.kind,a.init),flush=True)
