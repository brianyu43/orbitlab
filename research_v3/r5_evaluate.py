"""Validation-only candidate choice and fully replayed external RGB evaluation.

Pixel metrics alone do not fulfill R5's required attribute and integration work.
Full float32prediction caches are temporary: remove only after every pixel was
replayed and the immutable checksums, rows and examples have been retained.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from v3_common import HERE,sha,dump,lock,event
from r5_intake import BASE,TASKS
from r5_data import Pairs
from r5_models import make,pixel_metrics,METRIC_NAMES
import r5_train as training
from r5_train import (KINDS,EPOCHS,freeze as training_freeze,location,partition,
    context,aggregate_rows,score,BATCH,resource_check,save_torch)

EVAL=BASE/'external/dsprites_hard/evaluation'

def freeze():
    training_freeze()
    return lock(BASE/'dsprites_evaluation_protocol.json',{'version':1,'source_sha256':sha(__file__),'training_protocol_sha256':sha(BASE/'dsprites_training_protocol.json'),
        'test_modes':['validation_selected','fixed20epoch'],'test_n':8000,'batch':BATCH,
        'test_policy':'Score all developmentarms after per-model validation selection and development candidate locking; never choose epochs or candidate usingtest. Same officialtestseenearlierforSingleAtomic.',
        'primary_metrics':METRIC_NAMES,'secondary_aggregation':'Changed error includes only scenes withchangedpixels; preserved andunchanged-scene damage retained. No nonfiniteimages silentlyexcluded.',
        'verification':'Replay all8000float32RGBoutputs andmetricrows per evaluated checkpoint withsameMPSbatch32,atol1e-6. Rehash source/data/weights; repeatpixelmetrics. Notindependentretraining.',
        'temporary_cache':'One1.57GBfloat32prediction cache per activeunit, explicitlytemporary and removedonlyafter fullpixelreplaypasses. RetainoriginalcacheSHA256, per-batchfloat32hashes,8000metricrows and32examples; checkpointreconstructsoutputs.',
        'examples':'First16officialtestindices plus16highestchanged-region-error scenes amongchangedscenes, selectedafterallrowsfordisplayonly. No scoreexclusions.',
        'scope':'Externalpixelstudy only. Objectattribute readout, allnew3Dtasks, matched moving-disk integration andindependenthuman applicability remainseparate requiredwork.'})

def candidate_selection(task):
    assert training.SPLIT_SEED==training.DEVELOPMENT,'Lock candidate on development partition only'
    freeze();path=EVAL/task/'development_selection.json';device=torch.device('mps');torch.set_num_threads(2)
    runs={k:json.loads((location(task,k,0)/'run.json').read_text()) for k in KINDS}
    vals={k:json.loads((location(task,k,0)/f'val_epoch{runs[k]["selected_epoch"]}.json').read_text())['metrics'] for k in KINDS}
    store=Pairs(task);copy=score(make('plain').to(device),store,partition(task)['val'],device,copy=True);store.close();cm=aggregate_rows(copy)
    eligible=[];details=[]
    for k in ['c4','object']:
        v=vals[k];p=vals['plain'];gates={'image_better_than_plain_and_copy':v['image_mse']<min(p['image_mse'],cm['image_mse']),
            'changed_better_than_plain_and_copy':v['changed_scene_changed_pixel_mse']<min(p['changed_scene_changed_pixel_mse'],cm['changed_scene_changed_pixel_mse']),
            'preserved_damage_not_over5percent_worse':v['preserved_pixel_mse']<=1.05*p['preserved_pixel_mse']}
        record={'candidate':k,'gates':gates,'passed':all(gates.values())};details.append(record)
        if record['passed']:eligible.append(k)
    chosen=min(eligible,key=lambda k:vals[k]['image_mse']) if eligible else None
    record={'evaluation_protocol_sha256':sha(BASE/'dsprites_evaluation_protocol.json'),'task':task,'runs':{k:sha(location(task,k,0)/'run.json') for k in KINDS},'selected_epochs':{k:runs[k]['selected_epoch'] for k in KINDS},
        'validation_metrics':vals,'copy_validation':cm,'gates':details,'selected_candidate':chosen,'confirmation_required_if_resources_allow':chosen is not None,'confirmation_started':False,'official_test_used':False}
    lock(path,record);return record

def selected_epoch(task,kind,init,mode):
    if kind=='copy':return None
    run=json.loads((location(task,kind,init)/'run.json').read_text());assert run['context']==context(task,kind,init)
    return run['selected_epoch'] if mode=='validation_selected' else 20

def unit(task,kind,init,mode):
    base=EVAL if training.SPLIT_SEED==training.DEVELOPMENT else BASE/'external/dsprites_hard/confirmation'/f'p{training.SPLIT_SEED}'/'evaluation'
    return base/task/f'{kind}_s{init}'/mode

def provenance(task,kind,init,mode):
    epoch=selected_epoch(task,kind,init,mode);selection=EVAL/task/'development_selection.json';assert selection.exists()
    return {'evaluation_protocol_sha256':sha(BASE/'dsprites_evaluation_protocol.json'),'selection_sha256':sha(selection),
        'data_manifest_sha256':sha(BASE/'data/dsprites_hard/manifest.json'),'audit_sha256':sha(BASE/'audits/dsprites_hard'/task/'summary.json'),
        'partition_seed':training.SPLIT_SEED,'partition_sha256':sha(training.prepare_partition(task,training.SPLIT_SEED)),
        'task':task,'kind':kind,'init':init,'mode':mode,'epoch':epoch,'checkpoint_sha256':sha(location(task,kind,init)/f'epoch{epoch}.pt') if epoch is not None else None}

def load(task,kind,init,mode,device):
    if kind=='copy':return None
    epoch=selected_epoch(task,kind,init,mode);ck=torch.load(location(task,kind,init)/f'epoch{epoch}.pt',map_location='cpu',weights_only=True)
    model=make(kind).to(device);model.load_state_dict(ck['state_dict']);model.eval();return model

@torch.no_grad()
def evaluate(task,kind,init,mode):
    freeze();assert torch.backends.mps.is_available();torch.set_num_threads(2);device=torch.device('mps');folder=unit(task,kind,init,mode);prov=provenance(task,kind,init,mode)
    if (folder/'summary.json').exists():
        old=json.loads((folder/'summary.json').read_text());assert old['provenance']==prov and old['rows_sha256']==sha(folder/'rows.npy');return
    resource_check();folder.mkdir(parents=True,exist_ok=True);model=load(task,kind,init,mode,device);store=Pairs(task)
    cache_path=folder/'prediction_verification_cache.npy';assert not cache_path.exists(),'Partialcachemustbeinspectedbeforecontrolledrecovery'
    cache=np.lib.format.open_memmap(cache_path,mode='w+',shape=(8000,3,128,128),dtype=np.float32);rows=[];hashes=[];started=time.perf_counter()
    for first in range(0,8000,BATCH):
        ids=np.arange(first,min(first+BATCH,8000));x,y=store.batch('test',ids,device);pred=x if kind=='copy' else model(x);assert torch.isfinite(pred).all()
        value=pred.cpu().numpy();cache[first:first+len(ids)]=value;hashes.append(hashlib.sha256(value.tobytes()).hexdigest());rows.append(pixel_metrics(pred,x,y).cpu().numpy())
    cache.flush();values=np.concatenate(rows);np.save(folder/'rows.npy',values);dump(folder/'batch_hashes.json',hashes)
    eligible=np.flatnonzero(values[:,3]>0);worst=eligible[np.argsort(-values[eligible,1],kind='stable')[:16]];ids=np.array(list(dict.fromkeys([*range(16),*worst.tolist()])),np.int64)
    x,y=store.batch('test',ids,torch.device('cpu'));save_torch(folder/'examples.pt',{'test_indices':torch.from_numpy(ids),'source':x,'target':y,'prediction':torch.from_numpy(np.array(cache[ids])),'policy':'first16plusworst16changedforillustrationonly'})
    result={'provenance':prov,'metrics':aggregate_rows(values),'rows_sha256':sha(folder/'rows.npy'),'cache_sha256':sha(cache_path),'batch_hashes_sha256':sha(folder/'batch_hashes.json'),
        'examples_sha256':sha(folder/'examples.pt'),'seconds':time.perf_counter()-started,'cache_policy':'Temporary; removeonlyafterallRGBreplayspass','attributes_evaluated':False}
    dump(folder/'summary.json',result);del cache;store.close();print('R5test',task,kind,init,mode,result['metrics'],flush=True)

@torch.no_grad()
def verify(task,kind,init,mode):
    freeze();assert torch.backends.mps.is_available();torch.set_num_threads(2);device=torch.device('mps');folder=unit(task,kind,init,mode);saved=json.loads((folder/'summary.json').read_text());assert saved['provenance']==provenance(task,kind,init,mode)
    for file,key in [('rows.npy','rows_sha256'),('batch_hashes.json','batch_hashes_sha256'),('examples.pt','examples_sha256')]:assert sha(folder/file)==saved[key]
    if (folder/'verification.json').exists():
        old=json.loads((folder/'verification.json').read_text());assert old['summary_sha256']==sha(folder/'summary.json')
        cp=folder/'prediction_verification_cache.npy'
        if cp.exists():
            assert not old['temporary_cache_removed_after_pass'] and sha(cp)==old['original_temporary_cache_sha256'];cp.unlink()
        if not old['temporary_cache_removed_after_pass']:
            old['temporary_cache_removed_after_pass']=True;dump(folder/'verification.json',old)
        return
    cp=folder/'prediction_verification_cache.npy';assert sha(cp)==saved['cache_sha256'];cached=np.load(cp,mmap_mode='r');model=load(task,kind,init,mode,device);store=Pairs(task);rows=[];maxerr=0.;started=time.perf_counter();bitexact=0
    recorded=json.loads((folder/'batch_hashes.json').read_text())
    for first in range(0,8000,BATCH):
        ids=np.arange(first,min(first+BATCH,8000));x,y=store.batch('test',ids,device);pred=x if kind=='copy' else model(x);a=pred.cpu().numpy();b=np.asarray(cached[first:first+len(ids)])
        err=float(np.abs(a-b).max());maxerr=max(err,maxerr);assert err<=1e-6
        bitexact+=int(hashlib.sha256(a.tobytes()).hexdigest()==recorded[first//BATCH]);rows.append(pixel_metrics(pred,x,y).cpu().numpy())
    actual=np.concatenate(rows);np.testing.assert_allclose(actual,np.load(folder/'rows.npy'),rtol=0,atol=1e-6)
    recomputed=aggregate_rows(actual)
    for k,v in saved['metrics'].items():
        if v is None:assert recomputed[k] is None
        else:assert abs(v-recomputed[k])<=1e-6
    store.close();del cached
    receipt={'summary_sha256':sha(folder/'summary.json'),'all_RGBoutputs_replayed':8000,'all_metricrows_recomputed':8000,'max_abs_pixel_error':maxerr,'bitexact_batches':bitexact,'total_batches':250,
        'original_temporary_cache_sha256':saved['cache_sha256'],'seconds':time.perf_counter()-started,'temporary_cache_removed_after_pass':False,'independent_retraining':False}
    dump(folder/'verification.json',receipt);cp.unlink();receipt['temporary_cache_removed_after_pass']=True;dump(folder/'verification.json',receipt);print('R5fullpixelreplay',task,kind,init,mode,maxerr,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate','verify']);p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--kind',choices=[*KINDS,'copy']);p.add_argument('--init',type=int,default=0);p.add_argument('--mode',choices=['validation_selected','fixed20epoch'],default='validation_selected');p.add_argument('--split-seed',type=int,choices=[training.DEVELOPMENT,*training.CONFIRMATION],default=training.DEVELOPMENT);a=p.parse_args();training.SPLIT_SEED=a.split_seed
    if a.stage=='select':print(candidate_selection(a.task),flush=True)
    else:{'evaluate':evaluate,'verify':verify}[a.stage](a.task,a.kind,a.init,a.mode)
