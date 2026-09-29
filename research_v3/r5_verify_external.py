"""Full validation forward replay and checkpoint/update accounting, not retraining."""
import argparse,json,time
import numpy as np
import torch
from v3_common import HERE,sha,dump,lock
from r5_intake import BASE,TASKS
from r5_models import make
from r4_core import state_hash
from r5_attribute_evaluate import backend

def verify(family,task,kind,init=0,split_seed=890101):
    mod=backend(family);tr=mod.training;tr.SPLIT_SEED=split_seed;tr.freeze();torch.set_num_threads(2);assert torch.backends.mps.is_available();device=torch.device('mps')
    out=tr.location(task,kind,init);run=json.loads((out/'run.json').read_text());ctx=tr.context(task,kind,init);assert run['context']==ctx and run['steps']==36000 and run['epochs']==20
    protocol={'source_sha256':sha(__file__),'run_sha256':sha(out/'run.json'),'context':ctx,'scope':'All3validationcheckpoints rerun onall6400validationimages, everymetricrow rechecked, epochchoice andsameexampleorder reproduced,finalweights+optimizerstep accounting. Not independent training or bitwisevalidationRGB comparison (originalvalidationRGBcaches were not retained).'}
    lock(out/'verification_protocol.json',protocol)
    if (out/'verification.json').exists():
        receipt=json.loads((out/'verification.json').read_text());assert receipt['protocol_sha256']==sha(out/'verification_protocol.json');return
    started=time.perf_counter();p=tr.partition(task);ids=p['train'];assert len(ids)==57600 and len(p['val'])==6400 and not(set(ids)&set(p['val']))
    rng=np.random.default_rng(952000+init);orders=np.stack([rng.permutation(ids) for _ in range(20)])
    order_hash=__import__('hashlib').sha256(orders.tobytes()).hexdigest();assert order_hash==run['order_sha256']
    torch.manual_seed(950000+init);model=make(kind);assert state_hash(model)==run['initial_state_sha256'];assert sum(v.numel() for v in model.parameters())==run['parameters']
    progress=torch.load(out/'progress.pt',map_location='cpu',weights_only=True);assert progress['step']==36000 and progress['context']==ctx and progress['order_sha256']==order_hash
    counts=[int(v['step']) for v in progress['optimizer']['state'].values() if 'step' in v];assert counts and all(v==36000 for v in counts)
    final=torch.load(out/'epoch20.pt',map_location='cpu',weights_only=True)
    for k,v in final['state_dict'].items():torch.testing.assert_close(v,progress['state_dict'][k],rtol=0,atol=0)
    store=tr.Pairs(task);vals={};replays=[]
    for epoch in (5,10,20):
        cp=out/f'epoch{epoch}.pt';vp=out/f'val_epoch{epoch}.json';rp=out/f'val_epoch{epoch}.npy';v=json.loads(vp.read_text())
        assert sha(cp)==run['checkpoints'][str(epoch)]==v['checkpoint_sha256'] and sha(vp)==run['validation'][str(epoch)] and sha(rp)==v['rows_sha256']
        ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['context']==ctx and ck['step']==1800*epoch and ck['epoch']==epoch
        model=make(kind).to(device);model.load_state_dict(ck['state_dict']);model.eval();rows=tr.score(model,store,p['val'],device)
        old=np.load(rp);assert rows.shape==old.shape==(6400,6);np.testing.assert_allclose(rows,old,rtol=0,atol=1e-6)
        actual=tr.aggregate_rows(rows)
        for k,expected in v['metrics'].items():
            if expected is None:assert actual[k] is None
            else:assert abs(actual[k]-expected)<=1e-6
        vals[epoch]=v['metrics']['image_mse'];replays.append({'epoch':epoch,'all_validation_forwards_repeated':6400,'all_metricrows_compared':6400,'max_abs_metric_difference':float(np.max(np.abs(rows-old)))})
        del model;torch.mps.empty_cache()
    store.close();chosen=min(vals,key=lambda e:(vals[e],e));assert chosen==run['selected_epoch']
    dump(out/'verification.json',{'protocol_sha256':sha(out/'verification_protocol.json'),'full_validation_replays':replays,'selected_epoch_recomputed':chosen,'initial_weights_recomputed':True,'all20epoch_exampleorders_recomputed':True,
        'optimizer_parameter_step_counts':{'parameters':len(counts),'min':min(counts),'max':max(counts)},'final_weights_match_progress_bitexact':True,'seconds':time.perf_counter()-started,'independent_retraining':False})
    print('R5 full validation and training accounting',family,task,kind,init,split_seed,'passed',flush=True)

def semantic_summary(family,task):
    mod=backend(family);tr=mod.training;labels=np.load(BASE/'audits'/f'{family}_hard'/task/'factors.npz');unchanged=np.all(labels['source'][64000:]==labels['target'][64000:],axis=(1,2));records=[]
    for kind in ('copy','plain','c4','object'):
        for mode in (['validation_selected'] if kind=='copy' else ['validation_selected','fixed20epoch']):
            path=mod.unit(task,kind,0,mode);d=json.loads((path/'summary.json').read_text());rows=np.load(path/'rows.npy');assert sha(path/'rows.npy')==d['rows_sha256']
            metrics={name:tr.aggregate_rows(rows[mask]) if mask.any() else None for name,mask in [('semantic_unchanged',unchanged),('semantic_changed',~unchanged)]}
            records.append({'kind':kind,'mode':mode,'pixel_summary_sha256':sha(path/'summary.json'),'groups':metrics})
    dest=BASE/'external'/f'{family}_hard'/'evaluation'/task/'semantic_scene_groups.json'
    dump(dest,{'source_sha256':sha(__file__),'audit_sha256':sha(BASE/'audits'/f'{family}_hard'/task/'summary.json'),'scope':'Separates scenes whose true object attributes are unchanged from exactRGBchangedpixelgroups; no primarygate or selection altered. In3D,lighting/shadows/noise may affect pixels beyond object attributes.','records':records})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','semantics']);p.add_argument('--family',choices=['dsprites','clevr','clevrtex'],default='dsprites');p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--kind',choices=['plain','c4','object']);p.add_argument('--init',type=int,default=0);p.add_argument('--split-seed',type=int,choices=[890101,890201,890202,890203],default=890101);a=p.parse_args()
    if a.stage=='training':verify(a.family,a.task,a.kind,a.init,a.split_seed)
    else:semantic_summary(a.family,a.task)
