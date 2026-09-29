"""Audit matched initializations, sampling and every train/validation prediction."""
import argparse
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,digest_tensor,state_digest
from dynamics_observation_length_data import NAME
from dynamics_observation_length_models import LengthEstimator,pooled
from train_observation_length import CONFIG,freeze


def independently_check_labels(ds,split,labels,c):
    truth=[];context=[]
    for mode in c['modes']:
        with np.load(HERE/f'data/{NAME}/d{ds}/world/{mode}/{split}/trajectories.npz') as a:
            raw=a['states'][:,7].copy();ctx=a['contexts'].copy()
        target=np.zeros((len(raw),6,6),np.float32)
        for i,scene in enumerate(raw):
            for obj in scene:
                if obj[6]>.5:target[i,int(obj[5])]=[*obj[:5],1]
        target[...,:5]/=np.array([31.5,31.5,3,3,8],np.float32);truth.append(target)
        context.append(np.c_[ctx[:,:2]+ctx[:,2:4],ctx[:,4]].astype(np.float32)/np.array([.1,.1,.01],np.float32))
    np.testing.assert_array_equal(labels['state'].numpy(),np.concatenate(truth))
    np.testing.assert_array_equal(labels['context'].numpy(),np.concatenate(context))


@torch.no_grad()
def verify(c,ds,length,kind,seed,inputs,labels):
    folder=HERE/f'runs/{NAME}/d{ds}/L{length}/{kind}_s{seed}'
    run=json.loads((folder/'run.json').read_text());development=json.loads((folder/'development.json').read_text())
    assert (run['data_seed'],run['length'],run['kind'],run['seed'])==(ds,length,kind,seed)
    assert run['steps']==4000 and run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
    assert run['checkpoint_sha256']==development['checkpoint_sha256']==r.sha(folder/'model.pt')
    assert development['predictions_sha256']==r.sha(folder/'development_predictions.pt')
    assert run['training_inputs_sha256']=={k:digest_tensor(v) for k,v in inputs['train'].items()}
    assert run['training_labels_sha256']=={k:digest_tensor(v) for k,v in labels['train'].items()}
    rng=torch.Generator().manual_seed(c['sampling_seed']+seed);sampling=hashlib.sha256()
    for _ in range(4000):sampling.update(torch.randint(768,(32,),generator=rng).numpy().tobytes())
    assert run['sampling_sha256']==sampling.hexdigest()
    optimizer=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
    assert torch.equal(optimizer['sampling_rng'],rng.get_state())
    assert all(int(v['step'])==4000 for v in optimizer['optimizer']['state'].values())
    torch.manual_seed(c['model_seed']+seed);model=LengthEstimator(kind)
    assert state_digest(model)==run['initial_weights_sha256']
    assert sum(p.numel() for p in model.parameters())==run['parameters']
    preflight=json.loads((HERE/f'reports/{NAME}/model_preflight.json').read_text())
    initial=next(x for x in preflight['initializations'] if (x['kind'],x['seed'])==(kind,seed))
    assert initial['initial_weights_sha256']==run['initial_weights_sha256'] and initial['parameters']==run['parameters']
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);assert ck['length']==length and ck['kind']==kind
    model.load_state_dict(ck['state_dict']);model.eval();saved=torch.load(folder/'development_predictions.pt',map_location='cpu',weights_only=True)
    predictions=checks=0
    for split in ['train','val']:
        x,y=inputs[split],labels[split];parts=[]
        for i in range(0,len(y['state']),32):parts.append(model({k:v[i:i+32] for k,v in x.items()}))
        out={k:torch.cat([v[k] for v in parts]) for k in ['state','context']}
        for key in out:torch.testing.assert_close(out[key],saved[split][key],atol=0,rtol=0)
        pred=out['state'].numpy();truth=y['state'].numpy();alive=truth[...,5]>.5;present=pred[...,5]>=0
        errors=abs(pred[...,:5]-truth[...,:5])*np.array([31.5,31.5,3,3,8])
        context_errors=abs(out['context'].numpy()-y['context'].numpy())*np.array([.1,.1,.01])
        metrics={'count_correct':np.mean(present.sum(-1)==alive.sum(-1)),'presence_accuracy':np.mean(present==alive),
            'factor_mae_on_true_color_slots':errors[alive].mean(0),'context_mae':context_errors.mean(0)}
        for key,value in metrics.items():np.testing.assert_allclose(development['metrics'][split][key],value,atol=2e-6,rtol=2e-6);checks+=1
        predictions+=len(truth)
    return {'all_passed':True,'data_seed':ds,'length':length,'kind':kind,'seed':seed,'predictions_replayed':predictions,
        'metric_checks':checks,'initial_weights_sha256':run['initial_weights_sha256'],'sampling_sha256':run['sampling_sha256'],
        'training_labels_sha256':run['training_labels_sha256'],'parameters':run['parameters'],
        'artifacts':{str((folder/f).relative_to(HERE)):r.sha(folder/f) for f in ['model.pt','optimizer.pt','run.json','development.json','development_predictions.pt']},
        'verifier_sha256':r.sha(__file__)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--available',action='store_true');args=parser.parse_args()
    c=freeze();torch.set_num_threads(c['threads']);results=[]
    for ds in c['data_seeds']:
        for kind in c['kinds']:
            for length in c['lengths']:
                inputs={};labels={}
                for seed in c['seeds']:
                    folder=HERE/f'runs/{NAME}/d{ds}/L{length}/{kind}_s{seed}'
                    if not (folder/'development.json').exists():
                        if args.available:continue
                        raise RuntimeError(f'Model not yet complete: {folder}')
                    receipt=HERE/f'reports/{NAME}/model_receipts/d{ds}_L{length}_{kind}_s{seed}.json'
                    if receipt.exists():
                        result=json.loads(receipt.read_text());assert result['verifier_sha256']==r.sha(__file__)
                        for p,sha in result['artifacts'].items():assert r.sha(HERE/p)==sha,p
                    else:
                        if not inputs:
                            for split in ['train','val']:
                                inputs[split],labels[split]=pooled(ds,kind,split,length)
                                assert len(labels[split]['state'])==(768 if split=='train' else 192)
                                independently_check_labels(ds,split,labels[split],c)
                        result=verify(c,ds,length,kind,seed,inputs,labels);dump(receipt,result)
                    results.append(result);print('verified length model',ds,length,kind,seed,flush=True)
    for kind in c['kinds']:
        for seed in c['seeds']:
            selected=[x for x in results if (x['kind'],x['seed'])==(kind,seed)]
            assert len({x['initial_weights_sha256'] for x in selected})<=1
            assert len({x['sampling_sha256'] for x in selected})<=1
            assert len({x['parameters'] for x in selected})<=1
            for ds in c['data_seeds']:
                assert len({json.dumps(x['training_labels_sha256'],sort_keys=True) for x in selected if x['data_seed']==ds})<=1
    complete=len(results)==72
    report={'all_passed':complete,'all_available_checked_passed':True,'models_verified':len(results),'expected_models':72,
        'train_validation_predictions_replayed':sum(x['predictions_replayed'] for x in results),
        'independent_metric_checks':sum(x['metric_checks'] for x in results),'results':results,
        'same_architecture_initialization_and_sampling_across_lengths':True,'same_targets_across_lengths':True,
        'verifier_sha256':r.sha(__file__),'scope':'Training and train/validation inference audit; heldout and future predictions remain separate.'}
    root=HERE/f'reports/{NAME}';dump(root/'model_verification_progress.json',report)
    if complete:
        summary=root/'development_summary.json';s=json.loads(summary.read_text());assert len(s['results'])==72
        assert s['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
        keys=[(x['data_seed'],x['length'],x['kind'],x['seed']) for x in s['results']]
        assert len(set(keys))==72 and set(keys)=={(x['data_seed'],x['length'],x['kind'],x['seed']) for x in results}
        for x in s['results']:
            folder=HERE/f"runs/{NAME}/d{x['data_seed']}/L{x['length']}/{x['kind']}_s{x['seed']}"
            assert x['run']==json.loads((folder/'run.json').read_text()) and x['development']==json.loads((folder/'development.json').read_text())
        report['summary_sha256']=r.sha(summary);dump(root/'development_verification.json',report)
    print('length model audit',len(results),'/72',flush=True)


if __name__=='__main__':main()
