"""Replay fresh estimator train/val predictions and audit matched training inputs."""
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,digest_tensor,state_digest
from dynamics_repeat_data import NAME
from dynamics_repeat_observation import CONFIG,freeze,pooled
from dynamics_observation_models import ObservationEstimator


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(c['threads']);summary=HERE/f'reports/{NAME}/observation_development_summary.json'
    all_results=json.loads(summary.read_text());assert all_results['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
    expected={(ds,kind,seed) for ds in c['data_seeds'] for kind in c['kinds'] for seed in c['seeds']}
    listed=[(v['data_seed'],v['kind'],v['seed']) for v in all_results['results']]
    assert set(listed)==expected and len(listed)==12
    reports=[];predictions=metrics_checked=0
    for ds in c['data_seeds']:
        for kind in c['kinds']:
            inputs={};labels={}
            for split in ['train','val']:
                inputs[split],labels[split]=pooled(ds,kind,split)
                assert len(labels[split]['state'])==(768 if split=='train' else 192)
                # Independent supervision construction directly from the raw train/val archive.
                truth=[];context=[]
                for mode in c['modes']:
                    with np.load(HERE/f'data/{NAME}/d{ds}/world/{mode}/{split}/trajectories.npz') as a:
                        raw=a['states'][:,3].copy();ctx=a['contexts'].copy()
                    target=np.zeros((len(raw),6,6),np.float32)
                    for i,state in enumerate(raw):
                        for s in state:
                            if s[6]>.5:target[i,int(s[5])]=[*s[:5],1]
                    target[...,:5]/=np.array([31.5,31.5,3,3,8],np.float32);truth.append(target)
                    context.append(np.c_[ctx[:,:2]+ctx[:,2:4],ctx[:,4]].astype(np.float32)/np.array([.1,.1,.01],np.float32))
                    if kind=='rgb_cnn':
                        with np.load(HERE/f'data/{NAME}/d{ds}/inputs/{mode}/{split}/observations.npz') as a:assert a.files==['images']
                np.testing.assert_array_equal(labels[split]['state'].numpy(),np.concatenate(truth))
                np.testing.assert_array_equal(labels[split]['context'].numpy(),np.concatenate(context))
            for seed in c['seeds']:
                folder=HERE/f'runs/{NAME}/d{ds}/observation/{kind}_s{seed}'
                run=json.loads((folder/'run.json').read_text());development=json.loads((folder/'development.json').read_text())
                assert run['data_seed']==ds and run['kind']==kind and run['seed']==seed and run['steps']==4000
                assert run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
                assert run['checkpoint_sha256']==development['checkpoint_sha256']==r.sha(folder/'model.pt')
                assert development['predictions_sha256']==r.sha(folder/'development_predictions.pt')
                assert run['training_inputs_sha256']=={k:digest_tensor(v) for k,v in inputs['train'].items()}
                assert run['training_labels_sha256']=={k:digest_tensor(v) for k,v in labels['train'].items()}
                rng=torch.Generator().manual_seed(c['sampling_seed']+seed);sampling=hashlib.sha256()
                for _ in range(4000):sampling.update(torch.randint(768,(32,),generator=rng).numpy().tobytes())
                assert run['sampling_sha256']==sampling.hexdigest()
                opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
                assert torch.equal(opt['sampling_rng'],rng.get_state()) and all(int(v['step'])==4000 for v in opt['optimizer']['state'].values())
                torch.manual_seed(c['model_seed']+seed);model=ObservationEstimator(kind)
                old=json.loads((HERE/f'runs/dynamics_observation_models_v1/{kind}_s{seed}/run.json').read_text())
                assert state_digest(model)==run['initial_weights_sha256']==old['initial_weights_sha256']
                assert sampling.hexdigest()==old['sampling_sha256']
                assert sum(v.numel() for v in model.parameters())==run['parameters']==old['parameters']
                ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
                saved=torch.load(folder/'development_predictions.pt',map_location='cpu',weights_only=True)
                for split in ['train','val']:
                    x,y=inputs[split],labels[split];parts=[]
                    for i in range(0,len(y['state']),32):parts.append(model({k:v[i:i+32] for k,v in x.items()}))
                    out={k:torch.cat([p[k] for p in parts]) for k in ['state','context']}
                    for k in out:torch.testing.assert_close(out[k],saved[split][k],atol=0,rtol=0)
                    pred=out['state'].numpy();target=y['state'].numpy();actual=target[...,5]>.5;presence=pred[...,5]>=0
                    factor_error=np.abs(pred[...,:5]-target[...,:5])*np.array([31.5,31.5,3,3,8])
                    ctx_error=np.abs(out['context'].numpy()-y['context'].numpy())*np.array([.1,.1,.01])
                    expected={'count_correct':float(np.mean(presence.sum(-1)==actual.sum(-1))),
                        'presence_accuracy':float(np.mean(presence==actual)),
                        'factor_mae_on_true_color_slots':factor_error[actual].mean(0),
                        'context_mae':ctx_error.mean(0)}
                    for key,value in expected.items():
                        np.testing.assert_allclose(development['metrics'][split][key],value,atol=2e-6,rtol=2e-6);metrics_checked+=1
                    predictions+=len(y['state'])
                entry=next(v for v in all_results['results'] if (v['data_seed'],v['kind'],v['seed'])==(ds,kind,seed))
                assert entry['run']==run and entry['development']==development
                reports.append({'data_seed':ds,'kind':kind,'seed':seed,'checkpoint_sha256':run['checkpoint_sha256']})
                print('verified repeat observation model',ds,kind,seed,flush=True)
    dump(HERE/f'reports/{NAME}/observation_development_verification.json',{'all_passed':True,'models_verified':len(reports),
        'train_validation_predictions_replayed':predictions,'independent_metric_checks':metrics_checked,
        'train_only_input_and_label_hashes_checked':True,'original_initialization_sampling_and_optimizer_steps_matched':True,
        'summary_sha256':r.sha(summary),'verifier_sha256':r.sha(__file__),'results':reports,
        'scope':'New-data estimator training/development audit. Heldout inference, state substitution, and long-rollout repeat evaluations remain separate.'})


if __name__=='__main__':main()
