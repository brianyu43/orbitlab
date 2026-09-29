"""Replay six train/val estimators and check train-only inputs and sampling."""
import hashlib
import json
import numpy as np
import torch
from common import HERE,dump,r,digest_tensor,state_digest
from dynamics_observation_models import ObservationEstimator,supervised_loss
from train_dynamics_observation import NAME,freeze,pooled


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(2);root=HERE/f'reports/{NAME}';study=json.loads((root/'development_summary.json').read_text());count=0;predictions=0
    assert study['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    assert len(study['results'])==6
    for entry in study['results']:
        kind,seed=entry['kind'],entry['seed'];folder=HERE/f'runs/{NAME}/{kind}_s{seed}';run=json.loads((folder/'run.json').read_text());assert entry['run']==run
        assert run['steps']==c['steps'] and run['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
        assert run['checkpoint_sha256']==r.sha(folder/'model.pt')==entry['development']['checkpoint_sha256']
        train,labels=pooled(kind,'train')
        for k,v in train.items():assert digest_tensor(v)==run['training_inputs_sha256'][k]
        for k,v in labels.items():assert digest_tensor(v)==run['training_labels_sha256'][k]
        assert len(labels['state'])==768
        torch.manual_seed(c['model_seed']+seed);model=ObservationEstimator(kind);assert state_digest(model)==run['initial_weights_sha256']
        assert sum(v.numel() for v in model.parameters())==run['parameters']
        rng=torch.Generator().manual_seed(c['sampling_seed']+seed);h=hashlib.sha256()
        for _ in range(c['steps']):h.update(torch.randint(768,(c['batch'],),generator=rng).numpy().tobytes())
        assert h.hexdigest()==run['sampling_sha256'];opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
        assert torch.equal(opt['sampling_rng'],rng.get_state()) and all(int(v['step'])==c['steps'] for v in opt['optimizer']['state'].values())
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
        assert r.sha(folder/'development_predictions.pt')==entry['development']['predictions_sha256']
        saved=torch.load(folder/'development_predictions.pt',map_location='cpu',weights_only=True)
        for split in ['train','val']:
            x,y=pooled(kind,split);parts=[]
            for i in range(0,len(y['state']),32):parts.append(model({k:v[i:i+32] for k,v in x.items()}))
            pred={k:torch.cat([v[k] for v in parts]) for k in ['state','context']}
            for k in pred:torch.testing.assert_close(pred[k],saved[split][k],atol=0,rtol=0)
            p=pred['state'].numpy();t=y['state'].numpy();truth=t[...,5]>.5;present=p[...,5]>=0
            error=np.abs(p[...,:5]-t[...,:5])*np.array([31.5,31.5,3,3,8],np.float32)
            metrics=entry['development']['metrics'][split]
            assert abs(np.mean(present.sum(-1)==truth.sum(-1))-metrics['count_correct'])<1e-7
            assert abs(np.mean(present==truth)-metrics['presence_accuracy'])<1e-7
            np.testing.assert_allclose(error[truth].mean(0),metrics['factor_mae_on_true_color_slots'],atol=2e-5,rtol=0)
            context_error=np.abs(pred['context'].numpy()-y['context'].numpy())*np.array([.1,.1,.01],np.float32)
            np.testing.assert_allclose(context_error.mean(0),metrics['context_mae'],atol=1e-7,rtol=0)
            predictions+=len(t)
        count+=1;print('verified four-frame learned estimator',kind,seed,flush=True)
    for seed in c['seeds']:
        pair=[v['run'] for v in study['results'] if v['seed']==seed]
        assert len({v['sampling_sha256'] for v in pair})==1
    dump(root/'development_verification.json',{'all_passed':True,'models_verified':count,'train_validation_predictions_replayed':predictions,
        'train_only_input_and_label_hashes_verified':True,'matching_episode_sampling_and_optimizer_steps_verified':True,
        'verifier_sha256':r.sha(__file__),'scope':study['scope']})


if __name__=='__main__':main()
