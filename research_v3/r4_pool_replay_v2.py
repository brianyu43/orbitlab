"""Replay immutable R4pools using their actual six-thread inference setting.

The standalone original `r4_train.py pools` entry point did not call the trainer's
two-thread setup. The original default was6, reproduced bit-for-bit. No cache,
normalizer, model, data, score or numeric tolerance is changed by this verifier.
"""
import json
import torch
from v3_common import HERE,sha,dump,lock
from r4_data import BASE,DEVELOPMENT
from r4_train import freeze,root,data,load_model
from r4_core import o,oracle_codes

@torch.no_grad()
def main():
    freeze();run=root(DEVELOPMENT,0);folder=run/'pools';manifest=json.loads((folder/'manifest.json').read_text())
    lock(BASE/'pool_replay_protocol_v2.json',{'source_sha256':sha(__file__),'original_manifest_sha256':sha(folder/'manifest.json'),
        'cache_inference_threads':6,'training_and_primary_evaluation_threads':2,'replay_tolerance':0,
        'reason':'Originalstandalonepoolsentrypoint usedTorchdefault6threads, whereas originalverifierforced2; recovered6setting reproduces originalcachebit-for-bit. Training itself callsset_num_threads2.',
        'mutation_policy':'Originaltraining/evaluationprotocols,sources,pools,normalizersandcheckpointsunchanged. This is a provenance/verification correction, not a changed scientificmetricthreshold.'})
    for n,h in manifest['files'].items():assert sha(folder/n)==h
    models={rep:load_model(DEVELOPMENT,0,f'ae_{rep}')[0] for rep in ['base','spatial']};norm=torch.load(folder/'normalization.pt',map_location='cpu',weights_only=True);checks=[]
    for split in ['train','val','ood']:
        torch.set_num_threads(6);x,y,meta=data(DEVELOPMENT,split);saved=torch.load(folder/f'{split}.pt',map_location='cpu',weights_only=True)
        torch.testing.assert_close(y.repeat(4,1),saved['labels'],rtol=0,atol=0)
        raw={rep:[] for rep in ['base','spatial','oracle']}
        for k in range(4):
            raw['oracle'].append(oracle_codes(y,meta,k))
            for rep,model in models.items():raw[rep].append(torch.cat([model.encode(o.rotate(x[i:i+64],k)) for i in range(0,len(x),64)]))
        for rep,values in raw.items():
            value=torch.cat(values);torch.testing.assert_close(value,saved['raw'][rep],rtol=0,atol=0)
            if split=='train':
                torch.testing.assert_close(value.mean((0,1),keepdim=True),norm[rep]['mean'],rtol=0,atol=0)
                torch.testing.assert_close(value.std((0,1),unbiased=False,keepdim=True).clamp_min(.02),norm[rep]['std'],rtol=0,atol=0)
            torch.testing.assert_close((value-norm[rep]['mean'])/norm[rep]['std'],saved['z'][rep],rtol=0,atol=0)
            for k in range(4):torch.testing.assert_close(values[k],o.rho(values[0],k),rtol=0,atol=1e-5)
        torch.testing.assert_close(saved['center'],saved['raw']['oracle'][...,10:12],rtol=0,atol=0)
        # Quantify the actual2thread setting used by training/evaluation separately.
        torch.set_num_threads(2);deltas={}
        for rep,model in models.items():
            two=torch.cat([torch.cat([model.encode(o.rotate(x[i:i+64],k)) for i in range(0,len(x),64)]) for k in range(4)])
            delta=(two-saved['raw'][rep]).abs();deltas[rep]={'max_abs':float(delta.max()),'differing_elements':int((delta>0).sum()),'total_elements':delta.numel()}
        checks.append({'split':split,'images_per_encoder':len(x)*4,'all_original_codes_and_normalization_bitexact':True,'two_thread_diagnostic':deltas});print('R4poolbitexact',split,deltas,flush=True)
    dump(folder/'verification.json',{'manifest_sha256':sha(folder/'manifest.json'),'checks':checks,'cache_inference_threads':6,'train_only_normalization_recomputed_bitexact':True,'source_sha256':sha(__file__),'amendment_protocol_sha256':sha(BASE/'pool_replay_protocol_v2.json')})

if __name__=='__main__':main()
