"""Seventy-two matched-budget estimators for paired observation lengths."""
import argparse
import hashlib
import json
import time
import torch
from common import HERE,dump,r,digest_tensor,state_digest
from dynamics_observation_length_data import NAME,DATA
from dynamics_observation_length_models import LengthEstimator,pooled
from dynamics_observation_models import supervised_loss
from prepare_observation_length_inputs import freeze as input_freeze

CONFIG='dynamics_observation_length_models_v1'


def freeze():
    input_freeze();path=HERE/f'configs/{CONFIG}.json'
    evidence=[f'data/{NAME}/input_manifest.json',f'reports/{NAME}/input_verification.json',f'reports/{NAME}/model_preflight.json']
    for p in evidence[1:]:assert json.loads((HERE/p).read_text())['all_passed']
    verified=json.loads((HERE/evidence[1]).read_text());preflight=json.loads((HERE/evidence[2]).read_text())
    assert verified['manifest_sha256']==preflight['manifest_sha256']==r.sha(HERE/evidence[0])
    assert verified['verifier_sha256']==r.sha(HERE/'verify_observation_length_inputs.py')
    for p,sha in preflight['source_sha256'].items():assert r.sha(HERE/p)==sha,p
    if not path.exists():
        old=json.loads((HERE/'configs/dynamics_observation_models_v1.json').read_text())
        keys=['kinds','seeds','modes','steps','batch','lr','model_seed','sampling_seed','threads','normalization','outputs','loss']
        dump(path,{**{k:old[k] for k in keys},'data_seeds':DATA,'lengths':[1,2,4,8],
            'input_config_sha256':r.sha(HERE/'configs/dynamics_observation_length_inputs_v1.json'),
            'evidence_sha256':{p:r.sha(HERE/p) for p in evidence},
            'source_sha256':{p:r.sha(HERE/p) for p in ['train_observation_length.py','dynamics_observation_length_models.py',
                'dynamics_observation_length_reference.py','dynamics_observation_models.py','planning_C07_OBSERVATION_LENGTH_KO.md']},
            'training':'768 pooled train episodes per data seed. Predict t=7 state/context; no hidden frames, future frames, true count/ID/mask or mode input. Same initial weights and sampling order across L.',
            'selection':'All final 4000-step checkpoints evaluated. No length/checkpoint/seed selection from validation or heldout performance.',
            'priors':'CNN uses last L RGB frames and availability flags. MLP also uses known-palette circle measurements and visible-only free-flight estimates. Same capacity within architecture across L.',
            'scope':'Train and validation diagnostics at t=7; heldout substitutions, autonomous forecasts and cross-data paired length analysis remain separate.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for family in ['evidence_sha256','source_sha256']:
        for p,sha in c[family].items():assert r.sha(HERE/p)==sha,p
    assert c['input_config_sha256']==r.sha(HERE/'configs/dynamics_observation_length_inputs_v1.json')
    return c


def train(c,ds,length,kind,seed,device):
    folder=HERE/f'runs/{NAME}/d{ds}/L{length}/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert run['checkpoint_sha256']==r.sha(folder/'model.pt')
        assert run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json');return folder
    if folder.exists():raise RuntimeError(f'Incomplete estimator training retained: {folder}')
    folder.mkdir(parents=True);x,y=pooled(ds,kind,'train',length);n=len(y['state']);assert n==768
    input_hash={k:digest_tensor(v) for k,v in x.items()};label_hash={k:digest_tensor(v) for k,v in y.items()}
    x={k:v.to(device) for k,v in x.items()};y={k:v.to(device) for k,v in y.items()}
    torch.manual_seed(c['model_seed']+seed);model=LengthEstimator(kind).to(device);initial=state_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(c['sampling_seed']+seed)
    sampling=hashlib.sha256();logs=[];started=time.perf_counter()
    dump(folder/'started.json',{'data_seed':ds,'length':length,'kind':kind,'seed':seed,'device':str(device),
        'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json')})
    for step in range(1,c['steps']+1):
        indices=torch.randint(n,(c['batch'],),generator=rng);sampling.update(indices.numpy().tobytes());indices=indices.to(device)
        output=model({k:v[indices] for k,v in x.items()});loss,terms=supervised_loss(output,{k:v[indices] for k,v in y.items()})
        if not torch.isfinite(loss):raise FloatingPointError('Length estimator loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Length estimator gradient')
        optimizer.step()
        if step%1000==0:
            if device.type=='mps':torch.mps.synchronize()
            row={'step':step,'loss':float(loss.detach()),**{k:float(v) for k,v in terms.items()},'seconds':time.perf_counter()-started}
            logs.append(row);print('length train',ds,length,kind,seed,row,flush=True)
    if device.type=='mps':torch.mps.synchronize()
    seconds=time.perf_counter()-started
    torch.save({'kind':kind,'length':length,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},folder/'model.pt')
    torch.save({'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'data_seed':ds,'length':length,'kind':kind,'seed':seed,'steps':c['steps'],'device':str(device),
        'training_seconds':seconds,'parameters':sum(v.numel() for v in model.parameters()),'initial_weights_sha256':initial,
        'sampling_sha256':sampling.hexdigest(),'training_inputs_sha256':input_hash,'training_labels_sha256':label_hash,
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'log':logs})
    return folder


@torch.no_grad()
def evaluate(folder):
    if (folder/'development.json').exists():
        v=json.loads((folder/'development.json').read_text());assert v['predictions_sha256']==r.sha(folder/'development_predictions.pt');return v
    run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=LengthEstimator(ck['kind']);model.load_state_dict(ck['state_dict']);model.eval();outputs={};metrics={}
    for split in ['train','val']:
        x,y=pooled(run['data_seed'],ck['kind'],split,run['length']);parts=[]
        for i in range(0,len(y['state']),32):parts.append(model({k:v[i:i+32] for k,v in x.items()}))
        prediction={k:torch.cat([v[k] for v in parts]) for k in ['state','context']};outputs[split]=prediction
        truth=y['state'];alive=truth[...,5]>.5;present=prediction['state'][...,5]>=0
        error=(prediction['state'][...,:5]-truth[...,:5]).abs()*truth.new_tensor([31.5,31.5,3,3,8])
        metrics[split]={'count_correct':float((present.sum(-1)==alive.sum(-1)).float().mean()),
            'presence_accuracy':float((present==alive).float().mean()),'factor_mae_on_true_color_slots':error[alive].mean(0).tolist(),
            'context_mae':((prediction['context']-y['context']).abs()*truth.new_tensor([.1,.1,.01])).mean(0).tolist()}
    torch.save(outputs,folder/'development_predictions.pt')
    report={'metrics':metrics,'checkpoint_sha256':r.sha(folder/'model.pt'),'predictions_sha256':r.sha(folder/'development_predictions.pt'),
            'scope':'Pooled train/validation CPU predictions at fixed budget; no heldout selection.'}
    dump(folder/'development.json',report);print('length development',run['data_seed'],run['length'],folder.name,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--cnn-device',choices=['cpu','mps'],default='mps');args=parser.parse_args();c=freeze()
    if args.prepare_only:return
    torch.set_num_threads(c['threads'])
    if args.cnn_device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Requested MPS unavailable; no silent device switch')
    results=[];started=time.time()
    for ds in c['data_seeds']:
        for seed in c['seeds']:
            for length in c['lengths']:
                for kind in c['kinds']:
                    device=torch.device(args.cnn_device if kind=='rgb_cnn' else 'cpu');folder=train(c,ds,length,kind,seed,device)
                    results.append({'data_seed':ds,'length':length,'kind':kind,'seed':seed,
                        'run':json.loads((folder/'run.json').read_text()),'development':evaluate(folder)})
                    dump(HERE/f'reports/{NAME}/training_progress.json',{'completed_models':len(results),'expected_models':72,
                        'last':[ds,length,kind,seed],'seconds':time.time()-started})
                    if torch.backends.mps.is_available():torch.mps.empty_cache()
    dump(HERE/f'reports/{NAME}/development_summary.json',{'results':results,'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'scope':c['scope']})


if __name__=='__main__':main()
