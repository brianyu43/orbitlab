"""Twelve fresh C04 estimator trainings on the two verified new data seeds."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from common import HERE,dump,r,digest_tensor,state_digest
from dynamics_repeat_data import NAME,freeze as freeze_parent
from dynamics_observation_models import ObservationEstimator,measurement_inputs,supervised_loss

CONFIG='dynamics_repeat_observation_v1'
MODES=['isotropic','fixed_gravity','variable_force']


def freeze():
    parent=freeze_parent();verification=HERE/f'reports/{NAME}/observation_data_verification.json'
    assert json.loads(verification.read_text())['all_passed'];path=HERE/f'configs/{CONFIG}.json'
    if not path.exists():
        old=json.loads((HERE/'configs/dynamics_observation_models_v1.json').read_text())
        keys=['kinds','seeds','modes','steps','batch','lr','model_seed','sampling_seed','threads','normalization','outputs','loss','priors']
        dump(path,{**{k:old[k] for k in keys},'data_seeds':parent['new_data_seeds'],
            'parent_config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
            'observation_manifest_sha256':r.sha(HERE/f'data/{NAME}/observation_manifest.json'),
            'data_verification_sha256':r.sha(verification),
            'source_sha256':{p:r.sha(HERE/p) for p in ['dynamics_repeat_observation.py','dynamics_observation_models.py','train_dynamics_observation.py']},
            'scope':'Same fixed C04 training protocol on two new train archives, twelve fresh models. Train/val diagnostics only here; frozen heldout and autonomous evaluations remain separate.',
            'initialization':'Original model and sampling seeds crossed with new data seeds. No old trained weights loaded.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for p,sha in c['source_sha256'].items():assert r.sha(HERE/p)==sha,p
    assert c['observation_manifest_sha256']==r.sha(HERE/f'data/{NAME}/observation_manifest.json')
    assert c['data_verification_sha256']==r.sha(verification)
    return c


def load_inputs(data_seed,mode,split,kind):
    root=HERE/f'data/{NAME}/d{data_seed}'
    if kind=='rgb_cnn':
        with np.load(root/f'inputs/{mode}/{split}/observations.npz') as a:
            assert a.files==['images'];return {'images':torch.from_numpy(a['images'].copy()).permute(0,1,4,2,3).float()/255}
    with np.load(root/f'measurements/{mode}/{split}/predictions.npz') as a:
        assert set(a.files)=={'features','states','contexts'}
        return measurement_inputs(a['features'],a['states'],a['contexts'])


def load_labels(data_seed,mode,split):
    with np.load(HERE/f'data/{NAME}/d{data_seed}/inputs/{mode}/{split}/labels.npz') as a:
        state=a['past_states'][:,-1].copy();context=np.c_[a['net_acceleration'],a['drag']]
    target=np.zeros((len(state),6,6),np.float32)
    for i,ss in enumerate(state):
        for s in ss:
            if s[4]>0:target[i,int(s[5])]=[*s[:5],1]
    target=torch.from_numpy(target);target[...,:5]/=target.new_tensor([31.5,31.5,3,3,8])
    return {'state':target,'context':torch.tensor(context,dtype=torch.float32)/torch.tensor([.1,.1,.01])}


def pooled(data_seed,kind,split):
    inputs=[load_inputs(data_seed,mode,split,kind) for mode in MODES]
    labels=[load_labels(data_seed,mode,split) for mode in MODES]
    return ({k:torch.cat([x[k] for x in inputs]) for k in inputs[0]},
            {k:torch.cat([x[k] for x in labels]) for k in labels[0]})


def train(c,data_seed,kind,seed,device):
    folder=HERE/f'runs/{NAME}/d{data_seed}/observation/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert run['checkpoint_sha256']==r.sha(folder/'model.pt')
        assert run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json');return folder
    if folder.exists():raise RuntimeError(f'Incomplete estimator training retained: {folder}')
    folder.mkdir(parents=True);x,y=pooled(data_seed,kind,'train');n=len(y['state']);assert n==768
    input_hash={k:digest_tensor(v) for k,v in x.items()};label_hash={k:digest_tensor(v) for k,v in y.items()}
    x={k:v.to(device) for k,v in x.items()};y={k:v.to(device) for k,v in y.items()}
    torch.manual_seed(c['model_seed']+seed);model=ObservationEstimator(kind).to(device);initial=state_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(c['sampling_seed']+seed)
    sampling=hashlib.sha256();logs=[];started=time.perf_counter()
    dump(folder/'started.json',{'data_seed':data_seed,'kind':kind,'seed':seed,'device':str(device),'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json')})
    for step in range(1,c['steps']+1):
        indices=torch.randint(n,(c['batch'],),generator=rng);sampling.update(indices.numpy().tobytes());indices=indices.to(device)
        prediction=model({k:v[indices] for k,v in x.items()});loss,terms=supervised_loss(prediction,{k:v[indices] for k,v in y.items()})
        if not torch.isfinite(loss):raise FloatingPointError('Repeat observation loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Repeat observation gradient')
        optimizer.step()
        if step%1000==0:
            if device.type=='mps':torch.mps.synchronize()
            row={'step':step,'loss':float(loss.detach()),**{k:float(v) for k,v in terms.items()},'seconds':time.perf_counter()-started}
            logs.append(row);print('repeat observation train',data_seed,kind,seed,row,flush=True)
    if device.type=='mps':torch.mps.synchronize()
    seconds=time.perf_counter()-started
    torch.save({'kind':kind,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},folder/'model.pt')
    torch.save({'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'data_seed':data_seed,'kind':kind,'seed':seed,'steps':c['steps'],'device':str(device),
        'training_seconds':seconds,'parameters':sum(v.numel() for v in model.parameters()),'initial_weights_sha256':initial,
        'sampling_sha256':sampling.hexdigest(),'training_inputs_sha256':input_hash,'training_labels_sha256':label_hash,
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'log':logs})
    return folder


@torch.no_grad()
def evaluate(folder):
    if (folder/'development.json').exists():return json.loads((folder/'development.json').read_text())
    run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=ObservationEstimator(ck['kind']);model.load_state_dict(ck['state_dict']);model.eval();outputs={};metrics={}
    for split in ['train','val']:
        x,y=pooled(run['data_seed'],ck['kind'],split);parts=[]
        for i in range(0,len(y['state']),32):parts.append(model({k:v[i:i+32] for k,v in x.items()}))
        prediction={k:torch.cat([v[k] for v in parts]) for k in ['state','context']};outputs[split]=prediction
        truth=y['state'];alive=truth[...,5]>.5;present=prediction['state'][...,5]>=0
        error=(prediction['state'][...,:5]-truth[...,:5]).abs()*truth.new_tensor([31.5,31.5,3,3,8])
        metrics[split]={'count_correct':float((present.sum(-1)==alive.sum(-1)).float().mean()),
            'presence_accuracy':float((present==alive).float().mean()),'factor_mae_on_true_color_slots':error[alive].mean(0).tolist(),
            'context_mae':((prediction['context']-y['context']).abs()*truth.new_tensor([.1,.1,.01])).mean(0).tolist()}
    torch.save(outputs,folder/'development_predictions.pt')
    report={'metrics':metrics,'checkpoint_sha256':r.sha(folder/'model.pt'),'predictions_sha256':r.sha(folder/'development_predictions.pt'),
            'scope':'Pooled train/validation CPU predictions at the fixed budget; no heldout model/seed selection.'}
    dump(folder/'development.json',report);print('repeat observation development',run['data_seed'],folder.name,metrics['val'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--cnn-device',choices=['cpu','mps'],default='mps');args=parser.parse_args()
    c=freeze()
    if args.prepare_only:return
    torch.set_num_threads(c['threads'])
    if args.cnn_device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Requested MPS is unavailable; no silent device switch')
    results=[];started=time.time()
    for ds in c['data_seeds']:
        for seed in c['seeds']:
            for kind in c['kinds']:
                device=torch.device(args.cnn_device if kind=='rgb_cnn' else 'cpu');folder=train(c,ds,kind,seed,device)
                development=evaluate(folder);results.append({'data_seed':ds,'kind':kind,'seed':seed,
                    'run':json.loads((folder/'run.json').read_text()),'development':development})
                dump(HERE/f'reports/{NAME}/observation_progress.json',{'completed_models':len(results),'expected_models':12,
                    'last':[ds,kind,seed],'seconds':time.time()-started})
                if torch.backends.mps.is_available():torch.mps.empty_cache()
    dump(HERE/f'reports/{NAME}/observation_development_summary.json',{'results':results,'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'scope':c['scope']})


if __name__=='__main__':main()
