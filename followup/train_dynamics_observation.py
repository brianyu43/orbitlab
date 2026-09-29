"""Fixed-budget supervised RGB and measurement estimators; train/val only."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from common import HERE,dump,fresh_dir,r,digest_tensor,state_digest
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_rgb_baseline import NAME as BASELINE
from dynamics_observation_models import ObservationEstimator,load_inputs,load_labels,supervised_loss,decode

NAME='dynamics_observation_models_v1'
MODES=['isotropic','fixed_gravity','variable_force']


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():dump(path,{'kinds':['rgb_cnn','measurement_mlp'],'seeds':[0,1,2],'modes':MODES,'steps':4000,'batch':32,'lr':.0005,
        'model_seed':996000,'sampling_seed':996100,'threads':2,
        'inputs_manifest_sha256':r.sha(HERE/f'data/{INPUTS}/manifest.json'),'measurement_manifest_sha256':r.sha(HERE/f'data/{BASELINE}/manifest.json'),
        'sources':{p:r.sha(HERE/p) for p in ['dynamics_observation_models.py','train_dynamics_observation.py','dynamics_rgb_baseline.py']},
        'training':'Pool the 768 initial four-frame train windows across three environments. No mode input, no future frames, one window per original episode. Supervise t=3 states and constant net acceleration/drag using train labels only.',
        'selection':'Common fixed 4000-step budget; evaluate train/val only before held-out protocol. No best-checkpoint/seed selection. Three initializations per architecture.',
        'normalization':'Fixed declared physical scales. No fitted normalization. Measurement features clipped to +/-5; analytic base velocity +/-6px/frame, base force +/-0.2px/frame^2, base drag 0..0.05.',
        'outputs':'Six color-addressed state proposals and net force/drag. Presence threshold .5. Positive radius/drag via softplus. Distinct observable colors provide track addresses, not hidden persistent IDs.',
        'loss':{'present_object_five_factor_mse':1,'presence_bce':.2,'normalized_context_mse':.2},
        'priors':'Direct CNN uses four RGB frames and fixed color-addressed supervision. Measurement MLP additionally uses known-palette circle fits and analytic dynamics estimates. Parameter counts and execution devices/time differ.',
        'claim_boundary':'Supervised synthetic-video state/environment inference. Not unsupervised discovery, autonomous rollout, or independent data-seed confirmation.',
        'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for p,sha in c['sources'].items():assert r.sha(HERE/p)==sha,p
    assert c['inputs_manifest_sha256']==r.sha(HERE/f'data/{INPUTS}/manifest.json') and c['measurement_manifest_sha256']==r.sha(HERE/f'data/{BASELINE}/manifest.json')
    return c


def pooled(kind,split):
    parts=[load_inputs(mode,split,kind) for mode in MODES];labels=[load_labels(mode,split) for mode in MODES]
    return {k:torch.cat([v[k] for v in parts]) for k in parts[0]},{k:torch.cat([v[k] for v in labels]) for k in labels[0]}


def train(c,kind,seed,device):
    folder=HERE/f'runs/{NAME}/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert r.sha(folder/'model.pt')==run['checkpoint_sha256'];return folder
    fresh_dir(folder);inputs,labels=pooled(kind,'train');n=len(labels['state']);assert n==768
    input_hashes={k:digest_tensor(v) for k,v in inputs.items()};label_hashes={k:digest_tensor(v) for k,v in labels.items()}
    inputs={k:v.to(device) for k,v in inputs.items()};labels={k:v.to(device) for k,v in labels.items()}
    torch.manual_seed(c['model_seed']+seed);model=ObservationEstimator(kind).to(device);initial=state_digest(model);opt=torch.optim.Adam(model.parameters(),lr=c['lr'])
    rng=torch.Generator().manual_seed(c['sampling_seed']+seed);sampling=hashlib.sha256();logs=[];start=time.perf_counter()
    dump(folder/'started.json',{'device':str(device),'kind':kind,'seed':seed,'config_sha256':r.sha(HERE/f'configs/{NAME}.json')})
    for step in range(1,c['steps']+1):
        idx=torch.randint(n,(c['batch'],),generator=rng);sampling.update(idx.numpy().tobytes());idx=idx.to(device)
        pred=model({k:v[idx] for k,v in inputs.items()});loss,terms=supervised_loss(pred,{k:v[idx] for k,v in labels.items()})
        if not torch.isfinite(loss):raise FloatingPointError('Observation loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Observation gradient')
        opt.step()
        if step%1000==0:
            if device.type=='mps':torch.mps.synchronize()
            row={'step':step,'loss':float(loss.detach()),**{k:float(v) for k,v in terms.items()},'seconds':time.perf_counter()-start};logs.append(row)
            print('observation training',kind,seed,row,flush=True)
    if device.type=='mps':torch.mps.synchronize()
    elapsed=time.perf_counter()-start
    torch.save({'kind':kind,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'kind':kind,'seed':seed,'steps':c['steps'],'device':str(device),'training_seconds':elapsed,
        'parameters':sum(v.numel() for v in model.parameters()),'initial_weights_sha256':initial,'sampling_sha256':sampling.hexdigest(),
        'training_inputs_sha256':input_hashes,'training_labels_sha256':label_hashes,'checkpoint_sha256':r.sha(folder/'model.pt'),
        'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'log':logs})
    return folder


@torch.no_grad()
def evaluate(folder):
    if (folder/'development.json').exists():return json.loads((folder/'development.json').read_text())
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=ObservationEstimator(ck['kind']);model.load_state_dict(ck['state_dict']);model.eval()
    outputs={};metrics={}
    for split in ['train','val']:
        x,y=pooled(ck['kind'],split);p=[];ctx=[]
        for i in range(0,len(y['state']),32):
            out=model({k:v[i:i+32] for k,v in x.items()});p.append(out['state']);ctx.append(out['context'])
        pred={'state':torch.cat(p),'context':torch.cat(ctx)};outputs[split]=pred
        truth=y['state'];alive=truth[...,5]>.5;present=pred['state'][...,5]>=0
        error=(pred['state'][...,:5]-truth[...,:5]).abs()*truth.new_tensor([31.5,31.5,3,3,8])
        metrics[split]={'count_correct':float((present.sum(-1)==alive.sum(-1)).float().mean()),
            'presence_accuracy':float((present==alive).float().mean()),
            'factor_mae_on_true_color_slots':error[alive].mean(0).tolist(),
            'context_mae':((pred['context']-y['context']).abs()*y['context'].new_tensor([.1,.1,.01])).mean(0).tolist()}
    torch.save(outputs,folder/'development_predictions.pt')
    report={'metrics':metrics,'checkpoint_sha256':r.sha(folder/'model.pt'),'predictions_sha256':r.sha(folder/'development_predictions.pt'),
        'scope':'Pooled train/validation diagnostics, evaluated on CPU. Factor errors are at true color slots even if presence is missed; presence/count are reported separately.'}
    dump(folder/'development.json',report);print('observation development',folder.name,metrics['val'],flush=True);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--cnn-device',choices=['cpu','mps'],default='mps');args=parser.parse_args();c=freeze()
    if args.prepare_only:return
    torch.set_num_threads(c['threads'])
    if args.cnn_device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Requested MPS device is unavailable; do not silently switch or duplicate runs')
    reports=[]
    for seed in c['seeds']:
        for kind in c['kinds']:
            device=torch.device(args.cnn_device if kind=='rgb_cnn' else 'cpu');folder=train(c,kind,seed,device);metrics=evaluate(folder)
            reports.append({'kind':kind,'seed':seed,'run':json.loads((folder/'run.json').read_text()),'development':metrics})
            dump(HERE/f'reports/{NAME}/progress.json',{'completed_models':len(reports),'expected_models':6})
            if torch.backends.mps.is_available():torch.mps.empty_cache()
    dump(HERE/f'reports/{NAME}/development_summary.json',{'results':reports,'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'scope':c['claim_boundary']})


if __name__=='__main__':main()
