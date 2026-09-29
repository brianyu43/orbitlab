"""Fixed 24-model external preview experiment; heldout results never select a run."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from common import HERE,dump,r,digest_tensor,state_digest
from svib_preview_data import NAME,ALPHAS,alpha_name
from svib_preview_baselines import freeze as baseline_freeze
from svib_preview_models import PreviewPredictor
from svib_preview_metrics import load_images,score,summarize

CONFIG='svib_preview_models_v1'


def freeze():
    baseline_freeze();preflight=HERE/f'reports/{NAME}/model_preflight.json';p=json.loads(preflight.read_text());assert p['all_passed']
    for f,sha in p['sources'].items():assert r.sha(HERE/f)==sha
    baseline=HERE/f'reports/{NAME}/baselines/verification.json';v=json.loads(baseline.read_text());assert v['all_passed']
    assert v['summary_sha256']==r.sha(baseline.parent/'summary.json')
    path=HERE/f'configs/{CONFIG}.json'
    if not path.exists():dump(path,{'alphas':ALPHAS,'kinds':['plain','c4'],'seeds':[0,1,2],'steps':2000,'batch':8,'lr':.0005,
        'gradient_clip':1.,'model_seed':773000,'sampling_seed':773100,'threads':2,'loss':'Full-image mean squared error, source RGB -> target RGB.',
        'data_config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'data_manifest_sha256':r.sha(HERE/f'data/{NAME}/manifest.json'),
        'preflight_sha256':r.sha(preflight),'baseline_verification_sha256':r.sha(baseline),
        'sources':{f:r.sha(HERE/f) for f in ['train_svib_preview.py','svib_preview_models.py','svib_preview_metrics.py','planning_B06_SVIB_PREVIEW_KO.md']},
        'selection':'Fixed final checkpoint for every kind/alpha/seed. No validation/test choice, early stopping or source checkpoint reuse.',
        'compute':'Same parameters and optimizer steps, different forward compute: C4 averages four rotated applications. Wall times on this shared local host are descriptive.',
        'scope':'24 small predictors on official 100-pair-per-alpha preview, with 70 training pairs each. No full SVIB benchmark or independent-data replication claim.',
        'frozen_unix':time.time()})
    c=json.loads(path.read_text());assert c['data_config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    assert c['data_manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json')
    assert c['preflight_sha256']==r.sha(preflight) and c['baseline_verification_sha256']==r.sha(baseline)
    for f,sha in c['sources'].items():assert r.sha(HERE/f)==sha
    return c


def train(c,alpha,kind,seed,device):
    folder=HERE/f'runs/{NAME}/{alpha_name(alpha)}/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
        assert run['checkpoint_sha256']==r.sha(folder/'model.pt');return folder
    if folder.exists():raise RuntimeError(f'Incomplete training preserved: {folder}')
    folder.mkdir(parents=True);source,target,_=load_images(f'{alpha_name(alpha)}/train');assert len(source)==70
    hashes={'source':digest_tensor(source),'target':digest_tensor(target)};source=source.to(device);target=target.to(device)
    torch.manual_seed(c['model_seed']+seed);model=PreviewPredictor(kind).to(device);initial=state_digest(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(c['sampling_seed']+seed)
    sampling=hashlib.sha256();logs=[];started=time.perf_counter()
    dump(folder/'started.json',{'alpha':alpha,'kind':kind,'seed':seed,'device':str(device),'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json')})
    for step in range(1,c['steps']+1):
        index=torch.randint(70,(c['batch'],),generator=rng);sampling.update(index.numpy().tobytes());index=index.to(device)
        prediction=model(source[index]);loss=(prediction-target[index]).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('SVIB preview training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
        if not torch.isfinite(norm):raise FloatingPointError('SVIB preview gradient')
        optimizer.step()
        if step%500==0:
            if device.type=='mps':torch.mps.synchronize()
            row={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-started};logs.append(row)
            print('SVIB preview training',alpha,kind,seed,row,flush=True)
    if device.type=='mps':torch.mps.synchronize()
    seconds=time.perf_counter()-started
    torch.save({'kind':kind,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}},folder/'model.pt')
    torch.save({'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'alpha':alpha,'kind':kind,'seed':seed,'device':str(device),'steps':c['steps'],'training_seconds':seconds,
        'initial_weights_sha256':initial,'sampling_sha256':sampling.hexdigest(),'training_tensors_sha256':hashes,
        'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':r.sha(folder/'model.pt'),
        'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'loss_log':logs})
    return folder


@torch.no_grad()
def evaluate(folder):
    run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=PreviewPredictor(ck['kind']);model.load_state_dict(ck['state_dict']);model.eval();results=[]
    for split in ['train','val','test_id','heldout']:
        out=folder/split
        if (out/'metrics.json').exists():
            result=json.loads((out/'metrics.json').read_text())
            for f,sha in result['files'].items():assert r.sha(out/f)==sha
            results.append(result);continue
        if out.exists():raise RuntimeError(f'Incomplete evaluation retained: {out}')
        out.mkdir();address='heldout' if split=='heldout' else f"{alpha_name(run['alpha'])}/{split}"
        source,target,masks=load_images(address);start=time.perf_counter()
        pred=torch.cat([model(source[i:i+16]) for i in range(0,len(source),16)]);seconds=time.perf_counter()-start
        rows=score(pred.numpy(),source.numpy(),target.numpy(),masks)
        np.savez_compressed(out/'predictions.npz',predictions=pred.numpy());r.write_csv(out/'rows.csv',rows)
        equivariance=None
        if split=='heldout':
            mae=[];maximum=[]
            for k in [1,2,3]:
                rotated=torch.cat([model(torch.rot90(source[i:i+16],k,(-2,-1))) for i in range(0,len(source),16)])
                error=(rotated-torch.rot90(pred,k,(-2,-1))).abs().flatten(1)
                mae.append(error.mean(1).numpy());maximum.append(error.max(1).values.numpy())
            ma=np.stack(mae,1);mx=np.stack(maximum,1)
            if run['kind']=='c4':assert mx.max()<=2e-6
            np.savez_compressed(out/'equivariance.npz',mae=ma,max_abs=mx)
            equivariance={'episodes':len(source),'rotations':[1,2,3],'mean_absolute_gap':float(ma.mean()),'maximum_absolute_gap':float(mx.max()),
                'scope':'RGB output rotation consistency on every external preview input, not target-image accuracy.'}
        result={'alpha':run['alpha'],'kind':run['kind'],'seed':run['seed'],'split':split,'address':address,'episodes':len(source),
            'metrics':summarize(rows),'equivariance':equivariance,'cpu_forward_seconds':seconds,'forward_batch_size':16,
            'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':run['config_sha256'],
            'input_sha256':r.sha(HERE/f'data/{NAME}/{address}/inputs.npz'),'target_sha256':r.sha(HERE/f'data/{NAME}/{address}/labels.npz'),
            'files':{p.name:r.sha(p) for p in out.iterdir() if p.is_file()}}
        dump(out/'metrics.json',result);results.append(result)
    dump(folder/'evaluation.json',{'results':results,'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':run['config_sha256']})
    print('SVIB preview evaluation',run['alpha'],run['kind'],run['seed'],flush=True)
    return results


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--device',choices=['mps','cpu'],default='mps');args=parser.parse_args();c=freeze()
    if args.prepare_only:return
    torch.set_num_threads(c['threads']);device=torch.device(args.device)
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS requested but unavailable; no silent CPU switch')
    results=[];start=time.time()
    for alpha in c['alphas']:
        for seed in c['seeds']:
            for kind in c['kinds']:
                folder=train(c,alpha,kind,seed,device);evaluation=evaluate(folder)
                results.append({'alpha':alpha,'kind':kind,'seed':seed,'run':json.loads((folder/'run.json').read_text()),'evaluation':evaluation})
                dump(HERE/f'reports/{NAME}/training_progress.json',{'completed_models':len(results),'expected_models':24,'last':[alpha,kind,seed],'seconds':time.time()-start})
                if torch.backends.mps.is_available():torch.mps.empty_cache()
    dump(HERE/f'reports/{NAME}/model_summary.json',{'results':results,'models':24,'evaluated_prediction_instances':4800,
        'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'scope':c['scope']})


if __name__=='__main__':main()
