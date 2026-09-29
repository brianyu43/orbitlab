"""Three supervised head initializations on fixed image representations."""
import hashlib
import json
import time
import numpy as np
import torch
from common import HERE,dump,fresh_dir,state_digest,digest_tensor,r,update_status
from object_state_readout import StateReadout,set_loss,measure


def labels(split):
    # Ground truth is supervision/evaluation data, never model input.
    with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as a:return torch.from_numpy(a['state_slots'].copy())


def freeze():
    cp=HERE/'configs/object_readout_v1.json'
    if not cp.exists():dump(cp,{'kinds':['flat','slot'],'seeds':[0,1,2],'steps':6000,'batch':128,'lr':.001,'threads':2,
        'slot_cache_manifest_sha256':r.sha(HERE/'data/object_slot_cache_v1/manifest.json'),
        'model_sha256':r.sha(HERE/'object_state_readout.py'),'runner_sha256':r.sha(__file__),
        'training':'Shared 32-128-128-16 MLP. Train-only feature means/stds. Two-object set loss with learned presence for five predicted slots.',
        'loss_weights':{'shape_ce':1,'color_ce':1,'xy_mean_square':20,'radius_square':5,'pose_mean_square':1,'presence_bce':1},
        'inference':'Only frozen RGB slot features; no true count or assignment. Presence sigmoid >= .5; shape/color/C4 pose decoded by argmax.',
        'selection':'Fixed final 6000 steps, train/validation evaluation only. Heads vary across three seeds, encoder initialization remains one.',
        'claim_boundary':'Supervised state readout, not label-free concept discovery, end-to-end editing or independent test confirmation.',
        'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def features(kind,split):
    return torch.load(HERE/f'data/object_slot_cache_v1/{kind}_{split}.pt',map_location='cpu',weights_only=True)['slots']


def train(c,kind,seed):
    folder=HERE/f'runs/object_readout_v1/{kind}_s{seed}'
    if (folder/'run.json').exists():return folder
    fresh_dir(folder);x=features(kind,'train');target=labels('train');mean=x.mean((0,1));std=x.std((0,1),unbiased=False).clamp_min(1e-4)
    torch.manual_seed(994000+seed);model=StateReadout(mean,std);initial=state_digest(model.net)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(994100+seed)
    sample=hashlib.sha256();logs=[];start=time.perf_counter()
    for step in range(1,c['steps']+1):
        idx=torch.randint(len(x),(c['batch'],),generator=rng);sample.update(idx.numpy().tobytes())
        pred=model(x[idx]);loss,_=set_loss(pred,target[idx])
        if not torch.isfinite(loss):raise FloatingPointError('Readout loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Readout gradient')
        opt.step()
        if step%1000==0:logs.append({'step':step,'loss':float(loss.detach())});print('readout',kind,seed,logs[-1],flush=True)
    elapsed=time.perf_counter()-start
    torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'kind':kind,'seed':seed,'steps':c['steps'],'parameters':sum(p.numel() for p in model.parameters()),
        'training_seconds':elapsed,'initial_trainable_weights_sha256':initial,'sample_sha256':sample.hexdigest(),
        'feature_sha256':digest_tensor(x),'training_label_sha256':digest_tensor(target),'checkpoint_sha256':r.sha(folder/'model.pt'),
        'config_sha256':r.sha(HERE/'configs/object_readout_v1.json'),'loss_log':logs})
    return folder


@torch.no_grad()
def evaluate(folder,c):
    if (folder/'metrics.json').exists():return
    run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=StateReadout(ck['mean'],ck['std']);model.load_state_dict(ck['state_dict']);model.eval();all_raw={};all_matches={};summaries={}
    for split in ['train','val']:
        x=features(run['kind'],split);truth=labels(split);raw=torch.cat([model(batch) for batch in x.split(128)])
        rows,matches=measure(raw,truth);r.write_csv(folder/f'{split}_rows.csv',rows);all_raw[split]=raw;all_matches[split]=matches
        keys=['count_correct','matched_fraction','shape_correct','color_correct','position_within_one_pixel','radius_within_half_pixel','pose_correct','object_all_factors_correct','full_scene_success']
        summaries[split]={key:r.bootstrap([v[key] for v in rows]) for key in keys}
        position=[v['matched_position_mae_pixels'] for v in rows if v['matched_position_mae_pixels'] is not None]
        summaries[split]['matched_position_mae_pixels']=r.bootstrap(position) if position else None
        print('readout eval',run['kind'],run['seed'],split,{key:round(summaries[split][key]['mean'],4) for key in ['count_correct','shape_correct','color_correct','position_within_one_pixel','full_scene_success']},flush=True)
    torch.save(all_raw,folder/'predictions.pt');dump(folder/'assignments.json',all_matches)
    dump(folder/'metrics.json',{'splits':summaries,'checkpoint_sha256':r.sha(folder/'model.pt'),'predictions_sha256':r.sha(folder/'predictions.pt'),
        'claim_boundary':c['claim_boundary']})


def main():
    c=freeze();torch.set_num_threads(c['threads']);results=[]
    update_status('B05','in_progress',['configs/object_readout_v1.json','planning_B05_EXECUTION_KO.md'],note='Supervised state readout from frozen RGB slots is training. No image-intervention success yet; fresh confirmation data and controller transfer pending.')
    for seed in c['seeds']:
        pair=[]
        for kind in c['kinds']:
            folder=train(c,kind,seed);evaluate(folder,c);run=json.loads((folder/'run.json').read_text());pair.append(run)
            results.append({'kind':kind,'seed':seed,'run':run,'metrics':json.loads((folder/'metrics.json').read_text())})
            dump(HERE/'reports/object_readout_v1/progress.json',{'completed_models':len(results),'expected_models':6,'results':results})
        for key in ['initial_trainable_weights_sha256','sample_sha256','parameters']:assert pair[0][key]==pair[1][key]
    dump(HERE/'reports/object_readout_v1/summary.json',{'results':results,'trainings':6,'config_sha256':r.sha(HERE/'configs/object_readout_v1.json'),
        'claim_boundary':c['claim_boundary']})


if __name__=='__main__':main()
