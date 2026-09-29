"""Matched retraining with missing environmental inputs, never posthoc-only masking."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_models import Transition,load_split,permute_slots

NAME='dynamics_context_omission_v1'
DATA=[640031,997101,997102]
KINDS=['interaction','wrong_exact','joint_exact']
MASKS={'full':[1,1,1],'no_net':[0,0,1],'no_drag':[1,1,0],'no_context':[0,0,0]}


def world(ds):return HERE/('data/dynamics_world_v1' if ds==640031 else f'data/dynamics_repeats_v1/d{ds}/world')


def source_model(ds,kind,seed):
    return HERE/('runs/dynamics_state_study_v1' if ds==640031 else f'runs/dynamics_repeats_v1/d{ds}/state')/f'variable_force/{kind}_s{seed}'


def observed_context(context,condition):return context*context.new_tensor(MASKS[condition])


def freeze():
    p=HERE/f'configs/{NAME}.json'
    if not p.exists():
        for relative in ['reports/dynamics_state_study_v1/verification.json','reports/dynamics_repeats_v1/state_verification.json','reports/dynamics_repeats_v1/data_verification.json']:
            assert json.loads((HERE/relative).read_text())['all_passed']
        data={};models={};groups=[]
        for ds in DATA:
            manifest=json.loads((world(ds)/'manifest.json').read_text())
            for g in manifest['groups']:
                path=world(ds)/g['mode']/g['split']/'trajectories.npz';data[str(path.relative_to(HERE))]=r.sha(path)
                if g['split'] not in ['train','val']:groups.append({'data_seed':ds,'mode':g['mode'],'split':g['split'],'episodes':g['episodes']})
            for kind in KINDS:
                for seed in [0,1,2]:
                    for f in ['model.pt','run.json']:
                        path=source_model(ds,kind,seed)/f;models[str(path.relative_to(HERE))]=r.sha(path)
        dump(p,{'data_seeds':DATA,'kinds':KINDS,'seeds':[0,1,2],'conditions':MASKS,'train_mode':'variable_force',
            'groups':groups,'steps':12000,'batch':64,'lr':.001,'hidden':48,'threads':2,'clip_norm':1.,
            'model_seed_base':765000,'sampling_seed_base':765100,'expected_new_trainings':81,'expected_conditions':108,
            'data_sha256':data,'full_context_models_sha256':models,
            'source_sha256':{f:r.sha(HERE/f) for f in ['dynamics_context_omission.py','dynamics_models.py','planning_C07_CONTEXT_OMISSION_KO.md']},
            'frozen_unix':time.time(),
            'scope':'Variable-force training only; same fixed checkpoint across all evaluation modes. True current state and action, context masked identically in training and inference. 81 fresh trainings and 27 frozen full-context controls. Not image inference or autonomous rollout.'})
    c=json.loads(p.read_text())
    for family in ['data_sha256','full_context_models_sha256','source_sha256']:
        for path,sha in c[family].items():assert r.sha(HERE/path)==sha,path
    assert len(c['groups'])==39
    return c


def preflight(c):
    torch.set_num_threads(2);data=load_split(world(DATA[0]),'variable_force','train')
    s,a,ctx=data['state'][:64],data['action'][:64],data['context'][:64];tests=[]
    for condition,mask in MASKS.items():
        x=observed_context(ctx,condition);changed=ctx+torch.tensor([.79,-.31,.42])*(1-torch.tensor(mask))
        torch.testing.assert_close(x,observed_context(changed,condition),rtol=0,atol=0)
        for index,seen in enumerate(mask):
            if seen:torch.testing.assert_close(x[:,index],ctx[:,index],rtol=0,atol=0)
            else:assert not x[:,index].any()
        tests.append('input mask preserves observed and excludes hidden: '+condition)
    for condition in ['no_net','no_context']:
        torch.manual_seed(501);left=Transition('wrong_exact');right=Transition('joint_exact')
        with torch.no_grad():left.net[-1].weight.normal_(0,.02);left.net[-1].bias.normal_(0,.02)
        right.load_state_dict(left.state_dict());x=observed_context(ctx,condition)
        p=left(s,a,x);q=right(s,a,x);torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-7)
        loss=(p-data['next'][:64]).square().mean();other=(q-data['next'][:64]).square().mean();loss.backward();other.backward()
        error=0.
        for u,v in zip(left.parameters(),right.parameters()):
            torch.testing.assert_close(u.grad,v.grad,atol=1e-7,rtol=1e-6);error=max(error,float((u.grad-v.grad).abs().max()))
        tests.append({'zero_net_structure_equivalence':condition,'max_gradient_error':error,'max_prediction_error':float((p-q).abs().max().detach())})
    dump(HERE/f'reports/{NAME}/preflight.json',{'all_passed':True,'tests':tests,'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'source_sha256':r.sha(__file__),
        'scope':'Masks and nontrivial initialized forward/gradient equivalence only; no claim that training or evaluation is complete.'})
    print('context omission preflight passed',flush=True)


def train(c,ds,condition,kind,seed,training):
    folder=HERE/f'runs/{NAME}/d{ds}/{condition}/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert run['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
        assert run['checkpoint_sha256']==r.sha(HERE/run['checkpoint_path']);return folder
    if folder.exists():raise RuntimeError(f'Incomplete training retained: {folder}')
    folder.mkdir(parents=True);config_sha=r.sha(HERE/f'configs/{NAME}.json')
    if condition=='full':
        src=source_model(ds,kind,seed);run=json.loads((src/'run.json').read_text());path=src/'model.pt'
        dump(folder/'run.json',{**run,'data_seed':ds,'condition':condition,'kind':kind,'seed':seed,'new_training_steps':0,
            'checkpoint_path':str(path.relative_to(HERE)),'checkpoint_sha256':r.sha(path),'source_run_sha256':r.sha(src/'run.json'),
            'config_sha256':config_sha,'scope':c['scope']});return folder
    dump(folder/'started.json',{'data_seed':ds,'condition':condition,'kind':kind,'seed':seed,'config_sha256':config_sha})
    torch.manual_seed(c['model_seed_base']+seed);model=Transition(kind,c['hidden']);initial=state_digest(model)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(c['sampling_seed_base']+seed)
    samples=hashlib.sha256();permutations=hashlib.sha256();logs=[];start=time.perf_counter()
    context=observed_context(training['context'],condition)
    for step in range(1,c['steps']+1):
        idx=torch.cat([torch.randint(len(training['state']),(32,),generator=rng),torch.randint(len(training['state']),(32,),generator=rng)])
        order=torch.rand(64,4,generator=rng).argsort(-1);samples.update(idx.numpy().tobytes());permutations.update(order.numpy().tobytes())
        state,action,target=permute_slots(training['state'][idx],training['action'][idx],training['next'][idx],order)
        pred=model(state,action,context[idx]);live=state[...,-1:];loss=((pred-target).square()*live).sum()/(4*live.sum())
        if not torch.isfinite(loss):raise FloatingPointError('Omission loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['clip_norm'])
        if not torch.isfinite(norm):raise FloatingPointError('Omission gradient')
        opt.step()
        if step%3000==0:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-start};logs.append(row)
            print('omission train',ds,condition,kind,seed,row,flush=True)
    path=folder/'model.pt';torch.save({'kind':kind,'hidden':c['hidden'],'state_dict':model.state_dict()},path)
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'data_seed':ds,'condition':condition,'kind':kind,'seed':seed,'steps':c['steps'],'new_training_steps':c['steps'],
        'initial_weights_sha256':initial,'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':time.perf_counter()-start,
        'sample_sha256':samples.hexdigest(),'permutation_sha256':permutations.hexdigest(),'loss_log':logs,
        'checkpoint_path':str(path.relative_to(HERE)),'checkpoint_sha256':r.sha(path),'config_sha256':config_sha,
        'training_archive_sha256':r.sha(world(ds)/'variable_force/train/trajectories.npz'),'scope':c['scope']})
    return folder


@torch.no_grad()
def evaluate(folder,c,data):
    if (folder/'metrics.json').exists():return
    run=json.loads((folder/'run.json').read_text());ck=torch.load(HERE/run['checkpoint_path'],map_location='cpu',weights_only=True)
    model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();outputs={};rows=[];metrics={}
    for (mode,split),item in data.items():
        name=mode+'__'+split;context=observed_context(item['context'],run['condition']);parts=[]
        for i in range(0,len(item['state']),128):parts.append(model(item['state'][i:i+128],item['action'][i:i+128],context[i:i+128]))
        pred=torch.cat(parts);assert torch.isfinite(pred).all();outputs[name]=pred;live=item['state'][...,-1]
        error=(pred-item['next']).abs()*pred.new_tensor([31.5,31.5,3,3]);pos=(error[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1))
        vel=(error[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1));pair=item['events'][:,0]>0;wall=item['events'][:,1]>0
        for episode in torch.unique(item['episode']).tolist():
            scene=item['episode']==episode
            for category,mask in [('all',torch.ones_like(scene)),('no_contact',~(pair|wall)),('contact',pair|wall),('pair',pair),('wall_only',wall&~pair)]:
                choose=scene&mask
                if choose.any():rows.append({'mode':mode,'split':split,'episode':episode,'category':category,'transitions':int(choose.sum()),
                    'position_mae_pixels':float(pos[choose].mean()),'velocity_mae_pixels_per_frame':float(vel[choose].mean())})
        metrics[name]={}
        for category in ['all','no_contact','contact','pair','wall_only']:
            selected=[x for x in rows if (x['mode'],x['split'],x['category'])==(mode,split,category)]
            metrics[name][category]={'episodes':len(selected),**{key:r.bootstrap([x[key] for x in selected]) if selected else None
                for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']}}
    torch.save(outputs,folder/'evaluation.pt');r.write_csv(folder/'rows.csv',rows)
    dump(folder/'metrics.json',{'groups':metrics,'data_sha256':{m+'__'+s:r.sha(world(run['data_seed'])/m/s/'trajectories.npz') for m,s in data},
        'checkpoint_sha256':run['checkpoint_sha256'],'config_sha256':run['config_sha256'],
        'files':{f:r.sha(folder/f) for f in ['evaluation.pt','rows.csv']}})
    print('omission evaluated',run['data_seed'],run['condition'],run['kind'],run['seed'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');args=p.parse_args();c=freeze();torch.set_num_threads(c['threads'])
    preflight(c)
    if args.preflight_only:return
    results=[];start=time.time()
    for ds in DATA:
        training=load_split(world(ds),'variable_force','train')
        evaluation={(g['mode'],g['split']):load_split(world(ds),g['mode'],g['split']) for g in c['groups'] if g['data_seed']==ds}
        # Train/validation diagnostics use the same mask, without checkpoint selection.
        evaluation['variable_force','train']=training;evaluation['variable_force','val']=load_split(world(ds),'variable_force','val')
        for condition in MASKS:
            for seed in c['seeds']:
                for kind in KINDS:
                    folder=train(c,ds,condition,kind,seed,training);evaluate(folder,c,evaluation)
                    run=json.loads((folder/'run.json').read_text());metric=json.loads((folder/'metrics.json').read_text())
                    results.append({'data_seed':ds,'condition':condition,'kind':kind,'seed':seed,'run':run,'metrics':metric})
                    dump(HERE/f'reports/{NAME}/progress.json',{'completed_conditions':len(results),'expected_conditions':108,
                        'completed_new_trainings':sum(x['run']['new_training_steps']>0 for x in results),'expected_new_trainings':81,
                        'last':[ds,condition,kind,seed],'seconds':time.time()-start})
    dump(HERE/f'reports/{NAME}/summary.json',{'results':results,'conditions':len(results),'new_trainings':81,
        'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'seconds':time.time()-start,'scope':c['scope']})


if __name__=='__main__':main()
