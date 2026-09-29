"""126 fresh state-predictor trainings using the unchanged C03 protocol."""
import hashlib
import json
import time
import numpy as np
import torch
from common import HERE,dump,r,state_digest
from dynamics_models import Transition,load_split,permute_slots,rotate_state,rotate_vec,rotate_context,rotate_motion
from dynamics_repeat_data import NAME,freeze as freeze_parent

CONFIG='dynamics_repeat_state_v1'


def freeze():
    parent=freeze_parent();path=HERE/f'configs/{CONFIG}.json'
    verification=HERE/f'reports/{NAME}/data_verification.json'
    assert json.loads(verification.read_text())['all_passed']
    if not path.exists():
        old=json.loads((HERE/'configs/dynamics_state_study_v1.json').read_text())
        keys=['modes','kinds','seeds','steps','batch','lr','hidden','threads','evaluation_splits','force_ood_modes']
        dump(path,{**{k:old[k] for k in keys},'data_seeds':parent['new_data_seeds'],
            'model_seed_base':765000,'sampling_seed_base':765100,'augmentation_seed_base':765200,
            'parent_config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
            'data_manifest_sha256':r.sha(HERE/f'data/{NAME}/manifest.json'),'data_verification_sha256':r.sha(verification),
            'source_sha256':{p:r.sha(HERE/p) for p in ['dynamics_repeat_state.py','dynamics_models.py','dynamics_state_study.py']},
            'scope':'All 126 models start from their original protocol initialization, trained only on the corresponding NEW train archive. Uniform teacher-forced transitions. No old checkpoint reuse or tuning on new heldout outcomes.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    assert c['parent_config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    assert c['data_manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json') and c['data_verification_sha256']==r.sha(verification)
    for path,sha in c['source_sha256'].items():assert r.sha(HERE/path)==sha,path
    return c


def train(c,data_seed,mode,kind,seed):
    root=HERE/f'data/{NAME}/d{data_seed}/world';folder=HERE/f'runs/{NAME}/d{data_seed}/state/{mode}/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text())
        assert run['checkpoint_sha256']==r.sha(folder/'model.pt') and run['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
        assert run['training_archive_sha256']==r.sha(root/mode/'train/trajectories.npz')
        return folder
    if folder.exists():raise RuntimeError(f'Incomplete training retained for diagnosis: {folder}')
    folder.mkdir(parents=True);data=load_split(root,mode,'train')
    assert data['states_sequence'].shape==(256,20,4,12)
    torch.manual_seed(c['model_seed_base']+seed);model=Transition(kind,c['hidden']);initial=state_digest(model)
    opt=torch.optim.Adam(model.parameters(),lr=c['lr']);rng=torch.Generator().manual_seed(c['sampling_seed_base']+seed)
    aug=torch.Generator().manual_seed(c['augmentation_seed_base']+seed);samples=hashlib.sha256();permutations=hashlib.sha256()
    started=time.perf_counter();logs=[]
    dump(folder/'started.json',{'data_seed':data_seed,'mode':mode,'kind':kind,'seed':seed,'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json')})
    for step in range(1,c['steps']+1):
        idx=torch.cat([torch.randint(len(data['state']),(32,),generator=rng),torch.randint(len(data['state']),(32,),generator=rng)])
        order=torch.rand(64,4,generator=rng).argsort(-1);samples.update(idx.numpy().tobytes());permutations.update(order.numpy().tobytes())
        s,a,target=permute_slots(data['state'][idx],data['action'][idx],data['next'][idx],order);ctx=data['context'][idx]
        if kind=='augmented':
            k=int(torch.randint(4,(),generator=aug));s=rotate_state(s,k);a=rotate_vec(a,k);target=rotate_motion(target,k);ctx=rotate_context(ctx,k)
        prediction=model(s,a,ctx);mask=s[...,-1:];loss=((prediction-target).square()*mask).sum()/(4*mask.sum())
        if not torch.isfinite(loss):raise FloatingPointError('Repeat-state nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Repeat-state nonfinite gradient')
        opt.step()
        if step%3000==0:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-started};logs.append(row)
            print('repeat state train',data_seed,mode,kind,seed,row,flush=True)
    seconds=time.perf_counter()-started
    torch.save({'kind':kind,'hidden':c['hidden'],'state_dict':model.state_dict()},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state(),'augmentation_rng':aug.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'data_seed':data_seed,'mode':mode,'kind':kind,'seed':seed,'steps':c['steps'],
        'new_training_steps':c['steps'],'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':seconds,
        'initial_weights_sha256':initial,'sample_sha256':samples.hexdigest(),'permutation_sha256':permutations.hexdigest(),
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),
        'training_archive_sha256':r.sha(root/mode/'train/trajectories.npz'),'loss_log':logs})
    return folder


@torch.no_grad()
def evaluate(folder,c):
    if (folder/'metrics.json').exists():return
    run=json.loads((folder/'run.json').read_text());root=HERE/f'data/{NAME}/d{run["data_seed"]}/world'
    ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=Transition(ck['kind'],ck['hidden'])
    model.load_state_dict(ck['state_dict']);model.eval();outputs={};rows=[];summaries={};equiv=[]
    for split in c['evaluation_splits']:
        if split=='force_ood' and run['mode'] not in c['force_ood_modes']:continue
        data=load_split(root,run['mode'],split);parts=[]
        for i in range(0,len(data['state']),128):parts.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
        pred=torch.cat(parts);assert torch.isfinite(pred).all();outputs[split]=pred;mask=data['state'][...,-1]
        error=(pred-data['next']).abs()*pred.new_tensor([31.5,31.5,3,3])
        pos=(error[...,:2].sum(-1)*mask).sum(-1)/(2*mask.sum(-1));vel=(error[...,2:].sum(-1)*mask).sum(-1)/(2*mask.sum(-1))
        pair=data['events'][:,0]>0;wall=data['events'][:,1]>0
        for episode in torch.unique(data['episode']).tolist():
            scene=data['episode']==episode
            for category,condition in [('all',torch.ones_like(scene)),('no_contact',~(pair|wall)),('contact',pair|wall),('pair',pair),('wall_only',wall&~pair)]:
                selected=scene&condition
                if selected.any():rows.append({'split':split,'episode':episode,'category':category,'transitions':int(selected.sum()),
                    'position_mae_pixels':float(pos[selected].mean()),'velocity_mae_pixels_per_frame':float(vel[selected].mean())})
        summaries[split]={}
        for category in ['all','no_contact','contact','pair','wall_only']:
            selected=[v for v in rows if v['split']==split and v['category']==category]
            summaries[split][category]={'episodes':len(selected),**{key:r.bootstrap([v[key] for v in selected]) for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']}}
        if split=='test':
            idx=torch.arange(32)*64+3;s=data['state'][idx];a=data['action'][idx];ctx=data['context'][idx];base=model(s,a,ctx)
            for k in [1,2,3]:
                for label,context in [('joint',rotate_context(ctx,k)),('state_only',ctx)]:
                    out=model(rotate_state(s,k),rotate_vec(a,k),context)
                    err=(out-rotate_motion(base,k)).abs()*s.new_tensor([31.5,31.5,3,3])
                    equiv.append({'rotation':k,'condition':label,'max_physical_coordinate_error':float(err.max())})
    torch.save(outputs,folder/'evaluation.pt');r.write_csv(folder/'rows.csv',rows)
    dump(folder/'metrics.json',{'splits':summaries,'equivariance':equiv,'checkpoint_sha256':r.sha(folder/'model.pt'),
        'evaluation_sha256':r.sha(folder/'evaluation.pt'),'rows_sha256':r.sha(folder/'rows.csv'),
        'data_sha256':{split:r.sha(root/run['mode']/split/'trajectories.npz') for split in outputs},
        'relaxed_raw_fraction':float(1-model.mix_logit.sigmoid()) if run['kind']=='relaxed' else None})
    print('repeat state evaluated',run['data_seed'],run['mode'],run['kind'],run['seed'],summaries['test']['all']['velocity_mae_pixels_per_frame']['mean'],flush=True)


def main():
    c=freeze();torch.set_num_threads(c['threads']);results=[];started=time.time()
    for data_seed in c['data_seeds']:
        for mode in c['modes']:
            for seed in c['seeds']:
                pair=[]
                for kind in c['kinds']:
                    folder=train(c,data_seed,mode,kind,seed);evaluate(folder,c)
                    run=json.loads((folder/'run.json').read_text());pair.append(run)
                    results.append({'data_seed':data_seed,'mode':mode,'kind':kind,'seed':seed,'run':run,'metrics':json.loads((folder/'metrics.json').read_text())})
                    dump(HERE/f'reports/{NAME}/state_progress.json',{'completed_models':len(results),'expected_models':126,
                        'last':[data_seed,mode,kind,seed],'seconds':time.time()-started})
                for key in ['sample_sha256','permutation_sha256']:assert len({v[key] for v in pair})==1
                assert len({v['initial_weights_sha256'] for v in pair if v['kind'] in ['interaction','augmented','wrong_exact','joint_exact']})==1
    dump(HERE/f'reports/{NAME}/state_summary.json',{'results':results,'models':len(results),'new_trainings':len(results),
        'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'seconds':time.time()-started,'scope':c['scope']})


if __name__=='__main__':main()
