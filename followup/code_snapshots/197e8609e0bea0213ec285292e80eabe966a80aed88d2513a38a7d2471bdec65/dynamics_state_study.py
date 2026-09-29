"""Locked three-initialization comparison of all seven state-oracle predictors."""
import hashlib
import json
import shutil
import time
import torch
import numpy as np
from common import HERE,dump,fresh_dir,r,state_digest,update_status
from dynamics_models import Transition,load_split,permute_slots,rotate_state,rotate_vec,rotate_context,rotate_motion


def freeze():
    cp=HERE/'configs/dynamics_state_study_v1.json'
    if not cp.exists():
        verified=json.loads((HERE/'reports/dynamics_curve_v1/verification.json').read_text());assert verified['all_passed']
        reused={f'{mode}/{kind}':r.sha(HERE/f'runs/dynamics_curve_v1/{mode}/{kind}_s0_step12000/model.pt')
            for mode in ['isotropic','fixed_gravity','variable_force'] for kind in ['interaction','wrong_exact','joint_exact']}
        dump(cp,{'modes':['isotropic','fixed_gravity','variable_force'],'kinds':['flat','object','interaction','augmented','wrong_exact','joint_exact','relaxed'],
            'seeds':[0,1,2],'steps':12000,'batch':64,'lr':.001,'hidden':48,'threads':2,
            'evaluation_splits':['test','attribute_ood','count3','count4','force_ood'],
            'force_ood_modes':['variable_force'],'reused_seed_zero_checkpoints':reused,
            'model_sha256':r.sha(HERE/'dynamics_models.py'),'runner_sha256':r.sha(__file__),
            'data_manifest_sha256':r.sha(HERE/'data/dynamics_world_v1/manifest.json'),
            'selection':'Common fixed 12k budget, not best per-model validation checkpoint. Uniform train transitions. Three initializations on one data-generation seed.',
            'primary':'Mean per-episode one-step velocity MAE in pixels/frame; position MAE and contact/noncontact/pair/wall-only strata are also reported.',
            'scope':'True state and known net acceleration/drag at every step. Held-out one-step generalization, not autonomous rollout or image-based prediction.',
            'budgets':'Same steps, batch and loss; parameter counts and wall time differ. Four interaction symmetry controls share raw weights and training samples.',
            'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def train(c,mode,kind,seed):
    folder=HERE/f'runs/dynamics_state_study_v1/{mode}/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert r.sha(folder/'model.pt')==run['checkpoint_sha256'];return folder
    fresh_dir(folder);parent_key=f'{mode}/{kind}'
    if seed==0 and parent_key in c['reused_seed_zero_checkpoints']:
        parent=HERE/f'runs/dynamics_curve_v1/{mode}/{kind}_s0_step12000'
        assert r.sha(parent/'model.pt')==c['reused_seed_zero_checkpoints'][parent_key]
        old=json.loads((parent/'run.json').read_text());shutil.copy2(parent/'model.pt',folder/'model.pt');shutil.copy2(parent/'optimizer.pt',folder/'optimizer.pt')
        dump(folder/'run.json',{**old,'seed':seed,'config_sha256':r.sha(HERE/'configs/dynamics_state_study_v1.json'),
            'reused_from':str(parent.relative_to(HERE)),'original_run_sha256':r.sha(parent/'run.json'),'new_training_steps':0})
        return folder
    data=load_split(HERE/'data/dynamics_world_v1',mode,'train');torch.manual_seed(765000+seed)
    model=Transition(kind,c['hidden']);initial=state_digest(model);opt=torch.optim.Adam(model.parameters(),lr=c['lr'])
    rng=torch.Generator().manual_seed(765100+seed);aug=torch.Generator().manual_seed(765200+seed)
    sh=hashlib.sha256();ph=hashlib.sha256();logs=[];start=time.perf_counter()
    dump(folder/'started.json',{'mode':mode,'kind':kind,'seed':seed,'config_sha256':r.sha(HERE/'configs/dynamics_state_study_v1.json')})
    for step in range(1,c['steps']+1):
        idx=torch.cat([torch.randint(len(data['state']),(32,),generator=rng),torch.randint(len(data['state']),(32,),generator=rng)])
        order=torch.rand(64,4,generator=rng).argsort(-1);sh.update(idx.numpy().tobytes());ph.update(order.numpy().tobytes())
        s,a,target=permute_slots(data['state'][idx],data['action'][idx],data['next'][idx],order);context=data['context'][idx]
        if kind=='augmented':
            k=int(torch.randint(4,(),generator=aug));s=rotate_state(s,k);a=rotate_vec(a,k);target=rotate_motion(target,k);context=rotate_context(context,k)
        prediction=model(s,a,context);mask=s[...,-1:];loss=((prediction-target).square()*mask).sum()/(4*mask.sum())
        if not torch.isfinite(loss):raise FloatingPointError('State study loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('State study gradient')
        opt.step()
        if step%3000==0:logs.append({'step':step,'loss':float(loss.detach())})
    seconds=time.perf_counter()-start
    torch.save({'kind':kind,'hidden':c['hidden'],'state_dict':model.state_dict()},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state(),'augmentation_rng':aug.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'mode':mode,'kind':kind,'seed':seed,'steps':c['steps'],'new_training_steps':c['steps'],
        'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':seconds,'initial_weights_sha256':initial,
        'sample_sha256':sh.hexdigest(),'permutation_sha256':ph.hexdigest(),'checkpoint_sha256':r.sha(folder/'model.pt'),
        'config_sha256':r.sha(HERE/'configs/dynamics_state_study_v1.json'),'training_archive_sha256':r.sha(HERE/f'data/dynamics_world_v1/{mode}/train/trajectories.npz'),
        'loss_log':logs})
    print('trained',mode,kind,seed,round(seconds,1),flush=True);return folder


@torch.no_grad()
def evaluate(folder,c):
    if (folder/'metrics.json').exists():return
    run=json.loads((folder/'run.json').read_text());ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();outputs={};rows=[];summaries={};equiv=[]
    for split in c['evaluation_splits']:
        if split=='force_ood' and run['mode'] not in c['force_ood_modes']:continue
        data=load_split(HERE/'data/dynamics_world_v1',run['mode'],split);pieces=[]
        for i in range(0,len(data['state']),128):pieces.append(model(data['state'][i:i+128],data['action'][i:i+128],data['context'][i:i+128]))
        pred=torch.cat(pieces);outputs[split]=pred;mask=data['state'][...,-1]
        error=(pred-data['next']).abs()*pred.new_tensor([31.5,31.5,3,3])
        pos=(error[...,:2].sum(-1)*mask).sum(-1)/(2*mask.sum(-1));vel=(error[...,2:].sum(-1)*mask).sum(-1)/(2*mask.sum(-1))
        pair=data['events'][:,0]>0;wall=data['events'][:,1]>0
        for episode in torch.unique(data['episode']).tolist():
            scene=data['episode']==episode
            for category,condition in [('all',torch.ones_like(scene)),('no_contact',~(pair|wall)),('contact',pair|wall),('pair',pair),('wall_only',wall&~pair)]:
                select=scene&condition
                if select.any():rows.append({'split':split,'episode':episode,'category':category,'transitions':int(select.sum()),
                    'position_mae_pixels':float(pos[select].mean()),'velocity_mae_pixels_per_frame':float(vel[select].mean())})
        summaries[split]={}
        for category in ['all','no_contact','contact','pair','wall_only']:
            selected=[v for v in rows if v['split']==split and v['category']==category]
            summaries[split][category]={'episodes':len(selected),**{key:r.bootstrap([v[key] for v in selected]) for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']}}
        if split=='test':
            indices=torch.arange(32)*64+3;s=data['state'][indices];a=data['action'][indices];context=data['context'][indices];base=model(s,a,context)
            for k in [1,2,3]:
                for label,ctx in [('joint',rotate_context(context,k)),('state_only',context)]:
                    rotated=model(rotate_state(s,k),rotate_vec(a,k),ctx);error=(rotated-rotate_motion(base,k)).abs()*s.new_tensor([31.5,31.5,3,3])
                    equiv.append({'rotation':k,'condition':label,'max_physical_coordinate_error':float(error.max())})
    torch.save(outputs,folder/'evaluation.pt');r.write_csv(folder/'rows.csv',rows)
    dump(folder/'metrics.json',{'splits':summaries,'equivariance':equiv,'checkpoint_sha256':r.sha(folder/'model.pt'),
        'evaluation_sha256':r.sha(folder/'evaluation.pt'),'data_sha256':{split:r.sha(HERE/f'data/dynamics_world_v1/{run["mode"]}/{split}/trajectories.npz') for split in outputs},
        'relaxed_raw_fraction':float(1-model.mix_logit.sigmoid()) if run['kind']=='relaxed' else None})
    print('evaluated',run['mode'],run['kind'],run['seed'],round(summaries['test']['all']['velocity_mae_pixels_per_frame']['mean'],5),flush=True)


def main():
    c=freeze();torch.set_num_threads(c['threads']);results=[]
    update_status('C03','in_progress',['configs/dynamics_state_study_v1.json','reports/dynamics_curve_v1/verification.json'],
        note='Locked three-initialization seven-model state study running. Reuses nine verified 12k seed-zero checkpoints; 54 new trainings. Held-out one-step evaluation, not long rollout.')
    for mode in c['modes']:
        for seed in c['seeds']:
            pair=[]
            for kind in c['kinds']:
                folder=train(c,mode,kind,seed);evaluate(folder,c);run=json.loads((folder/'run.json').read_text());pair.append(run)
                results.append({'mode':mode,'kind':kind,'seed':seed,'run':run,'metrics':json.loads((folder/'metrics.json').read_text())})
                dump(HERE/'reports/dynamics_state_study_v1/progress.json',{'completed_models':len(results),'expected_models':63,'results':results})
            for key in ['sample_sha256','permutation_sha256']:assert len({v[key] for v in pair})==1
            controls=[v for v in pair if v['kind'] in ['interaction','augmented','wrong_exact','joint_exact']]
            assert len({v['initial_weights_sha256'] for v in controls})==1
    dump(HERE/'reports/dynamics_state_study_v1/summary.json',{'results':results,'models':63,
        'new_trainings':sum(v['run']['new_training_steps']>0 for v in results),'reused_models':sum(v['run']['new_training_steps']==0 for v in results),
        'config_sha256':r.sha(HERE/'configs/dynamics_state_study_v1.json'),'claim_boundary':c['scope']})


if __name__=='__main__':main()
