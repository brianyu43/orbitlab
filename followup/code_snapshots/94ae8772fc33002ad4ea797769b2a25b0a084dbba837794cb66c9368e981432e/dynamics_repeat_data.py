"""Fresh crossed-data repeats, preserving C01/C02 generator and contracts."""
import json
import time
import numpy as np
from common import HERE,dump,r
import dynamics_world as d
from build_dynamics_data import pad

NAME='dynamics_repeats_v1'
DATA_SEEDS=[997101,997102]


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        old=json.loads((HERE/'configs/dynamics_world_v1.json').read_text())
        prior=['configs/dynamics_world_v1.json','configs/dynamics_state_study_v1.json',
               'configs/dynamics_observation_models_v1.json','configs/dynamics_observation_eval_v1.json',
               'configs/dynamics_autonomous_v1.json','data/dynamics_world_v1/manifest.json']
        sources=['dynamics_repeat_data.py','dynamics_world.py','build_dynamics_data.py','dynamics_models.py',
                 'dynamics_observation_models.py','dynamics_rgb_baseline.py','dynamics_autonomous.py',
                 'dynamics_autonomous_metrics.py','planning_C07_EXECUTION_KO.md']
        dump(path,{'original_data_seed':old['seed'],'new_data_seeds':DATA_SEEDS,'generator':old,
            'initialization_seeds':[0,1,2],'state_model_seed_base':765000,'state_sampling_seed_base':765100,
            'state_augmentation_seed_base':765200,'observation_model_seed_base':996000,'observation_sampling_seed_base':996100,
            'prior_sha256':{p:r.sha(HERE/p) for p in prior},'source_sha256':{p:r.sha(HERE/p) for p in sources},
            'replication':'Two new generation seeds crossed with the SAME three initialization/sampling seeds. Same data sizes, modes, splits, physics, action time, model architectures and fixed training budgets. New models train from scratch; no old-data checkpoint transfer.',
            'state_scope':'All three modes and seven state predictors at 12000 steps, 64 transitions/batch, Adam .001, hidden 48, uniform sampling and slot permutations. 126 new trainings.',
            'observation_scope':'Both RGB CNN and measured-input MLP at 4000 steps, batch32 Adam .0005, pooled train across modes; 12 new trainings. Same known renderer/physics priors and fixed scales.',
            'evaluation_scope':'C03 teacher-forced one-step; C04 four-frame state/context substitutions; C05/C06 61-step recurrence and same-past opposite action. Same metrics and numerical-failure policy, every seed retained.',
            'primary_comparisons':['joint_exact versus wrong_exact and interaction one-step velocity error by mode/split/contact stratum',
                'Autonomous position/velocity errors, numerical failures and geometry violations by horizon/input/split',
                'Target and nontarget counterfactual response versus zero-response control'],
            'independence':'Data seed is the independent replication unit. Initialization seeds are crossed and paired across datasets/models, not additional independent datasets. No favorable-seed selection.',
            'not_yet_this_dataset':'Observation-length and missing-context experiments use separate later frozen protocols, retaining C07.4/C07.5 scope.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for field in ['prior_sha256','source_sha256']:
        for path,sha in c[field].items():assert r.sha(HERE/path)==sha,path
    assert c['new_data_seeds']==DATA_SEEDS
    return c


def original_orbits():
    manifest=json.loads((HERE/'data/dynamics_world_v1/manifest.json').read_text())
    physical=set();images=set()
    for g in manifest['groups']:
        path=(HERE/g['path']).parent/'records.json';assert r.sha(path)==g['records_sha256']
        for record in json.loads(path.read_text()):
            physical.add(record['initial_orbit_sha256']);images.add(record['initial_image_orbit_sha256'])
    assert len(physical)==len(images)==2240
    return physical,images


def register(records,physical,images):
    for record in records:
        ph,im=record['initial_orbit_sha256'],record['initial_image_orbit_sha256']
        assert ph not in physical and im not in images,record['group']
        physical.add(ph);images.add(im)


def build_group(c,seed,mode,split,number,physical,images):
    spec=c['generator'];root=HERE/f'data/{NAME}/d{seed}/world';folder=root/mode/split
    if (folder/'group.json').exists():
        meta=json.loads((folder/'group.json').read_text())
        assert r.sha(folder/'trajectories.npz')==meta['archive_sha256']
        assert r.sha(folder/'records.json')==meta['records_sha256']
        register(json.loads((folder/'records.json').read_text()),physical,images)
        return meta
    if folder.exists():raise RuntimeError(f'Incomplete group retained for diagnosis: {folder}')
    folder.mkdir(parents=True)
    count={'count3':3,'count4':4}.get(split,2)
    length=spec['observation_frames']+(spec['train_future_frames'] if split=='train' else spec['eval_future_frames'])
    arrays={k:[] for k in ['states','contexts','actions','events','observations','observation_segmentation']}
    records=[];counterfactuals=[];cf_events=[]
    for i in range(number):
        state,context,action,target=d.sample(i,seed,split,mode,count);action=action[:length-1]
        record={'id':i,'group':f'd{seed}/{mode}/{split}/{i}','n_objects':count,'target_id':target,
                'initial_orbit_sha256':d.initial_orbit_hash(state,context,action[3]),
                'initial_image_orbit_sha256':r.orbit_hash(d.render(state)['image']),
                'seed_components':[seed,split,mode,i]}
        register([record],physical,images)
        states,events=d.rollout(state,context,action,physics=d.Physics(**spec['physics']))
        assert max(d.violation(v) for v in states)<=spec['physics']['overlap_tolerance']*1.1
        observed=[d.render(v) for v in states[:4]]
        arrays['states'].append(pad(states,count));arrays['contexts'].append(context)
        arrays['actions'].append(pad(action,count));arrays['events'].append(events)
        arrays['observations'].append(np.stack([v['image'] for v in observed]))
        arrays['observation_segmentation'].append(np.stack([v['segmentation'] for v in observed]))
        if split in spec['counterfactual_splits']:
            cf,ev=d.rollout(state,context,-action,physics=d.Physics(**spec['physics']))
            np.testing.assert_array_equal(cf[:4],states[:4]);counterfactuals.append(pad(cf,count));cf_events.append(ev)
            record['counterfactual_same_group']=True
        records.append(record)
    data={k:np.stack(v) for k,v in arrays.items()}
    data['n_objects']=np.full(number,count,np.int8);data['target_ids']=np.array([v['target_id'] for v in records],np.int8)
    if counterfactuals:data.update(counterfactual_states=np.stack(counterfactuals),counterfactual_events=np.stack(cf_events))
    np.savez_compressed(folder/'trajectories.npz',**data);dump(folder/'records.json',records)
    group={'data_seed':seed,'mode':mode,'split':split,'episodes':number,'frames':length,'n_objects':count,
           'counterfactual_episodes':len(counterfactuals),'pair_impulses':int(data['events'][:,:,0].sum()),
           'wall_impulses':int(data['events'][:,:,1].sum()),
           'path':str((folder/'trajectories.npz').relative_to(HERE)),
           'archive_sha256':r.sha(folder/'trajectories.npz'),'records_sha256':r.sha(folder/'records.json')}
    dump(folder/'group.json',group);return group


def main():
    c=freeze();physical,images=original_orbits();all_groups=[];started=time.time()
    spec=c['generator']
    for seed in DATA_SEEDS:
        groups=[]
        for mode in spec['modes']:
            for split,number in spec['sizes'].items():
                if split=='force_ood' and mode not in spec['force_ood_modes']:continue
                group=build_group(c,seed,mode,split,number,physical,images);groups.append(group);all_groups.append(group)
                print('repeat data',seed,mode,split,number,flush=True)
                dump(HERE/f'reports/{NAME}/data_progress.json',{'groups_completed':len(all_groups),'expected_groups':38,'last':[seed,mode,split],'seconds':time.time()-started})
        dump(HERE/f'data/{NAME}/d{seed}/world/manifest.json',{'data_seed':seed,'groups':groups,'episodes':sum(g['episodes'] for g in groups),
            'counterfactual_episodes':sum(g['counterfactual_episodes'] for g in groups),'config_sha256':r.sha(HERE/f'configs/{NAME}.json')})
    dump(HERE/f'data/{NAME}/manifest.json',{'groups':all_groups,'new_episodes':sum(g['episodes'] for g in all_groups),
        'new_counterfactual_episodes':sum(g['counterfactual_episodes'] for g in all_groups),
        'unique_physical_orbits_including_original':len(physical),'unique_initial_image_orbits_including_original':len(images),
        'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'builder_sha256':r.sha(__file__),'seconds':time.time()-started})


if __name__=='__main__':main()
