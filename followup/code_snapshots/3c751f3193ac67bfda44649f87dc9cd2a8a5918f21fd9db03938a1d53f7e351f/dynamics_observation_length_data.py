"""Paired t=7-action worlds for observation lengths 1/2/4/8."""
import json
import time
import numpy as np
from common import HERE,dump,r
import dynamics_world as d
from build_dynamics_data import pad

NAME='dynamics_observation_length_v1'
DATA=[640031,997101,997102]


def source_root(ds):return HERE/('data/dynamics_world_v1' if ds==640031 else f'data/dynamics_repeats_v1/d{ds}/world')


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        for file in ['reports/dynamics_world_v1/verification.json','reports/dynamics_repeats_v1/data_verification.json']:
            assert json.loads((HERE/file).read_text())['all_passed']
        generator=json.loads((HERE/'configs/dynamics_world_v1.json').read_text());groups=[];source_files={}
        for ds in DATA:
            manifest=source_root(ds)/'manifest.json';source_files[str(manifest.relative_to(HERE))]=r.sha(manifest)
            for g in json.loads(manifest.read_text())['groups']:
                archive=source_root(ds)/g['mode']/g['split']/'trajectories.npz';records=archive.parent/'records.json'
                assert r.sha(archive)==g['archive_sha256'] and r.sha(records)==g['records_sha256']
                source_files[str(archive.relative_to(HERE))]=r.sha(archive);source_files[str(records.relative_to(HERE))]=r.sha(records)
                groups.append({'data_seed':ds,'mode':g['mode'],'split':g['split'],'episodes':g['episodes'],'n_objects':g['n_objects'],
                    'source_path':str(archive.relative_to(HERE)),'source_records':str(records.relative_to(HERE))})
        dump(path,{'data_seeds':DATA,'lengths':[1,2,4,8],'anchor':7,'observed_carrier_frames':8,'train_future_frames':16,'eval_future_frames':61,
            'generator':generator,'groups':groups,'source_files':source_files,
            'source_sha256':{f:r.sha(HERE/f) for f in ['dynamics_observation_length_data.py','dynamics_observation_length_reference.py','dynamics_world.py','build_dynamics_data.py','dynamics_rgb_baseline.py','planning_C07_OBSERVATION_LENGTH_KO.md']},
            'reference_protocol':{'length1':'Last-frame position/radius/presence; zero velocity and context.',
                'length2':'Last observed displacement as velocity; zero context.',
                'length4_8':'Shared decay and force fit over complete visible tracks, same centered least-squares equations as C04, clipped decay exp(-.05)..1, unbounded force.',
                'zero_design':'Fallback decay=1, not identified drag.',
                'contact':'No GT event filtering; past contact only for scoring.'},
            'frozen_unix':time.time(),'scope':'Reuse original independent initial draws and partitions, move impulse to transition 7 and rerun dynamics. Paired observation-length study, not three extra independent datasets. Every length shares t=7 state/action/future.'})
    c=json.loads(path.read_text())
    for family in ['source_files','source_sha256']:
        for p,sha in c[family].items():assert r.sha(HERE/p)==sha,p
    assert len(c['groups'])==57
    return c


def build(c,g):
    ds,mode,split=[g[k] for k in ['data_seed','mode','split']];root=HERE/f'data/{NAME}/d{ds}/world/{mode}/{split}'
    if (root/'group.json').exists():
        result=json.loads((root/'group.json').read_text())
        for p,sha in result['files'].items():assert r.sha(root/p)==sha
        return result
    if root.exists():raise RuntimeError(f'Incomplete group retained for diagnosis: {root}')
    root.mkdir(parents=True);physics=d.Physics(**c['generator']['physics']);frames=8+(16 if split=='train' else 61)
    with np.load(HERE/g['source_path']) as a:source={k:a[k].copy() for k in ['states','contexts','actions','n_objects','target_ids']}
    old_records=json.loads((HERE/g['source_records']).read_text());arrays={k:[] for k in ['states','contexts','actions','events','observations','observation_segmentation']}
    records=[];cf_states=[];cf_events=[]
    for i in range(g['episodes']):
        n=int(source['n_objects'][i]);assert n==g['n_objects'];initial=source['states'][i,0,:n];context=source['contexts'][i]
        actions=np.zeros((frames-1,n,2));actions[7]=source['actions'][i,3,:n]
        states,events=d.rollout(initial,context,actions,physics=physics)
        assert max(d.violation(s) for s in states)<=physics.overlap_tolerance*1.1
        observed=[d.render(s) for s in states[:8]]
        for key,value in [('states',pad(states,n)),('contexts',context),('actions',pad(actions,n)),('events',events),
                          ('observations',np.stack([x['image'] for x in observed])),('observation_segmentation',np.stack([x['segmentation'] for x in observed]))]:arrays[key].append(value)
        previous=old_records[i];record={'id':i,'data_seed':ds,'mode':mode,'split':split,'n_objects':n,'target_id':int(source['target_ids'][i]),
            'source_initial_orbit_sha256':previous['initial_orbit_sha256'],'source_initial_image_orbit_sha256':previous['initial_image_orbit_sha256'],
            't7_initial_orbit_sha256':d.initial_orbit_hash(initial,context,actions[7]),
            'anchor_image_orbit_sha256':r.orbit_hash(observed[7]['image'])}
        if split in c['generator']['counterfactual_splits']:
            cf,ev=d.rollout(initial,context,-actions,physics=physics);np.testing.assert_array_equal(cf[:8],states[:8])
            cf_states.append(pad(cf,n));cf_events.append(ev)
        records.append(record)
    data={k:np.stack(v) for k,v in arrays.items()};data['n_objects']=source['n_objects'];data['target_ids']=source['target_ids']
    if cf_states:data.update(counterfactual_states=np.stack(cf_states),counterfactual_events=np.stack(cf_events))
    np.savez_compressed(root/'trajectories.npz',**data);dump(root/'records.json',records)
    result={**g,'path':str((root/'trajectories.npz').relative_to(HERE)),'frames':frames,'counterfactual_episodes':len(cf_states),
        'pair_impulses':int(data['events'][:,:,0].sum()),'wall_impulses':int(data['events'][:,:,1].sum()),
        'files':{p:r.sha(root/p) for p in ['trajectories.npz','records.json']},'config_sha256':r.sha(HERE/f'configs/{NAME}.json')}
    dump(root/'group.json',result);return result


def main():
    c=freeze();groups=[];start=time.time()
    for g in c['groups']:
        groups.append(build(c,g));print('t7 world built',g['data_seed'],g['mode'],g['split'],len(groups),flush=True)
        dump(HERE/f'reports/{NAME}/data_progress.json',{'groups_completed':len(groups),'expected_groups':57,'seconds':time.time()-start})
    dump(HERE/f'data/{NAME}/manifest.json',{'groups':groups,'episodes':sum(g['episodes'] for g in groups),
        'counterfactual_episodes':sum(g['counterfactual_episodes'] for g in groups),'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'builder_sha256':r.sha(__file__),'scope':c['scope'],'seconds':time.time()-start})


if __name__=='__main__':main()
