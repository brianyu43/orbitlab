"""Separate four pre-action RGB frames/commands from privileged scoring labels."""
import json
import time
import numpy as np
from common import HERE,dump,fresh_dir,r,update_status

NAME='dynamics_observation_v1'


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        dump(path,{'observation_times':[0,1,2,3],'predicted_time':4,'action_transition':3,
            'input_files':{'observations.npz':['images'],'commands.npz':['target_color','delta_velocity']},
            'command':'At t=3, change the velocity of the object with the named color by delta_velocity. Colors are distinct within each scene. Color is observable; persistent IDs/count/states/context are not provided.',
            'labels':'Past state t=0..3 and next state t=4, true net acceleration/drag, past contact events, next contact events, target ID: supervision/scoring and explicitly privileged controls only.',
            'comparison':'Same four frames across methods. No mode name is a model input. All train groups may be pooled; held-out groups remain separate.',
            'readout_scope':'This intake contains one initial observation window per original trajectory. No extra future training windows are implied.',
            'source_manifest_sha256':r.sha(HERE/'data/dynamics_world_v1/manifest.json'),
            'simulator_sha256':r.sha(HERE/'dynamics_world.py'),'source_sha256':r.sha(__file__),'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    assert c['source_manifest_sha256']==r.sha(HERE/'data/dynamics_world_v1/manifest.json')
    assert c['simulator_sha256']==r.sha(HERE/'dynamics_world.py') and c['source_sha256']==r.sha(__file__)
    return c


def main():
    c=freeze();root=fresh_dir(HERE/f'data/{NAME}');source=json.loads((HERE/'data/dynamics_world_v1/manifest.json').read_text());groups=[]
    update_status('C04','in_progress',['configs/dynamics_observation_v1.json','planning_C04_EXECUTION_KO.md'],
        note='Building explicit four-frame pre-action RGB/command inputs. Perception/environment inference and component controls remain in progress.')
    for item in source['groups']:
        path=HERE/item['path'];assert r.sha(path)==item['archive_sha256'];folder=fresh_dir(root/item['mode']/item['split'])
        with np.load(path) as a:
            images=a['observations'].copy();states=a['states'];actions=a['actions'];target=a['target_ids'].copy();n=len(images)
            assert images.shape[1:]==(4,64,64,3) and not actions[:,:3].any() and not actions[:,4:].any()
            colors=states[np.arange(n),3,target,5].astype(np.int64);delta=actions[np.arange(n),3,target].copy()
            assert np.all(np.count_nonzero(np.linalg.norm(actions[:,3],axis=-1),axis=1)==1)
            for i in range(n):
                alive=states[i,3,:,4]>0;assert len(set(states[i,3,alive,5]))==int(alive.sum())
            np.savez_compressed(folder/'observations.npz',images=images)
            np.savez_compressed(folder/'commands.npz',target_color=colors,delta_velocity=delta)
            np.savez_compressed(folder/'labels.npz',past_states=states[:,:4].copy(),next_states=states[:,4].copy(),
                net_acceleration=a['contexts'][:,:2]+a['contexts'][:,2:4],drag=a['contexts'][:,4].copy(),
                past_events=a['events'][:,:3].copy(),next_events=a['events'][:,3].copy(),target_ids=target)
            if 'counterfactual_states' in a:
                np.testing.assert_array_equal(a['counterfactual_states'][:,:4],states[:,:4])
        groups.append({'mode':item['mode'],'split':item['split'],'episodes':item['episodes'],
            'source_sha256':item['archive_sha256'],'files':{p.name:r.sha(p) for p in sorted(folder.iterdir())}})
        print('observation inputs',item['mode'],item['split'],item['episodes'],flush=True)
    dump(root/'manifest.json',{'groups':groups,'episodes':sum(v['episodes'] for v in groups),
        'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'scope':c['readout_scope']})


if __name__=='__main__':main()
