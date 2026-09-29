import csv
import json
import numpy as np
from common import HERE,dump,r
import dynamics_world as d


def main():
    root=HERE/'reports/dynamics_state_baselines_v1';s=json.loads((root/'summary.json').read_text());replays=rows=aggregates=0
    assert s['state_study_config_sha256']==r.sha(HERE/'configs/dynamics_state_study_v1.json')
    assert s['predictor_sha256']==r.sha(HERE/'dynamics_physical_baselines.py') and s['runner_sha256']==r.sha(HERE/'dynamics_state_baselines.py')
    for entry in s['files']:
        mode,split=entry['mode'],entry['split'];folder=root/mode/split;path=HERE/f'data/dynamics_world_v1/{mode}/{split}/trajectories.npz'
        assert r.sha(path)==entry['data_sha256'] and r.sha(folder/'predictions.npz')==entry['predictions_sha256'] and r.sha(folder/'rows.csv')==entry['rows_sha256']
        with np.load(path) as a:state=a['states'].copy();contexts=a['contexts'].copy();actions=a['actions'].copy();events=a['events'].copy();counts=a['n_objects'].copy()
        with np.load(folder/'predictions.npz') as a:pred={k:a[k].copy() for k in a.files}
        for i in range(len(state)):
            n=int(counts[i]);assert not np.any(pred['force_wall'][i,:,n:])
            for t in range(64):
                for j in range(n):
                    expected,_=d.step(state[i,t,j:j+1],contexts[i],actions[i,t,j:j+1])
                    np.testing.assert_allclose(pred['force_wall'][i,t,j],expected[0,:4],atol=1e-12,rtol=0);replays+=1
            velocity=state[i,:-1,:n,2:4]+actions[i,:,:n]
            np.testing.assert_allclose(pred['inertial'][i,:,:n],np.concatenate([state[i,:-1,:n,:2]+velocity,velocity],-1),atol=0,rtol=0)
        records=list(csv.DictReader((folder/'rows.csv').open()));live=state[:,:-1,:,4]>0
        for kind,p in pred.items():
            error=np.abs(p-state[:,1:,:,:4]);position=(error[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1));velocity=(error[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1))
            for row in [v for v in records if v['kind']==kind]:
                i=int(row['episode']);pair=events[i,:,0]>0;wall=events[i,:,1]>0
                select={'all':np.ones(64,bool),'no_contact':~(pair|wall),'contact':pair|wall,'pair':pair,'wall_only':wall&~pair}[row['category']]
                assert abs(position[i,select].mean()-float(row['position_mae_pixels']))<1e-12
                assert abs(velocity[i,select].mean()-float(row['velocity_mae_pixels_per_frame']))<1e-12;rows+=1
        for entry in [v for v in s['results'] if v['mode']==mode and v['split']==split]:
            selected=[v for v in records if v['kind']==entry['kind'] and v['category']==entry['category']]
            for key,stat in entry['metrics'].items():
                expected=r.bootstrap([float(v[key]) for v in selected]);assert abs(expected['mean']-stat['mean'])<1e-12
                np.testing.assert_allclose(expected['base_scene_ci95'],stat['base_scene_ci95'],atol=1e-12,rtol=0);aggregates+=1
        print('verified baseline',mode,split,flush=True)
    dump(root/'verification.json',{'all_passed':True,'individual_disk_references_replayed':replays,'metric_rows_verified':rows,
        'aggregate_and_interval_checks':aggregates,'verifier_sha256':r.sha(__file__),'claim_boundary':s['claim_boundary']})


if __name__=='__main__':main()
