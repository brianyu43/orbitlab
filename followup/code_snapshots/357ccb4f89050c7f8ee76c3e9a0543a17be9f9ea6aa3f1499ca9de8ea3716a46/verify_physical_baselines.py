import csv
import json
import numpy as np
from common import HERE,dump,r
import dynamics_world as d


def main():
    cp=HERE/'configs/dynamics_physical_baselines_v1.json';c=json.loads(cp.read_text())
    root=HERE/'reports/dynamics_physical_baselines_v1';summary=json.loads((root/'summary.json').read_text());checks=0;aggregates=0
    assert r.sha(cp)==summary['config_sha256'] and r.sha(HERE/'dynamics_physical_baselines.py')==c['source_sha256']
    assert r.sha(HERE/'data/dynamics_world_v1/manifest.json')==c['data_manifest_sha256']
    for mode in c['modes']:
        p=HERE/f'data/dynamics_world_v1/{mode}/val/trajectories.npz';entry=summary['files'][mode]
        assert r.sha(p)==entry['data_sha256'] and r.sha(root/f'{mode}_predictions.npz')==entry['predictions_sha256']
        assert r.sha(root/f'{mode}_rows.csv')==entry['rows_sha256']
        with np.load(p) as a:s=a['states'].copy();contexts=a['contexts'].copy();actions=a['actions'].copy();events=a['events'].copy()
        with np.load(root/f'{mode}_predictions.npz') as a:pred={k:a[k].copy() for k in a.files}
        n=2
        for episode in range(len(s)):
            for t in range(s.shape[1]-1):
                # Independent reference: full simulator with one disk at a time.
                for obj in range(n):
                    expected,_=d.step(s[episode,t,obj:obj+1],contexts[episode],actions[episode,t,obj:obj+1])
                    np.testing.assert_allclose(pred['force_wall'][episode,t,obj],expected[0,:4],atol=1e-12,rtol=0)
                state=s[episode,t,:n];vel=state[:,2:4]+actions[episode,t,:n]
                np.testing.assert_allclose(pred['inertial'][episode,t,:n],np.concatenate([state[:,:2]+vel,vel],-1),atol=0,rtol=0)
                checks+=1
        rows=list(csv.DictReader((root/f'{mode}_rows.csv').open()))
        for kind,predicted in pred.items():
            assert not predicted[:,:,n:].any()
            difference=np.abs(predicted[:,:,:n]-s[:,1:,:n,:4]);pos=difference[...,:2].mean((2,3));vel=difference[...,2:].mean((2,3))
            for row in [v for v in rows if v['baseline']==kind]:
                episode=int(row['episode']);pair=events[episode,:,0]>0;wall=events[episode,:,1]>0
                select={'all':np.ones(64,bool),'contact':pair|wall,'no_contact':~(pair|wall),'pair':pair,'wall_only':wall&~pair}[row['category']]
                assert abs(pos[episode,select].mean()-float(row['position_mae_pixels']))<1e-12
                assert abs(vel[episode,select].mean()-float(row['velocity_mae_pixels_per_frame']))<1e-12
        for result in [v for v in summary['results'] if v['mode']==mode]:
            selected=[v for v in rows if v['baseline']==result['baseline'] and v['category']==result['category']]
            for metric,stat in result['metrics'].items():assert abs(np.mean([float(v[metric]) for v in selected])-stat['mean'])<1e-12;aggregates+=1
    dump(root/'verification.json',{'all_passed':True,'two_disk_transition_references_replayed':checks,'individual_disk_references':2*checks,
        'aggregate_checks':aggregates,'verifier_sha256':r.sha(__file__),'claim_boundary':c['claim_boundary']})


if __name__=='__main__':main()
