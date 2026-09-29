"""Evaluate motion-only references on exactly the state study's held-out splits."""
import json
import numpy as np
from common import HERE,dump,fresh_dir,r
from dynamics_physical_baselines import predict


def main():
    config=HERE/'configs/dynamics_state_study_v1.json';c=json.loads(config.read_text());root=fresh_dir(HERE/'reports/dynamics_state_baselines_v1');results=[];files=[]
    for mode in c['modes']:
        for split in c['evaluation_splits']:
            if split=='force_ood' and mode not in c['force_ood_modes']:continue
            path=HERE/f'data/dynamics_world_v1/{mode}/{split}/trajectories.npz'
            with np.load(path) as a:s=a['states'].copy();context=a['contexts'].copy();actions=a['actions'].copy();events=a['events'].copy()
            contexts=np.broadcast_to(context[:,None,:],(len(s),s.shape[1]-1,5));live=s[:,:-1,:,4]>0
            out={kind:predict(s[:,:-1],contexts,actions,kind) for kind in ['inertial','force_wall']};rows=[]
            for kind,pred in out.items():
                difference=np.abs(pred-s[:,1:,:,:4]);position=(difference[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1))
                velocity=(difference[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1));pair=events[...,0]>0;wall=events[...,1]>0
                for episode in range(len(s)):
                    for name,select in [('all',np.ones(64,bool)),('no_contact',~(pair[episode]|wall[episode])),('contact',pair[episode]|wall[episode]),
                        ('pair',pair[episode]),('wall_only',wall[episode]&~pair[episode])]:
                        if select.any():rows.append({'kind':kind,'episode':episode,'category':name,'transitions':int(select.sum()),
                            'position_mae_pixels':float(position[episode,select].mean()),'velocity_mae_pixels_per_frame':float(velocity[episode,select].mean())})
                for category in ['all','no_contact','contact','pair','wall_only']:
                    selected=[v for v in rows if v['kind']==kind and v['category']==category]
                    results.append({'mode':mode,'split':split,'kind':kind,'category':category,'episodes':len(selected),
                        'metrics':{key:r.bootstrap([v[key] for v in selected]) for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']}})
            folder=fresh_dir(root/mode/split);np.savez_compressed(folder/'predictions.npz',**out);r.write_csv(folder/'rows.csv',rows)
            files.append({'mode':mode,'split':split,'episodes':len(s),'data_sha256':r.sha(path),
                'predictions_sha256':r.sha(folder/'predictions.npz'),'rows_sha256':r.sha(folder/'rows.csv')})
            print('baseline',mode,split,flush=True)
    dump(root/'summary.json',{'results':results,'files':files,'state_study_config_sha256':r.sha(config),
        'predictor_sha256':r.sha(HERE/'dynamics_physical_baselines.py'),'runner_sha256':r.sha(__file__),
        'claim_boundary':'Analytic one-step references. force_wall knows the simulator integration and boundaries and omits pair contacts; it is not a learned model.'})


if __name__=='__main__':main()
