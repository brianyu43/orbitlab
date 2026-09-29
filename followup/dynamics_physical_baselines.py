"""Non-learned motion references using known net force, drag and reflecting walls.

The force-wall baseline deliberately omits all disk-disk interaction. It knows
the simulator's frame discretization; it is a privileged mechanism reference.
"""
import json
import time
import numpy as np
from common import HERE,dump,fresh_dir,r


def predict(state,context,action,kind='force_wall'):
    state=np.asarray(state,np.float64);context=np.asarray(context,np.float64);action=np.asarray(action,np.float64)
    alive=state[...,4]>0;p=state[...,:2].copy();v=state[...,2:4].copy()+action
    if kind=='inertial':return np.concatenate([p+v,v],-1)*alive[...,None]
    if kind!='force_wall':raise ValueError(kind)
    h=1/8;net=context[...,:2]+context[...,2:4];drag=context[...,4]
    decay=np.exp(-drag*h);force=np.full_like(drag,h)
    np.divide(-np.expm1(-drag*h),drag,out=force,where=drag!=0)
    limit=31.5-state[...,4]
    for _ in range(8):
        v=v*decay[...,None,None]+net[...,None,:]*force[...,None,None]
        p=p+v*h
        outside=(np.abs(p)>limit[...,None])&alive[...,None]
        sign=np.sign(p)
        p=np.where(outside,sign*(2*limit[...,None]-np.abs(p)),p)
        v=np.where(outside&(v*sign>0),-v,v)
    return np.concatenate([p,v],-1)*alive[...,None]


def freeze():
    cp=HERE/'configs/dynamics_physical_baselines_v1.json'
    if not cp.exists():dump(cp,{'modes':['isotropic','fixed_gravity','variable_force'],'split':'val','baselines':['inertial','force_wall'],
        'information':'True current states, action, and known net acceleration/drag. force_wall also knows the simulator discretization and boundary rule; ignores all disk-disk interactions.',
        'substeps':8,'source_sha256':r.sha(__file__),'data_manifest_sha256':r.sha(HERE/'data/dynamics_world_v1/manifest.json'),
        'claim_boundary':'Analytic reference, not a learned model or inferred law. One-step evaluation supplies the true state each frame.',
        'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def main():
    c=freeze();root=fresh_dir(HERE/'reports/dynamics_physical_baselines_v1');summary=[]
    for mode in c['modes']:
        path=HERE/f'data/dynamics_world_v1/{mode}/val/trajectories.npz'
        with np.load(path) as a:s=a['states'].copy();context=a['contexts'].copy();action=a['actions'].copy();events=a['events'].copy()
        contexts=np.broadcast_to(context[:,None,:],(*s.shape[:1],s.shape[1]-1,5));live=s[:,:-1,:,4]>0
        pred={kind:predict(s[:,:-1],contexts,action,kind) for kind in c['baselines']};rows=[]
        for kind,p in pred.items():
            absolute=np.abs(p-s[:,1:,:,:4]);position=(absolute[...,:2].sum(-1)*live).sum(-1)/(2*live.sum(-1))
            velocity=(absolute[...,2:].sum(-1)*live).sum(-1)/(2*live.sum(-1));pair=events[...,0]>0;wall=events[...,1]>0
            for episode in range(len(s)):
                for name,selection in [('all',np.ones(s.shape[1]-1,bool)),('no_contact',~(pair[episode]|wall[episode])),
                    ('contact',pair[episode]|wall[episode]),('pair',pair[episode]),('wall_only',wall[episode]&~pair[episode])]:
                    if selection.any():rows.append({'baseline':kind,'episode':episode,'category':name,'transitions':int(selection.sum()),
                        'position_mae_pixels':float(position[episode,selection].mean()),'velocity_mae_pixels_per_frame':float(velocity[episode,selection].mean())})
            for category in sorted({v['category'] for v in rows if v['baseline']==kind}):
                selected=[v for v in rows if v['baseline']==kind and v['category']==category]
                item={'mode':mode,'baseline':kind,'category':category,'episodes':len(selected),
                    'metrics':{key:r.bootstrap([v[key] for v in selected]) for key in ['position_mae_pixels','velocity_mae_pixels_per_frame']}}
                summary.append(item);print(mode,kind,category,{k:round(v['mean'],6) for k,v in item['metrics'].items()},flush=True)
        np.savez_compressed(root/f'{mode}_predictions.npz',**pred);r.write_csv(root/f'{mode}_rows.csv',rows)
    dump(root/'summary.json',{'results':summary,'config_sha256':r.sha(HERE/'configs/dynamics_physical_baselines_v1.json'),
        'files':{mode:{'data_sha256':r.sha(HERE/f'data/dynamics_world_v1/{mode}/val/trajectories.npz'),
                      'predictions_sha256':r.sha(root/f'{mode}_predictions.npz'),'rows_sha256':r.sha(root/f'{mode}_rows.csv')} for mode in c['modes']},
        'claim_boundary':c['claim_boundary']})


if __name__=='__main__':main()
