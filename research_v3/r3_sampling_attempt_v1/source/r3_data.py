"""Fresh scene-family disk trajectories, with separate RGB observations and labels."""
import argparse,json,time
import numpy as np
import r3_core as c
import r3_physics as physics
import dynamics_world as w
from dynamics_rgb_baseline import measure_frame,infer
from v3_common import dump,sha,event

def category(events):
    paired=events[:,:,0].sum(1)>0;wall=events[:,:,1].sum(1)>0
    return np.where(paired,2,np.where(wall,1,0))

def prepare_group(seed,split,count):
    root=c.BASE/f'data/d{seed}/{split}/n{count}';root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():
        data=json.loads((root/'manifest.json').read_text())
        for name,digest in data['files'].items():assert sha(root/name)==digest
        return
    n=384 if split=='train' else 96;quota=n//3;frames=20 if split=='train' else 131
    buckets=[[],[],[]];candidate=0
    while min(map(len,buckets))<quota:
        samples=[]
        for _ in range(128):
            mode=['isotropic','fixed_gravity','variable_force'][candidate%3]
            s,ctx,a,target=w.sample(candidate,seed,split,mode,count)
            samples.append((s,ctx,a[:20],target,candidate,mode));candidate+=1
        ss=np.stack([v[0] for v in samples]);contexts=np.stack([v[1] for v in samples]);actions=np.stack([v[2] for v in samples])
        _,ev=physics.rollout(ss,contexts,actions)
        cats=category(ev[:,3:19])
        for sample,cat in zip(samples,cats):
            if len(buckets[cat])<quota:buckets[cat].append(sample)
        if candidate>20000:raise RuntimeError(f'Stratification exhausted without relaxing criteria: {seed} {split} {count}')
    selected=[v for bucket in buckets for v in bucket]
    # Interleave the three strata so mini-batches and example sets do not depend on bucket layout.
    selected=[buckets[k][i] for i in range(quota) for k in range(3)]
    initial=np.stack([v[0] for v in selected]);contexts=np.stack([v[1] for v in selected]);actions=np.zeros((n,frames,count,2))
    actions[:,:20]=np.stack([v[2] for v in selected]);states,events=physics.rollout(initial,contexts,actions)
    keys=[w.initial_orbit_hash(s,ctx,a[3]) for s,ctx,a in zip(initial,contexts,actions)]
    assert len(keys)==len(set(keys))
    stored={'states':states,'events':events,'contexts':contexts,'actions':actions,'target_ids':np.array([v[3] for v in selected]),'stratum':np.tile(np.arange(3),quota)}
    if split!='train':
        after=states[:,3].copy();base_actions=actions[:,3:].copy();reverse_actions=-base_actions
        rev,rev_events=physics.rollout(after,contexts,reverse_actions)
        force_context=contexts.copy();force_context[:,2]+=.04
        force,force_events=physics.rollout(after,force_context,base_actions)
        stored.update({'reversed_states':rev,'reversed_events':rev_events,'force_states':force,'force_events':force_events,'force_contexts':force_context})
    np.savez_compressed(root/'trajectories.npz',**stored)
    if split!='train':
        observations=np.stack([[w.render(frame)['image'] for frame in scene[:4]] for scene in states])
        np.savez_compressed(root/'observations.npz',images=observations)
        # Context estimates use only observations; no states or event flags are passed.
        inferred=[];info=[]
        for sequence in observations:
            features=np.stack([measure_frame(image) for image in sequence]);_,ctx,detail=infer(features)
            inferred.append([*ctx[:2],0,0,ctx[2]]);info.append(detail)
        np.savez_compressed(root/'rgb_context.npz',contexts=np.asarray(inferred))
        dump(root/'rgb_context_info.json',info)
    dump(root/'metadata.json',[{'candidate':v[4],'mode':v[5],'initial_orbit_sha256':key,'stratum':int(i%3)} for i,(v,key) in enumerate(zip(selected,keys))])
    max_violation=max(float(physics.violation(states[:,t]).max()) for t in range(states.shape[1]))
    assert max_violation<=1.01e-7
    files={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    dump(root/'manifest.json',{'seed':seed,'split':split,'count':count,'episodes':n,'frames':frames,'candidates':candidate,'files':files,'max_geometry_violation':max_violation,'protocol_sha256':sha(c.BASE/'protocol.json')})
    print('R3 data',seed,split,count,n,'candidates',candidate,flush=True);event('R3_data_group_complete',seed=seed,split=split,count=count)

def prepare(seed):
    c.freeze();prepare_group(seed,'train',2)
    split='val' if seed==c.DEVELOPMENT else 'test'
    for count in [2,3,4,6]:prepare_group(seed,split,count)
    paths=list((c.BASE/'data').glob('d*/*/n*/metadata.json'));seen=set()
    for p in paths:
        current={v['initial_orbit_sha256'] for v in json.loads(p.read_text())}
        assert not current&seen,p;seen.update(current)
    dump(c.BASE/'data_overlap_audit.json',{'groups':len(paths),'unique_initial_orbits':len(seen),'cross_new_group_overlap':0,'scope':'New R3 groups; historical cross-release numerical overlap audit remains required in final verification.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=c.DEVELOPMENT);a=p.parse_args();prepare(a.seed)
