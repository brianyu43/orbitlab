"""Check the expanded simulator against legacy physics and conservation laws."""
import numpy as np
import torch
import r3_physics as b
import r3_core as c
import dynamics_world as w
from v3_common import dump,sha

def main():
    checks=[]
    for n in [2,3,4,6]:
        samples=[w.sample(i,888101,'val','variable_force',n) for i in range(8)]
        initial=np.stack([s[0] for s in samples]);ctx=np.stack([s[1] for s in samples]);act=np.zeros((len(initial),131,n,2))
        act[:,:64]=np.stack([s[2] for s in samples]);states,events=b.rollout(initial,ctx,act)
        err=0.
        if n<=4:
            for i in range(len(initial)):
                ref,ev=w.rollout(initial[i],ctx[i],act[i]);err=max(err,float(np.abs(ref-states[i]).max()));assert np.array_equal(ev,events[i])
            assert err<1e-10
        assert max(b.violation(states[:,t]).max() for t in range(132))<=1.01e-7
        rot=np.stack([w.rotate_state(s,1) for s in initial]);rc=np.stack([w.rotate_context(s,1) for s in ctx]);ra=w.rotate_vectors(act,1)
        rotated,_=b.rollout(rot,rc,ra);expected=np.stack([np.stack([w.rotate_state(s,1) for s in seq]) for seq in states])
        assert np.max(abs(rotated-expected))<1e-8
        checks.append({'count':n,'episodes':len(initial),'steps':131,'reference_max_abs_difference':err if n<=4 else None,'rotation_max_difference':float(np.max(abs(rotated-expected)))})
    s=np.array([[[-5.,0,1.,0,4,0,1],[5.,0,-1.,0,4,1,1]]]);cs=np.zeros((1,5));a=np.zeros((1,5,2,2));traj,ev=b.rollout(s,cs,a)
    assert ev[...,0].sum()>0;assert np.allclose(traj[...,2:4].sum(2),0);assert np.allclose(np.square(traj[...,2:4]).sum((2,3)),2)
    x=c.encode_states(s);context=c.encode_context(cs);action=torch.zeros(1,2,2);m=c.model_for('contact_residual');y=m(x,action,context);y.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    counts={arm:sum(p.numel() for p in c.model_for(arm).parameters()) for arm in c.ARMS}
    assert len({counts[k] for k in ['one','multi','multi_bounded','mean','local_mean']})==1
    dump(c.BASE/'preflight.json',{'reference_agreement':checks,'elastic_energy_momentum':True,'coarse_autograd_finite':True,'parameter_counts':counts,
        'physics_sha256':sha(c.HERE/'r3_physics.py'),'core_sha256':sha(c.HERE/'r3_core.py'),'script_sha256':sha(__file__)})
    print('R3 131-step preflight passed',checks,flush=True)

if __name__=='__main__':main()
