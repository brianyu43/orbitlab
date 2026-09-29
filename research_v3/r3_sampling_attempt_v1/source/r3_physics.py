"""Batched float64 disk simulator, extended to six disks without changing physics.

Pairs retain persistent index order. This is reference data generation, not a
learned model. A separate coarse Torch solver is a declared physical prior.
"""
import numpy as np

def violation(s):
    err = np.maximum(0, (np.abs(s[..., :2])-(31.5-s[...,4,None])).max((1,2)))
    for i in range(s.shape[1]):
        for j in range(i+1,s.shape[1]):
            err=np.maximum(err,s[:,i,4]+s[:,j,4]-np.linalg.norm(s[:,j,:2]-s[:,i,:2],axis=-1))
    return err

def step(state,context,action,substeps=8,sweeps=32):
    s=np.array(state,dtype=np.float64,copy=True);c=np.asarray(context,dtype=np.float64)
    assert s.ndim==3 and s.shape[-1]==7 and 1<=s.shape[1]<=6
    assert c.shape==(len(s),5) and action.shape==(*s.shape[:2],2)
    assert np.isfinite(s).all() and np.isfinite(c).all() and np.isfinite(action).all()
    assert (s[...,4]>0).all() and (s[...,6]==1).all() and (c[:,4]>=0).all()
    s[...,2:4]+=action;h=1/substeps;net=c[:,:2]+c[:,2:4];drag=c[:,4]
    displacement=np.linalg.norm(s[...,2:4],axis=-1)*h+np.linalg.norm(net,axis=-1)[:,None]*h*h
    if np.any(displacement>np.min(s[...,4],axis=1)[:,None]*.5): raise ValueError('Anti-tunneling regime exceeded')
    decay=np.exp(-drag*h);force=np.full_like(drag,h)
    np.divide(-np.expm1(-drag*h),drag,out=force,where=drag!=0)
    limits=31.5-s[...,4];events=np.zeros((len(s),2),int)
    pairs=[(i,j) for i in range(s.shape[1]) for j in range(i+1,s.shape[1])]
    for _ in range(substeps):
        s[...,2:4]=s[...,2:4]*decay[:,None,None]+net[:,None,:]*force[:,None,None]
        s[...,:2]+=s[...,2:4]*h
        outside=np.abs(s[...,:2])>limits[...,None];sign=np.sign(s[...,:2])
        bounce=outside&(s[...,2:4]*sign>0)
        s[...,:2]=np.where(outside,sign*(2*limits[...,None]-np.abs(s[...,:2])),s[...,:2])
        s[...,2:4]=np.where(bounce,-s[...,2:4],s[...,2:4]);events[:,1]+=bounce.sum((1,2))
        active=np.ones(len(s),bool)
        for _ in range(sweeps):
            for i,j in pairs:
                delta=s[:,j,:2]-s[:,i,:2];distance=np.linalg.norm(delta,axis=-1);contact=s[:,i,4]+s[:,j,4]
                hit=active&(distance<=contact+1e-10)
                if np.any(hit&(distance<1e-12)):raise FloatingPointError('Coincident centers')
                normal=delta/np.maximum(distance[:,None],1e-12)
                overlap=np.maximum(0,contact-distance)*hit
                s[:,i,:2]-=normal*overlap[:,None]/2;s[:,j,:2]+=normal*overlap[:,None]/2
                approach=((s[:,j,2:4]-s[:,i,2:4])*normal).sum(-1)
                impulse=np.minimum(approach,0)*hit
                s[:,i,2:4]+=impulse[:,None]*normal;s[:,j,2:4]-=impulse[:,None]*normal
                events[:,0]+=(hit&(approach<0))
            outside=(np.abs(s[...,:2])>limits[...,None])&active[:,None,None];sign=np.sign(s[...,:2])
            bounce=outside&(s[...,2:4]*sign>0)
            s[...,:2]=np.where(outside,sign*limits[...,None],s[...,:2]);s[...,2:4]=np.where(bounce,-s[...,2:4],s[...,2:4]);events[:,1]+=bounce.sum((1,2))
            active &= violation(s)>1e-7
            if not active.any():break
        if active.any():raise FloatingPointError('Projection did not converge')
    assert np.isfinite(s).all()
    return s,events

def rollout(initial,context,actions):
    states=[np.array(initial,dtype=np.float64)];events=[]
    for t in range(actions.shape[1]):
        s,e=step(states[-1],context,actions[:,t]);states.append(s);events.append(e)
    return np.stack(states,1),np.stack(events,1)
