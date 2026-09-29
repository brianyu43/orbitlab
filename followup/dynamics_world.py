"""Discrete equal-mass disk world, with joint C4 state/action/context transforms.

Positions are in pixel units about the image center; y points down. One frame
uses eight semi-implicit substeps. Contact impulses are perfectly elastic, but
finite-step contact positions are projected: this is a discrete simulator, not
an event-exact solution of continuous mechanics. Persistent IDs fix the order
of simultaneous contact resolution, independently of input slot order.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
import numpy as np

SIZE = 64
HALF = 31.5
# x, y, vx, vy, radius, color, mass; padding is added by dataset writers only.
PALETTE = np.array([[230,55,55],[55,205,85],[65,105,235],
                    [235,200,45],[190,75,210],[40,205,210]], dtype=np.float64)


@dataclass(frozen=True)
class Physics:
    substeps: int = 8
    contact_sweeps: int = 32
    overlap_tolerance: float = 1e-7


def rotate_vectors(a, k):
    a = np.array(a, dtype=np.float64, copy=True)
    for _ in range(k % 4):
        a = np.stack((a[...,1], -a[...,0]), axis=-1)
    return a


def rotate_state(s, k):
    s = np.array(s, dtype=np.float64, copy=True)
    s[:,:2] = rotate_vectors(s[:,:2], k)
    s[:,2:4] = rotate_vectors(s[:,2:4], k)
    return s


def rotate_context(c, k):
    c = np.array(c, dtype=np.float64, copy=True)
    c[:2] = rotate_vectors(c[:2], k)
    c[2:4] = rotate_vectors(c[2:4], k)
    return c


def net_acceleration(c):
    return np.asarray(c)[:2] + np.asarray(c)[2:4]


def violation(s):
    """Largest wall penetration or disk overlap, in pixels."""
    err = max(0., float((np.abs(s[:,:2]) - (HALF-s[:,4,None])).max()))
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            err = max(err, float(s[i,4]+s[j,4]-np.linalg.norm(s[j,:2]-s[i,:2])))
    return err


def step(state, context, action=None, ids=None, physics=Physics()):
    """One frame; action is an instantaneous per-object velocity increment.

    context=(gravity_x,gravity_y,wind_accel_x,wind_accel_y,isotropic_drag).
    Wind is explicitly a constant acceleration, not an aerodynamic wind field.
    Each pair impulse conserves momentum/kinetic energy without external force.
    """
    s = np.array(state, dtype=np.float64, copy=True)
    c = np.asarray(context, dtype=np.float64)
    ids = np.arange(len(s)) if ids is None else np.asarray(ids)
    if s.ndim!=2 or s.shape[1]!=7 or not 1<=len(s)<=4:
        raise ValueError('Expected 1..4 disks with seven state coordinates')
    if c.shape!=(5,) or c[4]<0 or not np.isfinite(c).all():
        raise ValueError('Invalid context')
    if len(set(ids.tolist()))!=len(s) or not np.isfinite(s).all():
        raise ValueError('Invalid state/IDs')
    if np.any(s[:,4]<=0) or np.any(s[:,6]!=1):
        raise ValueError('Positive radii and equal unit masses are required')
    a = np.zeros((len(s),2)) if action is None else np.asarray(action,dtype=np.float64)
    if a.shape!=(len(s),2) or not np.isfinite(a).all():
        raise ValueError('Action must be a finite velocity increment per disk')
    s[:,2:4] += a
    h=1/physics.substeps
    if np.max(np.linalg.norm(s[:,2:4],axis=1)*h + np.linalg.norm(net_acceleration(c))*h*h) > np.min(s[:,4])*.5:
        raise ValueError('Step displacement exceeds the declared anti-tunneling regime')
    order=np.argsort(ids);pairs=[(int(i),int(j)) for p,i in enumerate(order) for j in order[p+1:]]
    events={'pair_impulses':0,'wall_impulses':0,'projection_sweeps':0}
    decay=np.exp(-c[4]*h)
    force_scale=h if c[4]==0 else -np.expm1(-c[4]*h)/c[4]
    for _ in range(physics.substeps):
        s[:,2:4]=s[:,2:4]*decay+net_acceleration(c)*force_scale
        s[:,:2]+=s[:,2:4]*h
        # Reflect the within-substep overshoot at a wall.
        limits=HALF-s[:,4]
        for i in order:
            for axis in (0,1):
                if abs(s[i,axis])>limits[i]:
                    sign=np.sign(s[i,axis]);s[i,axis]=sign*(2*limits[i]-abs(s[i,axis]))
                    if s[i,axis+2]*sign>0:
                        s[i,axis+2]*=-1;events['wall_impulses']+=1
        for sweep in range(physics.contact_sweeps):
            for i,j in pairs:
                delta=s[j,:2]-s[i,:2];distance=float(np.linalg.norm(delta));contact=s[i,4]+s[j,4]
                if distance>contact+1e-10:continue
                if distance<1e-12:raise FloatingPointError('Undefined normal at coincident centers')
                normal=delta/distance
                overlap=max(0.,contact-distance)
                s[i,:2]-=normal*overlap/2;s[j,:2]+=normal*overlap/2
                approach=float(np.dot(s[j,2:4]-s[i,2:4],normal))
                if approach<0:
                    s[i,2:4]+=approach*normal;s[j,2:4]-=approach*normal
                    events['pair_impulses']+=1
            # A disk projection can push a neighboring disk into a wall.
            for i in order:
                for axis in (0,1):
                    if abs(s[i,axis])>limits[i]:
                        sign=np.sign(s[i,axis]);s[i,axis]=sign*limits[i]
                        if s[i,axis+2]*sign>0:
                            s[i,axis+2]*=-1;events['wall_impulses']+=1
            events['projection_sweeps']+=1
            if violation(s)<=physics.overlap_tolerance:break
        else:raise FloatingPointError('Contact projection did not converge')
    if not np.isfinite(s).all():raise FloatingPointError('Nonfinite state')
    return s,events


def rollout(initial, context, actions, ids=None, physics=Physics()):
    states=[np.asarray(initial,dtype=np.float64).copy()];events=[]
    for action in actions:
        nxt,event=step(states[-1],context,action,ids,physics)
        states.append(nxt);events.append([event['pair_impulses'],event['wall_impulses']])
    return np.stack(states),np.asarray(events,dtype=np.int32)


def render(state, ids=None):
    """Radial antialiasing and stable painter IDs commute with pixel C4."""
    s=np.asarray(state);ids=np.arange(len(s)) if ids is None else np.asarray(ids)
    yy,xx=np.mgrid[:SIZE,:SIZE].astype(np.float64);xx-=HALF;yy-=HALF
    alpha=np.clip(s[:,4,None,None]+.5-np.sqrt((xx-s[:,0,None,None])**2+(yy-s[:,1,None,None])**2),0,1)
    rgb=np.zeros((SIZE,SIZE,3),np.float64);visible=alpha.copy();trans=np.ones((SIZE,SIZE))
    order=np.argsort(ids)
    for i in order:
        a=alpha[i,:,:,None];rgb=rgb*(1-a)+PALETTE[int(s[i,5])]*a
    for i in order[::-1]:visible[i]*=trans;trans*=1-alpha[i]
    weights=np.concatenate([trans[None],visible],axis=0)
    labels=np.concatenate([[0],ids+1])[weights.argmax(0)]
    return {'image':np.rint(rgb).clip(0,255).astype(np.uint8),
            'segmentation':labels.astype(np.uint8),'visible':visible,'amodal':alpha}


def initial_orbit_hash(s, c, action):
    # Equivalent decompositions of a constant net force are the same physics.
    c=np.concatenate([net_acceleration(c),[0.,0.,float(c[4])]])
    hashes=[]
    for k in range(4):
        a=np.concatenate([rotate_state(s,k).ravel(),rotate_context(c,k),rotate_vectors(action,k).ravel()])
        a=np.round(a,9);a[a==0]=0
        hashes.append(hashlib.sha256(a.astype('<f8').tobytes()).hexdigest())
    return min(hashes)


def sample(index, seed, split, mode, count=2):
    splits=['train','val','test','attribute_ood','count3','count4','force_ood']
    modes=['isotropic','fixed_gravity','variable_force']
    rng=np.random.default_rng(np.random.SeedSequence([seed,splits.index(split),modes.index(mode),index]))
    state=[];used=set()
    for i in range(count):
        held=split=='attribute_ood' and i==0
        choices=[(ri,co) for ri in range(3) for co in range(6) if (ri==co)==held and co not in used]
        ri,color=choices[int(rng.integers(len(choices)))];radius=float(4+ri);used.add(color)
        for _ in range(10000):
            p=rng.uniform(-HALF+radius+2,HALF-radius-2,size=2)
            if all(np.linalg.norm(p-np.asarray(old[:2]))>radius+old[4]+2 for old in state):break
        else:raise RuntimeError('Placement failed')
        vel=rng.uniform(-1.5,1.5,size=2)
        state.append([*p,*vel,radius,color,1.])
    if mode=='isotropic':c=np.zeros(5)
    elif mode=='fixed_gravity':c=np.array([0.,.05,0.,0.,0.])
    else:
        c=np.array([0.,rng.uniform(.02,.06),rng.uniform(-.05,.05),rng.uniform(-.02,.02),rng.uniform(0,.01)])
        if split=='force_ood':
            # New upward resultant, absent from the variable-force training support.
            c[:4]=[0.,rng.uniform(-.08,-.04),rng.uniform(-.08,.08),rng.uniform(-.02,0.)]
    actions=np.zeros((64,count,2));target=int(rng.integers(count))
    actions[3,target]=rng.uniform(-1.2,1.2,size=2)
    return np.asarray(state),c,actions,target
