"""Controlled causal occlusion histories without color-based object separation.

Window order is current, t-1, t-2, t-3. A one-frame model gets window[0];
a recurrent model processes the reversed window. All predict current state.
"""
import hashlib,itertools,json
import numpy as np
import r1_core as r1
from v3_common import HERE

BASE=HERE/'r2';DEVELOPMENT=887101;CONFIRMATION=[887201,887202,887203]
CONDITIONS=list(itertools.product([0,1],[0,1],[0,1],[0.,.25,.5,.75]))

def background(seed,textured):
    if not textured:return np.zeros((64,64,3),np.float32)
    rng=np.random.default_rng(seed);yy,xx=np.mgrid[:64,:64]
    low=rng.uniform(8,28,3);phase=rng.uniform(0,6.28,3)
    value=low+7*np.sin(xx[...,None]*.23+phase)+6*np.cos(yy[...,None]*.19-phase)
    return np.clip(value,0,48).astype(np.float32)

def layers(objects,age=0):
    result=[]
    for v in objects:
        x=v['x']-age*v['vx'];y=v['y']-age*v['vy']
        patch=r1.alpha(v['shape'],v['radius'],v['pose'],'symmetric4_lanczos')
        a=np.zeros((64,64),np.float32);x0=x-16;y0=y-16
        l,t,r,b=max(0,x0),max(0,y0),min(64,x0+33),min(64,y0+33)
        if r>l and b>t:a[t:b,l:r]=patch[t-y0:b-y0,l-x0:r-x0]
        result.append(a)
    return np.stack(result)

def blocker(mask,fraction):
    """A vertical opaque panel with fractional boundary pixel; exact alpha-mass target."""
    result=np.zeros((64,64),np.float32)
    if fraction<=0:return result
    yy,xx=np.nonzero(mask>0);top,bottom=int(yy.min()),int(yy.max())+1;left,right=int(xx.min()),int(xx.max())+1
    mass=mask.sum();wanted=mass*fraction;columns=mask.sum(0);cum=np.cumsum(columns)
    cut=int(np.searchsorted(cum,wanted));before=cum[cut-1] if cut else 0
    result[top:bottom,left:cut]=1
    result[top:bottom,cut]=np.clip((wanted-before)/max(columns[cut],1e-8),0,1)
    return result

def render(objects,bg,occlusion,target,age=0):
    amodal=layers(objects,age);rgb=bg.copy()
    for v,a in zip(objects,amodal):rgb=rgb*(1-a[...,None])+np.array(v['rgb'],np.float32)*a[...,None]
    visible=amodal.copy();trans=np.ones((64,64),np.float32)
    for i in reversed(range(len(objects))):visible[i]*=trans;trans*=1-amodal[i]
    panel=blocker(amodal[target],occlusion*(3-age)/3)
    rgb=rgb*(1-panel[...,None])+np.array([116,120,124],np.float32)*panel[...,None]
    visible*=1-panel
    return np.rint(rgb).clip(0,255).astype(np.uint8),amodal,visible,panel

def vector_key(objects):
    current=[dict(v) for v in objects];variants=[]
    for _ in range(4):
        variants.append(tuple(sorted((v['shape'],tuple(v['rgb']),v['x'],v['y'],v['radius'],v['pose']%(2 if v['shape']==3 else 4),v['vx'],v['vy']) for v in current)))
        current=[{**v,'x':v['y'],'y':63-v['x'],'vx':v['vy'],'vy':-v['vx'],'pose':(v['pose']+1)%4} for v in current]
    return hashlib.sha256(repr(min(variants)).encode()).hexdigest()

def sample(seed,split,index,count,condition):
    same,continuous,textured,fraction=condition
    rng=np.random.default_rng(np.random.SeedSequence([seed,{'train':0,'calibration':1,'val':2,'test':3}[split],index,count]))
    palette=np.array(r1.o.PALETTE);colors=[]
    if continuous:
        for _ in range(count):
            for _ in range(1000):
                rgb=rng.integers(45,246,size=3)
                if rgb.max()-rgb.min()>70 and all(np.linalg.norm(rgb-np.array(old))>45 for old in colors):break
            colors.append(rgb.tolist())
    else:colors=palette[rng.choice(6,count,replace=False)].tolist()
    if same:colors=[colors[0]]*count
    objects=[]
    for i in range(count):
        shape=int(rng.integers(4));radius=int(rng.integers(5,9));pose=int(rng.integers(4))
        vx,vy=map(int,rng.integers(-1,2,size=2))
        for _ in range(5000):
            x,y=map(int,rng.integers(12,52,size=2));v={'id':i,'shape':shape,'pose':pose,'radius':radius,'x':x,'y':y,'vx':vx,'vy':vy,'rgb':colors[i]}
            candidate=objects+[v]
            if all(not ((layers(candidate,age)>.05).sum(0)>1).any() for age in range(4)):break
        else:raise RuntimeError('Placement exhausted without reducing count or size')
        objects.append(v)
    target=index%count;bg_seed=seed*100000+index+{'train':0,'calibration':40000,'val':50000,'test':70000}[split]
    bg=background(bg_seed,textured)
    frames=[];first=None
    for age in range(4):
        image,amodal,visible,panel=render(objects,bg,fraction,target,age);frames.append(image)
        if age==0:first=(amodal,visible,panel)
    amodal,visible,panel=first
    yy,xx=np.nonzero(visible[target]>.1)
    if not len(xx):raise RuntimeError('No visible click target')
    nearest=int(np.argmin((xx-xx.mean())**2+(yy-yy.mean())**2));click=[int(xx[nearest]),int(yy[nearest])]
    actual=1-visible[target].sum()/amodal[target].sum()
    # Other object fringe overlap can add a tiny amount beyond the panel target.
    assert abs(actual-fraction)<.005,(actual,fraction)
    record={'source_index':index,'objects':objects,'target_id':target,'click':click,'same_color':same,'continuous_color':continuous,'textured_background':textured,
        'requested_occlusion':fraction,'actual_target_occlusion':float(actual),'visible_fraction':[float(v.sum()/a.sum()) for v,a in zip(visible,amodal)],
        'background_seed':bg_seed,'family_sha256':vector_key(objects),'window_order':'current,t-1,t-2,t-3','current_reference_index':0}
    return np.stack(frames),amodal,visible,record
