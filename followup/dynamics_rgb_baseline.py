"""Renderer-aware RGB circle measurements and free-flight system identification.

This is an explicitly non-learned reference with known palette/circular shape
and the eight-substep integration rule. It never reads GT state or event data.
"""
import json
import time
import numpy as np
from scipy.optimize import least_squares
from common import HERE,dump,fresh_dir,r
from dynamics_world import PALETTE
from dynamics_observation_inputs import NAME as INPUTS

NAME='dynamics_rgb_baseline_v1'
FEATURES=['x','y','radius','present','alpha_mass','fit_rms','fit_converged']


def measure_frame(image):
    """Six color-addressed candidates; presence comes from RGB alpha mass only."""
    rgb=image.astype(np.float64);palette=PALETTE.copy();p2=(palette**2).sum(1)
    alpha=np.einsum('hwc,kc->hwk',rgb,palette)/p2
    error=((rgb[:,:,None,:]-np.clip(alpha,0,1)[:,:,:,None]*palette[None,None,:,:])**2).sum(-1)
    assigned=error.argmin(-1);best=np.take_along_axis(error,assigned[...,None],axis=-1)[...,0]
    yy,xx=np.mgrid[:64,:64].astype(float);xx-=31.5;yy-=31.5;output=np.zeros((6,len(FEATURES)))
    for color in range(6):
        mask=(assigned==color)&(best<=4)&(alpha[:,:,color]>.05)
        mass=np.where(mask,np.clip(alpha[:,:,color],0,1),0);area=mass.sum()
        if area<12:continue
        cx=float((xx*mass).sum()/area);cy=float((yy*mass).sum()/area);radius=float(np.clip(np.sqrt(area/np.pi),2.01,7.99))
        window=(abs(xx-cx)<=10)&(abs(yy-cy)<=10)
        # Other colors and mixed fringe pixels are not treated as black background.
        use=window&(((rgb**2).sum(-1)<2.25)|((assigned==color)&(best<=4)))
        x=xx[use];y=yy[use];observed=np.clip(alpha[:,:,color][use],0,1)
        def residual(theta):
            distance=np.sqrt((x-theta[0])**2+(y-theta[1])**2)
            return np.clip(theta[2]+.5-distance,0,1)-observed
        def jacobian(theta):
            dx=x-theta[0];dy=y-theta[1];distance=np.maximum(np.sqrt(dx*dx+dy*dy),1e-12)
            edge=((theta[2]+.5-distance)>0)&((theta[2]+.5-distance)<1)
            return np.stack([dx/distance,dy/distance,np.ones_like(x)],axis=1)*edge[:,None]
        fit=least_squares(residual,[cx,cy,radius],jac=jacobian,
            bounds=([cx-2,cy-2,2],[cx+2,cy+2,8]),max_nfev=30,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if not np.isfinite(fit.x).all():raise FloatingPointError('Nonfinite RGB circle fit')
        output[color]=[*fit.x,1.,area,float(np.sqrt(np.mean(fit.fun**2))),float(fit.success)]
    return output


def coefficients(drag):
    d=np.exp(-drag);sub=np.arange(1,9)/8
    A=float(np.exp(-drag*sub).mean())
    if drag<1e-10:return 1.,1.,.5625,1.
    B=float((-np.expm1(-drag*sub)/drag).mean());b=float(-np.expm1(-drag)/drag)
    return A,d,B,b


def fit_context(position,presence):
    valid=np.all(presence>=.5,axis=0);tracked=int(valid.sum())
    if not tracked:return np.zeros(3),{'complete_tracks':0,'design_variance':0.,'unclipped_decay':1.,'residual_rms':None,'identified_design':False}
    delta=np.diff(position[:,valid,:],axis=0)
    x=delta[:-1].reshape(-1,2);y=delta[1:].reshape(-1,2)
    xx=x-x.mean(0);yy=y-y.mean(0);denominator=float((xx*xx).sum())
    raw=float((xx*yy).sum()/denominator) if denominator>1e-12 else 1.
    decay=float(np.clip(raw,np.exp(-.05),1.));drag=float(-np.log(decay));b=1. if drag<1e-10 else float(-np.expm1(-drag)/drag)
    force=(y-decay*x).mean(0)/b;residual=y-decay*x-b*force
    return np.array([force[0],force[1],drag]),{'complete_tracks':tracked,'design_variance':denominator,
        'unclipped_decay':raw,'residual_rms':float(np.sqrt(np.mean(residual**2))),'identified_design':denominator>1e-12}


def infer(features):
    """Return six color slots and context from four image measurements only."""
    if features.shape!=(4,6,7):raise ValueError('Exactly four RGB measurements are required')
    context,info=fit_context(features[:,:,:2],features[:,:,3]);A,decay,B,b=coefficients(context[2])
    state=np.zeros((6,7),np.float64)
    for color in range(6):
        if features[-1,color,3]<.5:continue
        delta=features[-1,color,:2]-features[-2,color,:2] if features[-2,color,3]>=.5 else np.zeros(2)
        velocity=decay*(delta-B*context[:2])/A+b*context[:2]
        state[color]=[*features[-1,color,:2],*velocity,features[-1,color,2],color,1]
    return state,context,info


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():dump(path,{'input_manifest_sha256':r.sha(HERE/f'data/{INPUTS}/manifest.json'),
        'source_sha256':r.sha(__file__),'palette_sha256':r.sha(HERE/'dynamics_world.py'),
        'known_priors':'Six renderer colors, circular radial alpha, fixed black background, equal masses, distinct colors per scene, eight substeps/frame. Not learned visual concepts or discovered physics.',
        'image_only':'Color identity is an observable track label. No true count, IDs, states, modes, contexts, events or future pixels enter measurement or fitting.',
        'circle_fit':'RGB projection on each known palette ray. Squared RGB residual <=4, alpha>.05 for initial mass; detect if alpha mass>=12. Fit center and continuous radius in a 21x21 neighborhood, radius bounds [2,8], center +/-2px, max 30 evaluations.',
        'dynamics_fit':'Three position increments from four frames. Shared scalar decay and two force intercepts fit by centered least squares over all complete tracks and both adjacent-increment pairs. Decay clipped to [exp(-.05),1], force not clipped.',
        'contact_policy':'No GT event filtering. Free-flight equations applied to all episodes; past-contact strata are used only for scoring. No complete track => zero context; missing penultimate observation => zero last displacement.',
        'identification':'Zero design variance does not identify drag; fallback decay=1 is a convention, not recovery of true drag.',
        'features':FEATURES,'frozen_unix':time.time()})
    c=json.loads(path.read_text());assert c['source_sha256']==r.sha(__file__)
    assert c['input_manifest_sha256']==r.sha(HERE/f'data/{INPUTS}/manifest.json') and c['palette_sha256']==r.sha(HERE/'dynamics_world.py')
    return c


def main():
    c=freeze();root=fresh_dir(HERE/f'data/{NAME}');manifest=json.loads((HERE/f'data/{INPUTS}/manifest.json').read_text());groups=[]
    for g in manifest['groups']:
        mode,split=g['mode'],g['split'];path=HERE/f'data/{INPUTS}/{mode}/{split}/observations.npz'
        assert r.sha(path)==g['files']['observations.npz']
        with np.load(path) as a:images=a['images'].copy();assert a.files==['images']
        features=np.stack([[measure_frame(frame) for frame in sequence] for sequence in images]);states=[];contexts=[];info=[]
        for f in features:
            s,ctx,detail=infer(f);states.append(s);contexts.append(ctx);info.append(detail)
        folder=fresh_dir(root/mode/split);np.savez_compressed(folder/'predictions.npz',features=features,states=np.stack(states),contexts=np.stack(contexts));dump(folder/'fit_info.json',info)
        groups.append({'mode':mode,'split':split,'episodes':len(images),'source_rgb_sha256':r.sha(path),
            'prediction_sha256':r.sha(folder/'predictions.npz'),'info_sha256':r.sha(folder/'fit_info.json')})
        print('RGB measurement and kinematics',mode,split,len(images),flush=True)
    dump(root/'manifest.json',{'groups':groups,'episodes':sum(v['episodes'] for v in groups),
        'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'claim_boundary':c['known_priors']})


if __name__=='__main__':main()
