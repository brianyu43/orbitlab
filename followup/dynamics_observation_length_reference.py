"""Visible-window-only analytic reference with explicit short-window defaults."""
import numpy as np
from dynamics_rgb_baseline import fit_context,coefficients

LENGTHS=[1,2,4,8]


def infer_window(features):
    features=np.asarray(features,np.float64);length=len(features)
    if length not in LENGTHS or features.shape[1:]!=(6,7):raise ValueError('Visible measurements only, L in 1/2/4/8')
    if length<3:
        context=np.zeros(3);info={'complete_tracks':int((features[:,:,3]>=.5).all(0).sum()),'design_variance':0.,
            'unclipped_decay':1.,'residual_rms':None,'identified_design':False,'short_window_default':True}
    else:
        context,info=fit_context(features[:,:,:2],features[:,:,3]);info['short_window_default']=False
    A,decay,B,b=coefficients(context[2]);state=np.zeros((6,7))
    for color in range(6):
        if features[-1,color,3]<.5:continue
        displacement=features[-1,color,:2]-features[-2,color,:2] if length>=2 and features[-2,color,3]>=.5 else np.zeros(2)
        velocity=decay*(displacement-B*context[:2])/A+b*context[:2] if length>=2 else np.zeros(2)
        state[color]=[*features[-1,color,:2],*velocity,features[-1,color,2],color,1.]
    return state,context,info


def select_visible(features,length):
    if length not in LENGTHS or features.shape[-3:]!=(8,6,7):raise ValueError('Eight-position carrier expected')
    # Slice before fitting. Hidden measurements never contribute to a difference,
    # force fit, feature normalization or model input.
    return np.array(features[...,8-length:,:,:],copy=True)
