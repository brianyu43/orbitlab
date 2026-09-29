"""Focused recurrence, missing-object, numerical-failure and contact checks."""
import numpy as np
import torch
from dynamics_world import step
from dynamics_models import Transition
from dynamics_autonomous import autonomous, unpack_motion, canonical_sequence, contact_proxy
from dynamics_autonomous_metrics import score_rollout, score_counterfactual


def fixture(n=2):
    state=np.zeros((n,4,7))
    state[:,0]=[-12,0,0,0,4,1,1]
    state[:,1]=[12,0,0,0,4,3,1]
    action=np.zeros((n,4,2));action[:,0,0]=.1
    return state,np.zeros((n,3)),action,np.tile([1,3,0,2],(n,1))


def test_recurrence_and_single_action():
    state,ctx,action,order=fixture()
    pred=autonomous(state,ctx,action,'inertial')
    expected=state[:,None,:,:4]+np.zeros((2,62,4,4))
    expected[:,:,:,2:4]=state[:,None,:,2:4]+action[:,None]
    expected[:,0]=state[:,:,:4]
    expected[:,:,:,:2]=state[:,None,:,:2]+np.arange(62)[None,:,None,None]*(state[:,None,:,2:4]+action[:,None])
    np.testing.assert_allclose(pred['motion'],expected,atol=1e-12)
    learned=autonomous(state,ctx,action,'interaction',Transition('interaction'))
    np.testing.assert_allclose(learned['motion'],expected,atol=3e-5)
    assert learned['finite'].all() and (learned['first_nonfinite']==-1).all()


def test_numerical_failure_is_absorbing():
    class FailsOne(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,state,action,context):
            self.calls+=1;value=state[:,:,:4].clone()
            if self.calls==2:value[0]=float('nan')
            return value
    state,ctx,action,order=fixture()
    result=autonomous(state,ctx,action,'interaction',FailsOne())
    assert result['first_nonfinite'].tolist()==[2,-1]
    assert not result['finite'][0,2:].any() and result['finite'][1].all()
    assert np.isnan(result['motion'][0,2:]).all()
    truth=np.broadcast_to(state[:,None],(2,62,4,7)).copy()
    keys,v,_,_=score_rollout(result,state,order,ctx,action,canonical_sequence(truth),np.zeros((2,61,2)),np.ones(2,bool),np.zeros(2))
    assert v[0,1,keys.index('finite_fraction')]==0
    assert np.isnan(v[0,1,keys.index('position_mae_pixels_finite')])
    assert v[0,1,keys.index('scene_success')]==0


def test_scoring_and_response():
    state,ctx,action,order=fixture();base=autonomous(state,ctx,action,'inertial')
    cf=autonomous(state,ctx,-action,'inertial')
    def truth(result):
        raw=np.broadcast_to(state[:,None],(2,62,4,7)).copy();raw[...,:4]=result['motion'];return canonical_sequence(raw)
    t,ct=truth(base),truth(cf)
    keys,values,_,_=score_rollout(base,state,order,ctx,action,t,np.zeros((2,61,2)),np.ones(2,bool),np.zeros(2))
    assert np.all(values[:,:,keys.index('scene_success')]==1)
    assert np.all(values[:,:,keys.index('position_mae_pixels_finite')]==0)
    rk,rv=score_counterfactual(base,cf,state,order,t,ct,np.ones(2,dtype=int))
    assert np.all(rv[:,:,rk.index('target_position_response_mae_finite')]==0)
    assert np.all(rv[:,:,rk.index('nontarget_position_response_mae_finite')]==0)
    assert np.all(rv[:,-1,rk.index('target_position_zero_response_mae')]>0)
    missing=state.copy();missing[:,1]=0
    pred=autonomous(missing,ctx,action,'inertial')
    k,v,_,_=score_rollout(pred,missing,order,ctx,action,t,np.zeros((2,61,2)),np.ones(2,bool),np.zeros(2))
    assert np.all(v[:,:,k.index('recall')]==.5)
    assert np.all(v[:,:,k.index('scene_success')]==0)
    assert np.all(v[:,:,k.index('position_mae_pixels_finite')]>0)


def test_contact_proxy_and_crossing_without_impulse():
    states=[np.array([[-5.,0,1,0,4,1,1],[5.,0,-1,0,4,3,1]]),
            np.array([[27.,0,1,0,4,1,1]])]
    for index,state in enumerate(states):
        nxt,event=step(state,np.zeros(5));motion=np.stack([state[:,:4],nxt[:,:4]])[None]
        p,valid=contact_proxy(motion,state[None,:,4],np.ones((1,len(state)),bool),np.zeros((1,3)),np.zeros((1,len(state),2)),np.ones((1,2),bool))
        assert p[0,0,index] and valid.all()
        assert event['pair_impulses' if index==0 else 'wall_impulses']>0
    state=states[0];nxt=state.copy();nxt[:,:2]+=2*nxt[:,2:4]
    p,_=contact_proxy(np.stack([state[:,:4],nxt[:,:4]])[None],state[None,:,4],np.ones((1,2),bool),np.zeros((1,3)),np.zeros((1,2,2)),np.ones((1,2),bool))
    assert not p.any(), 'Geometric overlap without velocity response must not be scored as a collision'


if __name__=='__main__':
    torch.set_num_threads(2)
    for f in [test_recurrence_and_single_action,test_numerical_failure_is_absorbing,test_scoring_and_response,test_contact_proxy_and_crossing_without_impulse]:
        f();print(f.__name__,'passed')
