import numpy as np
from dynamics_world import render,rollout
from dynamics_rgb_baseline import measure_frame,infer,fit_context,coefficients


def test_free_flight_recovery_and_frame_integration():
    s=np.array([[-14.,-9.,.9,.6,4.,1.,1.],[12.,10.,-.4,.3,6.,5.,1.]])
    context=np.array([0.,.045,.03,-.01,.006]);trajectory,_=rollout(s,context,np.zeros((3,2,2)))
    positions=np.zeros((4,6,2));presence=np.zeros((4,6));features=np.zeros((4,6,7))
    for i,color in enumerate([1,5]):
        positions[:,color]=trajectory[:,i,:2];presence[:,color]=1
        features[:,color,:2]=trajectory[:,i,:2];features[:,color,2]=trajectory[:,i,4];features[:,color,3]=1
    expected=np.array([.03,.035,.006]);fitted,info=fit_context(positions,presence)
    np.testing.assert_allclose(fitted,expected,atol=1e-12,rtol=0);assert info['identified_design']
    predicted,context_est,_=infer(features)
    np.testing.assert_allclose(predicted[[1,5]],trajectory[-1],atol=1e-12,rtol=0)


def test_rgb_circle_fit_and_missing_objects():
    s=np.array([[-14.17,-9.31,.9,.6,4.,1.,1.],[12.46,10.08,-.4,.3,6.,5.,1.]])
    measured=measure_frame(render(s)['image']);assert np.where(measured[:,3]>.5)[0].tolist()==[1,5]
    np.testing.assert_allclose(measured[[1,5],:3],s[:,[0,1,4]],atol=.02,rtol=0)
    empty=measure_frame(np.zeros((64,64,3),np.uint8));assert not empty.any()
    state,ctx,info=infer(np.stack([empty]*4));assert not state.any() and not ctx.any() and not info['identified_design']


def test_equal_visible_past_can_hide_distinct_drag_and_action_response():
    velocity=np.array([.7,.4]);s=np.array([[-15.,-10.,*velocity,4.,1.,1.],[10.,10.,*velocity,5.,3.,1.]])
    actions=np.zeros((4,2,2));actions[3,0]=[.8,-.3];results=[]
    for drag in [.005,.01]:
        net=drag*velocity;context=np.array([0.,.02,net[0],net[1]-.02,drag])
        states,_=rollout(s,context,actions);results.append(states)
    np.testing.assert_allclose(results[0][:4],results[1][:4],atol=1e-13,rtol=0)
    for t in range(4):np.testing.assert_array_equal(render(results[0][t])['image'],render(results[1][t])['image'])
    assert np.max(np.abs(results[0][4]-results[1][4]))>.001
    positions=np.zeros((4,6,2));presence=np.zeros((4,6))
    for i,color in enumerate([1,3]):positions[:,color]=results[0][:4,i,:2];presence[:,color]=1
    fitted,info=fit_context(positions,presence);assert not info['identified_design']


if __name__=='__main__':
    for name in sorted(n for n in list(globals()) if n.startswith('test_')):
        globals()[name]();print(name,'passed')
