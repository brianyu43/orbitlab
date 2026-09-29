"""An exact-past ambiguity example under the declared simulator, not a dataset rate."""
import json
import numpy as np
from PIL import Image,ImageDraw
from common import HERE,dump,fresh_dir,r
from dynamics_world import rollout,render


def main():
    root=fresh_dir(HERE/'reports/observation_identifiability_v1')
    velocity=np.array([.7,.4]);initial=np.array([[-15.,-10.,*velocity,4.,1.,1.],[10.,10.,*velocity,5.,3.,1.]])
    actions=np.zeros((19,2,2));actions[3,0]=[.8,-.3];contexts=[];states=[];past=[]
    for drag in [.004,.009]:
        net=drag*velocity;context=np.array([0.,.0205,net[0],net[1]-.0205,drag])
        assert .02<=context[1]<=.06 and -.05<=context[2]<=.05 and -.02<=context[3]<=.02 and 0<=drag<.01
        trajectory,_=rollout(initial,context,actions);contexts.append(context);states.append(trajectory)
        past.append(np.stack([render(s)['image'] for s in trajectory[:4]]))
    np.testing.assert_allclose(states[0][:4],states[1][:4],atol=1e-13,rtol=0);np.testing.assert_array_equal(past[0],past[1])
    target0=states[0][4,:,:4];target1=states[1][4,:,:4]
    expected_difference=actions[3,0]*(np.exp(-contexts[0][4])-np.exp(-contexts[1][4]))
    np.testing.assert_allclose(states[0][4,0,2:4]-states[1][4,0,2:4],expected_difference,atol=1e-13,rtol=0)
    lower_bound=float(np.mean((target0-target1)**2)/4)
    np.savez_compressed(root/'counterexample.npz',initial=initial,contexts=np.stack(contexts),actions=actions,
        trajectories=np.stack(states),observed_images=np.stack(past))
    canvas=Image.new('RGB',(4*192,222),'white');draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(past[0]):
        canvas.paste(Image.fromarray(frame).resize((192,192),Image.Resampling.NEAREST),(i*192,30));draw.text((i*192+8,10),f't={i}: identical RGB in both worlds',fill='black')
    canvas.save(root/'identical_past.png')
    text='''# If a different future is possible with the same past video

2026-09-23. This is a boundary case configured within the simulator, and it is not the result of measuring the frequency of such cases among actual evaluation data.

When two objects move at the same constant speed v and the external resultant force a follows the relationship a = gamma * v with the resistance coefficient gamma, the forces and resistances cancel each other out. For each substep, the speed update is given by v*exp(-gamma*h) + a*(1-exp(-gamma*h))/gamma = v, so even if gamma is different, the past position and speed may still be the same.

Two environments were created with gamma = 0.004 and 0.009. The gravitational, wind, and resistance values for both environments are within the original variable_force range. The initial state and the four RGB inputs and input behaviors are identical, but when the first object is given the same velocity change k, the subsequent velocity becomes v + k*exp(-gamma), resulting in different values.

Therefore, there is no deterministic video prediction model that can definitively identify which of these two scenarios without additional information. Since the input is the same, the output is also the same. If we treat both cases with the same probability and evaluate the average squared error of the next x/y/vx/vy coordinates of the object, the minimum average error of the common output is the value obtained by dividing the square root of the average of the two difference squares by 4. This conditional lower bound for the two cases does not extend the lower bound of the entire dataset to the overall error bound.

If the speeds of several objects are sufficiently different or if the speed changes during observation, information can be generated that distinguishes resistance and force. This is not an assertion that all general non-contact trajectories are indistinguishable. It is an example used to distinguish boundaries that contain no information in the input, and cases where the model fails to use information properly, even if past videos are perfect.

The actual two trajectories and inputs are in counterexample.npz, the numerical values and verification are in verification.json, and the same four past images are in identical_past.png. They do not replace the observation length and independent repetition entire evaluation of C07.'''
    (root/'EXPLANATION_KO.md').write_text(text)
    dump(root/'verification.json',{'all_passed':True,'past_rgb_identical':True,'past_state_max_difference':float(abs(states[0][:4]-states[1][:4]).max()),
        'next_motion_max_difference':float(abs(target0-target1).max()),'equal_prior_next_motion_mse_lower_bound':lower_bound,
        'closed_form_kick_velocity_difference_verified':True,'contexts_within_declared_variable_force_ranges':True,
        'data_sha256':r.sha(root/'counterexample.npz'),'figure_sha256':r.sha(root/'identical_past.png'),'script_sha256':r.sha(__file__),
        'scope':'Constructed boundary example, not incidence in the sampled benchmark or a general dataset lower bound.'})
    print('Identical past, different next velocity verified; conditional two-case MSE floor',lower_bound)


if __name__=='__main__':main()
