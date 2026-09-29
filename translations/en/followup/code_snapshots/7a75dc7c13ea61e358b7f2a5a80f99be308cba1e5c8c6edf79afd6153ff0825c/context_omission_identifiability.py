"""Constructed identical observed state/action, different next-state labels."""
import json
import numpy as np
from common import HERE,dump,r
import dynamics_world as d
from dynamics_context_omission import NAME,MASKS,freeze


def closed_form(state,context,action):
    net=context[:2]+context[2:4];gamma=context[4];times=np.arange(1,9)/8
    gain=-np.expm1(-gamma*times)/gamma if gamma else times
    velocity=np.exp(-gamma*times[:,None,None])*(state[None,:,2:4]+action[None])+gain[:,None,None]*net
    return np.concatenate([state[:,:2]+velocity.sum(0)/8,velocity[-1]],axis=-1)


def main():
    freeze();root=HERE/f'reports/{NAME}/identifiability';root.mkdir(parents=True,exist_ok=False)
    state=np.array([[-12.,0.,1.1,.2,4.,0.,1.],[12.,1.,-.4,.3,5.,1.,1.]])
    action=np.array([[.6,-.2],[0.,0.]])
    cases={'no_net':(np.array([0.,.03,-.02,0.,.004]),np.array([0.,.05,.03,0.,.004])),
           'no_drag':(np.array([0.,.04,.02,.01,.002]),np.array([0.,.04,.02,.01,.009])),
           'no_context':(np.array([0.,.03,-.02,0.,.002]),np.array([0.,.05,.03,0.,.009]))}
    rows=[];arrays={'current_state':state,'action':action};scale=np.array([31.5,31.5,3,3]);checks=0
    rng=np.random.default_rng(997501)
    for condition,(ca,cb) in cases.items():
        observed=[np.r_[c[:2]+c[2:4],c[4]]/np.array([.1,.1,.01])*MASKS[condition] for c in [ca,cb]]
        np.testing.assert_array_equal(observed[0],observed[1])
        aa,ea=d.step(state,ca,action);bb,eb=d.step(state,cb,action)
        assert ea['pair_impulses']==eb['pair_impulses']==ea['wall_impulses']==eb['wall_impulses']==0
        for c,out in [(ca,aa),(cb,bb)]:np.testing.assert_allclose(out[:,:4],closed_form(state,c,action),atol=1e-14,rtol=0)
        ya,yb=aa[:,:4]/scale,bb[:,:4]/scale;middle=(ya+yb)/2
        floor=float(np.mean((ya-yb)**2)/4);assert floor>0
        for prediction in [middle,*[middle+rng.normal(0,.03,middle.shape) for _ in range(32)]]:
            actual=(np.mean((prediction-ya)**2)+np.mean((prediction-yb)**2))/2
            expected=np.mean((prediction-middle)**2)+floor
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-18);checks+=1
        rows.append({'condition':condition,'identical_observed_input':True,'different_labels':True,
                     'max_next_position_difference_pixels':float(np.abs(aa[:,:2]-bb[:,:2]).max()),
                     'max_next_velocity_difference_pixels_per_frame':float(np.abs(aa[:,2:4]-bb[:,2:4]).max()),
                     'equal_pair_normalized_MSE_minimum':floor})
        for key,value in [('context_a',ca),('context_b',cb),('observed_context',observed[0]),('next_a',aa),('next_b',bb)]:arrays[condition+'_'+key]=value
    np.savez_compressed(root/'pairs.npz',**arrays)
    lines=['# Example where there is no environmental information, the answer is not determined in one way','',
        'We kept the exact state and behavior of the two objects identical at the moment and only changed the environmental values hidden in the model. The inputs that blocked force or resistance were the same, but the next correct answer was different. The object sizes, speeds, and environmental values for the three examples were selected within the learning range of variable_force. However, as a configuration example where the same current state is assigned to different environments, it does not represent the frequency of random data.','',
        '| Hidden Information | Maximum difference at the next location (px) | Maximum difference at the next speed (px/frame) | Minimum normalized MSE for both cases |','| --- | ---: | ---: | ---: |']
    labels={'no_net':'completion','no_drag':'Resistance','no_context':'Competence and resistance\n\nA deterministic prediction model that receives an input such as'}
    for x in rows:lines.append(f"| {labels[x['condition']]} | {x['max_next_position_difference_pixels']:.8f} | {x['max_next_velocity_difference_pixels_per_frame']:.8f} | {x['equal_pair_normalized_MSE_minimum']:.10g} |")
    lines+=['',
        'outputs the same value. If the two correct answers are a and b, the squared error calculated with equal weights for both cases is the sum of “the squared distance between the prediction and the median (a+b)/2” and “one-fourth of the squared distance between the two correct answers”. Therefore, even if the minimum value is obtained from the median, an error remains. The table was calculated using the position/31.5 and speed/3 after normalization as the mean of the object and coordinate values.\n\nThe','',
        'simulator\'s one-step results were individually tested using closed-form free-motion equations at 8 time points, and wall and object collisions were confirmed to occur. The squared error decomposition was also tested for 32 predictions that differed from the median. This serves as a proof-of-concept test for the three configuration cases and does not indicate the Bayes error, long-term prediction error lower bound, or achievement of learning performance for any arbitrary set of actual evaluation data.','',
        'If we provide all environmental information, the two inputs will differ and this ambiguity will disappear. However, since the learned model may be wrong, we evaluate the information shortage and model learning failure separately.']
    (root/'EXPLANATION_KO.md').write_text('\n'.join(lines)+'\n')
    dump(root/'verification.json',{'all_passed':True,'constructed_pairs':3,'independent_free_flight_reference_checks':6,
        'squared_loss_identity_checks':checks,'results':rows,'pairs_sha256':r.sha(root/'pairs.npz'),
        'explanation_sha256':r.sha(root/'EXPLANATION_KO.md'),'script_sha256':r.sha(__file__),
        'simulator_sha256':r.sha(HERE/'dynamics_world.py'),'scope':'Three constructed equal-weight pairs only; not an empirical benchmark-wide Bayes bound.'})
    print('context omission identifiability checked',rows,flush=True)


if __name__=='__main__':main()
