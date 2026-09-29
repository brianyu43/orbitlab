"""Replay paired t=7 long forecasts and opposite actions for each observed length."""
import argparse
import json
import time
import numpy as np
import torch
from common import HERE,dump,r
from dynamics_observation_length_data import NAME
from dynamics_observation_length_eval import freeze,config_name,state_checkpoint
from dynamics_models import Transition
from dynamics_autonomous import HORIZONS,LEARNED
from dynamics_autonomous_metrics import score_rollout,score_counterfactual,summarize
from verify_dynamics_autonomous import canonical,replay,unpack,independent_contact,independent_primary,independent_response


def independent_inputs(ds,length,method,seed,mode,split):
    source=HERE/f'data/{NAME}/d{ds}/inputs/{mode}/{split}'
    with np.load(source/'commands.npz') as a:colors=a['target_color'].copy();dv=a['delta_velocity'].copy()
    if method=='oracle':
        with np.load(source/'labels.npz') as a:
            state=canonical(a['past_states'][:,7:8])[:,0];ctx=np.c_[a['net_acceleration'],a['drag']]
        scores=state[:,:,6]
    else:
        folder=HERE/f'runs/{NAME}/d{ds}/L{length}/observation_eval/{method}_s{seed}/{mode}/{split}'
        metric=json.loads((folder/'metrics.json').read_text())
        assert metric['files']['predictions.npz']==r.sha(folder/'predictions.npz')
        with np.load(folder/'predictions.npz') as a:
            state=a['estimated_states'].copy();ctx=a['estimated_contexts'].copy();scores=a['presence_scores'].copy()
    order=np.sort(np.argsort(-scores,axis=1,kind='stable')[:,:4],axis=1)
    initial=np.take_along_axis(state,order[:,:,None],axis=1)
    selected=(order==colors[:,None])&(initial[:,:,6]>.5)
    return {'initial':initial,'context':ctx,'order':order,'action':selected[:,:,None]*dv[:,None],
            'target_colors':colors,'found':selected.any(1),
            'overflow':(state[:,:,6]>.5).sum(1)-(initial[:,:,6]>.5).sum(1)}


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-seed',type=int,required=True);parser.add_argument('--length',type=int,choices=[1,2,4,8],required=True);args=parser.parse_args()
    ds=args.data_seed;length=args.length;c=freeze(ds,length);torch.set_num_threads(2);started=time.time()
    report=HERE/f'reports/{NAME}/d{ds}/L{length}';summary=json.loads((report/'autonomous_summary.json').read_text())
    config_sha=r.sha(HERE/f'configs/{config_name(ds,length)}.json')
    assert summary['config_sha256']==config_sha
    observation=json.loads((report/'observation_eval_verification.json').read_text())
    assert observation['all_passed'] and observation['summary_sha256']==r.sha(report/'observation_eval_summary.json')
    assert observation['verifier_sha256']==r.sha(HERE/'verify_observation_length_eval.py')
    expected={(m,s,g['mode'],g['split'],k) for m in c['autonomous_methods'] for s in c['seeds']
              for g in c['groups'] for k in c['autonomous_predictors']}
    listed=[(v['method'],v['seed'],v['mode'],v['split'],v['predictor']) for v in summary['results']]
    assert len(listed)==len(set(listed))==len(expected)==1404 and set(listed)==expected
    count=frames=cf_frames=primary=response=aggregate=inputs_checked=contact_frames=first_steps=0
    calibration=json.loads((report/'contact_proxy_calibration/manifest.json').read_text())['groups']
    assert len(calibration)==len(c['groups'])==13
    assert {(x['mode'],x['split']) for x in calibration}=={(x['mode'],x['split']) for x in c['groups']}
    calibration_scores=[]
    for g in c['groups']:
        assert r.sha(HERE/g['path'])==g['files']['trajectories.npz']
        with np.load(HERE/g['path']) as a:
            raw=a['states'][:,7:].copy();events=a['events'][:,7:].copy();action=a['actions'][:,7].copy()
            context=np.c_[a['contexts'][:,:2]+a['contexts'][:,2:4],a['contexts'][:,4]]
        cm=next(x for x in calibration if (x['mode'],x['split'])==(g['mode'],g['split']))
        assert cm['sha256']==r.sha(HERE/cm['path'])
        flags,valid=independent_contact(raw[...,:4],raw[:,0,:,4],raw[:,0,:,6]>.5,context,action,np.ones(raw.shape[:2],bool))
        with np.load(HERE/cm['path']) as a:
            np.testing.assert_array_equal(flags,a['predicted']);np.testing.assert_array_equal(events,a['true_events'])
            np.testing.assert_array_equal(valid,a['valid'])
        scores={}
        for j,kind in enumerate(['pair','wall']):
            actual=events[:,:,j]>0;pred=flags[:,:,j]
            scores[kind]={'tp':int((actual&pred).sum()),'fp':int((~actual&pred).sum()),'fn':int((actual&~pred).sum())}
        calibration_scores.append({'mode':g['mode'],'split':g['split'],'metrics':scores})
        contact_frames+=len(raw)*61
    models={}
    for seed in c['seeds']:
        for kind in LEARNED:
            ck=torch.load(state_checkpoint(ds,kind,seed),map_location='cpu',weights_only=True)
            model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();models[seed,kind]=model
    for method in c['autonomous_methods']:
        for seed in c['seeds']:
            for g in c['groups']:
                mode,split=g['mode'],g['split'];root=HERE/f'runs/{NAME}/d{ds}/L{length}/autonomous/{method}_s{seed}/{mode}/{split}'
                inputs=independent_inputs(ds,length,method,seed,mode,split)
                with np.load(root/'initial_inputs.npz') as a:
                    assert set(a.files)==set(inputs)
                    for name,value in inputs.items():np.testing.assert_array_equal(a[name],value)
                inputs_checked+=len(inputs['initial'])
                with np.load(HERE/g['path']) as a:
                    truth=canonical(a['states'][:,7:]);events=a['events'][:,7:].copy()
                    cf_truth=canonical(a['counterfactual_states'][:,7:]) if 'counterfactual_states' in a else None
                    cf_events=a['counterfactual_events'][:,7:].copy() if cf_truth is not None else None
                for kind in c['autonomous_predictors']:
                    folder=root/kind;m=json.loads((folder/'metrics.json').read_text())
                    assert m['length']==length
                    assert (m['data_seed'],m['method'],m['seed'],m['mode'],m['split'],m['predictor'])==(ds,method,seed,mode,split,kind)
                    assert m['input_sha256']==r.sha(root/'initial_inputs.npz') and m['config_sha256']==config_sha
                    assert m['files']['trajectories.npz']==r.sha(folder/'trajectories.npz')
                    with np.load(folder/'trajectories.npz') as a:stored={k:a[k].copy() for k in a.files}
                    model=models.get((seed,kind));base=None
                    for branch,label,event,sign in [('base',truth,events,1),('cf',cf_truth,cf_events,-1)]:
                        if label is None:continue
                        prefix='' if branch=='base' else 'cf_'
                        result=replay(inputs['initial'],inputs['context'],inputs['action']*sign,kind,model)
                        for key,value in result.items():np.testing.assert_allclose(stored[prefix+key],value,rtol=1e-10,atol=1e-8,equal_nan=True)
                        flags,valid=independent_contact(result['motion'],inputs['initial'][:,:,4],inputs['initial'][:,:,6]>.5,
                                                        inputs['context'],sign*inputs['action'],result['finite'])
                        np.testing.assert_array_equal(stored[prefix+'predicted_events'],flags)
                        np.testing.assert_array_equal(stored[prefix+'event_valid'],valid)
                        keys,values,_,_=score_rollout(result,inputs['initial'],inputs['order'],inputs['context'],sign*inputs['action'],
                                                      label,event,inputs['found'],inputs['overflow'])
                        np.testing.assert_array_equal(stored['metric_keys'],keys)
                        np.testing.assert_allclose(stored[prefix+'metric_values'],values,atol=1e-8,rtol=1e-10,equal_nan=True)
                        assert summarize(keys,stored[prefix+'metric_values'])==m['factual' if branch=='base' else 'counterfactual']
                        aggregate+=len(keys)*len(HORIZONS)
                        primary+=independent_primary(result,inputs,label,keys,stored[prefix+'metric_values'])
                        frames+=len(label)*61 if branch=='base' else 0;cf_frames+=len(label)*61 if branch=='cf' else 0
                        if branch=='base':base=result
                        else:
                            keys,values=score_counterfactual(base,result,inputs['initial'],inputs['order'],truth,cf_truth,inputs['target_colors'])
                            np.testing.assert_array_equal(keys,stored['response_metric_keys'])
                            np.testing.assert_allclose(values,stored['response_metric_values'],atol=1e-8,rtol=1e-10,equal_nan=True)
                            assert summarize(keys,stored['response_metric_values'])==m['response'];aggregate+=len(keys)*len(HORIZONS)
                            response+=independent_response(base,result,inputs,truth,cf_truth,keys,stored['response_metric_values'])
                    assert m['nonfinite_episodes']==int((base['first_nonfinite']>=0).sum())
                    if cf_truth is None:
                        assert m['counterfactual'] is None and m['response'] is None and not any(k.startswith('cf_') for k in stored)
                    if kind in c['observation_predictors']:
                        parent='rgb_cnn' if method=='oracle' else method
                        info='true_state_true_context' if method=='oracle' else 'estimated_state_estimated_context'
                        with np.load(HERE/f'runs/{NAME}/d{ds}/L{length}/observation_eval/{parent}_s{seed}/{mode}/{split}/predictions.npz') as a:
                            expected_first=a[info+'__'+kind][...,:4]
                        actual=unpack(base,inputs)[0][:,1]
                        np.testing.assert_allclose(actual,expected_first,atol=2e-5,rtol=1e-5)
                        first_steps+=len(actual)
                        assert m['first_step_C04_max_abs_error'] is not None
                    else:assert m['first_step_C04_max_abs_error'] is None
                    count+=1
                dump(report/'autonomous_verification_progress.json',{'verified_conditions':count,'expected_conditions':len(expected),
                                                                      'elapsed_seconds':time.time()-started})
                print('verified length autonomous',ds,method,seed,mode,split,count,flush=True)
    for item in summary['results']:
        path=HERE/f"runs/{NAME}/d{ds}/L{length}/autonomous/{item['method']}_s{item['seed']}/{item['mode']}/{item['split']}/{item['predictor']}/metrics.json"
        assert item==json.loads(path.read_text())
    dump(report/'autonomous_verification.json',{'all_passed':True,'data_seed':ds,'length':length,'conditions_verified':count,
        'initial_input_episodes_checked':inputs_checked,'factual_forecast_frames_replayed':frames,'counterfactual_forecast_frames_replayed':cf_frames,
        'independent_primary_metric_checks':primary,'independent_response_metric_checks':response,'aggregate_metric_checks':aggregate,
        'true_trajectory_contact_proxy_frames_checked':contact_frames,'contact_proxy_calibration':calibration_scores,
        'first_steps_compared_to_C04':first_steps,'input_packing_and_public_color_actions_independently_verified':True,
        'all_recurrences_replayed_without_future_labels':True,'summary_sha256':r.sha(report/'autonomous_summary.json'),
        'verifier_sha256':r.sha(__file__),'reference_helpers_sha256':r.sha(HERE/'verify_dynamics_autonomous.py'),
        'observation_verification_sha256':r.sha(report/'observation_eval_verification.json'),'seconds':time.time()-started,
        'scope':'Every condition and recurrent prediction for this data unit and observed length. Primary endpoint/response and contact checks use separate reference implementations; remaining metrics replay locked score code. Across-data aggregation remains separate.'})


if __name__=='__main__':main()
