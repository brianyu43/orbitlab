"""C05/C06 frozen recurrent forecasts and same-past action counterfactuals."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from common import HERE, dump, fresh_dir, r, update_status
from dynamics_models import Transition
from dynamics_observation_eval import pack, NAME as OBSERVATION_EVAL
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_autonomous import (HORIZONS, LEARNED, PREDICTORS, METHODS, autonomous,
                                canonical_sequence, unpack_motion, contact_proxy)
from dynamics_autonomous_metrics import score_rollout, score_counterfactual, summarize

NAME='dynamics_autonomous_v1'


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        for name in ['dynamics_observation_eval_v1/verification.json','dynamics_state_study_v1/verification.json']:
            assert json.loads((HERE/'reports'/name).read_text())['all_passed']
        manifest=json.loads((HERE/'data/dynamics_world_v1/manifest.json').read_text())
        groups=[g for g in manifest['groups'] if g['split'] not in ['train','val']]
        files={}
        for seed in range(3):
            for kind in LEARNED:
                name=f'runs/dynamics_state_study_v1/variable_force/{kind}_s{seed}/model.pt';files[name]=r.sha(HERE/name)
            for method in METHODS[1:]:
                for g in groups:
                    name=f"runs/{OBSERVATION_EVAL}/{method}_s{seed}/{g['mode']}/{g['split']}/predictions.npz"
                    files[name]=r.sha(HERE/name)
        sources=['dynamics_autonomous.py','dynamics_autonomous_metrics.py','dynamics_autonomous_study.py',
                 'test_dynamics_autonomous.py','dynamics_models.py','dynamics_physical_baselines.py',
                 'dynamics_observation_eval.py','planning_C05_C06_EXECUTION_KO.md']
        dump(path,{'methods':METHODS,'predictors':PREDICTORS,'seeds':[0,1,2],'horizons':HORIZONS,'steps':61,
            'groups':groups,'files':files,'source_sha256':{name:r.sha(HERE/name) for name in sources},
            'world_manifest_sha256':r.sha(HERE/'data/dynamics_world_v1/manifest.json'),
            'observation_manifest_sha256':r.sha(HERE/f'data/{INPUTS}/manifest.json'),
            'initial_information':'C04 cached RGB-only estimates or explicitly privileged t=3 true-state/true-context control. No later state, context, count, mask or matching is fed back. Same variable_force transition checkpoint in every environment.',
            'action':'Public target color and delta velocity, applied once. Counterfactual changes only its sign, retaining identical estimated past and context.',
            'identity':'Initial color-addressed top-four packing stays fixed. Color and presence are carried by design, not learned re-identification. Nearest-true-position accuracy is a diagnostic only; it never reorders predictions.',
            'nonfinite':'Record first failure step, NaN thereafter, failure fraction and finite-only error with denominators. No prediction clipping or added learned-model wall/contact correction.',
            'collision_proxy':{'velocity_residual_threshold':.1,'distance_margin_pixels':1.5,
                'meaning':'Frame interval near contact plus departure from free-flight velocity. Calibrate on true trajectories against simulator events. Derived proxy, not an event head or exact collision count.'},
            'metrics':'Endpoint horizons 1/4/8/16/32/61; all episodes equal weight. Missing true colors are zero-filled for error; matched-only error is separate. Both branches and target/nontarget response errors, including zero-response control.',
            'information_boundary':'One data-generation seed, 1,280 heldout base episodes and 512 same-past counterfactual pairs. Three weight initializations; analytic duplicates are not independent data.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for field in ['files','source_sha256']:
        for name,sha in c[field].items():assert r.sha(HERE/name)==sha,name
    for g in c['groups']:assert r.sha(HERE/g['path'])==g['archive_sha256'],g['path']
    assert r.sha(HERE/'data/dynamics_world_v1/manifest.json')==c['world_manifest_sha256']
    assert r.sha(HERE/f'data/{INPUTS}/manifest.json')==c['observation_manifest_sha256']
    return c


def initial_inputs(method,seed,mode,split):
    folder=HERE/f'data/{INPUTS}/{mode}/{split}'
    with np.load(folder/'commands.npz') as a:colors=a['target_color'].copy();dv=a['delta_velocity'].copy()
    if method=='oracle':
        with np.load(folder/'labels.npz') as a:
            states=canonical_sequence(a['past_states'][:,3:4])[:,0]
            ctx=np.column_stack([a['net_acceleration'],a['drag']])
        scores=states[...,6]
    else:
        with np.load(HERE/f'runs/{OBSERVATION_EVAL}/{method}_s{seed}/{mode}/{split}/predictions.npz') as a:
            states=a['estimated_states'].copy();ctx=a['estimated_contexts'].copy();scores=a['presence_scores'].copy()
    initial,order,action,found,overflow=pack(states,scores,colors,dv)
    return {'initial':initial,'order':order,'context':ctx,'action':action,'target_colors':colors,
            'found':found,'overflow':overflow}


def check_first_step(result,inputs,method,seed,mode,split,kind):
    if kind not in ['interaction','wrong_exact','joint_exact','inertial','force_wall']:return None
    parent='rgb_cnn' if method=='oracle' else method
    info='true_state_true_context' if method=='oracle' else 'estimated_state_estimated_context'
    with np.load(HERE/f'runs/{OBSERVATION_EVAL}/{parent}_s{seed}/{mode}/{split}/predictions.npz') as a:
        expected=a[info+'__'+kind][...,:4]
    actual=unpack_motion(result['motion'][:,1:2],inputs['initial'],inputs['order'])[0][:,0]
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5)
    return float(np.max(np.abs(actual-expected)))


def calibrate_proxy(group):
    mode,split=group['mode'],group['split'];out=HERE/f'reports/{NAME}/contact_proxy_calibration/{mode}/{split}'
    if (out/'summary.json').exists():return json.loads((out/'summary.json').read_text())
    fresh_dir(out)
    with np.load(HERE/group['path']) as a:
        state=a['states'][:,3:].copy();events=a['events'][:,3:].copy();action=a['actions'][:,3].copy()
        context=np.column_stack([a['contexts'][:,:2]+a['contexts'][:,2:4],a['contexts'][:,4]])
    pred,valid=contact_proxy(state[...,:4],state[:,0,:,4],state[:,0,:,6]>.5,context,action,np.ones(state.shape[:2],bool))
    np.savez_compressed(out/'predictions.npz',predicted=pred,true_events=events,valid=valid)
    stats={}
    for j,kind in enumerate(['pair','wall']):
        truth=events[...,j]>0;guess=pred[...,j]
        tp=int((truth&guess).sum());fp=int((~truth&guess).sum());fn=int((truth&~guess).sum())
        stats[kind]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
    result={'mode':mode,'split':split,'episodes':len(state),'metrics':stats,'sha256':r.sha(out/'predictions.npz')}
    dump(out/'summary.json',result);return result


@torch.no_grad()
def main():
    torch.set_num_threads(2);c=freeze();started=time.time();results=[]
    evidence=[f'configs/{NAME}.json','planning_C05_C06_EXECUTION_KO.md']
    for task in ['C05','C06']:update_status(task,'in_progress',evidence,
        'Frozen autonomous 61-step prediction and same-past opposite-action evaluation executing. Independent replay and reports remain required.')
    calibration=[calibrate_proxy(g) for g in c['groups']]
    dump(HERE/f'reports/{NAME}/contact_proxy_calibration/summary.json',{'groups':calibration})
    expected=len(METHODS)*3*len(c['groups'])*len(PREDICTORS)
    for method in METHODS:
        for seed in c['seeds']:
            models={}
            for kind in LEARNED:
                ck=torch.load(HERE/f'runs/dynamics_state_study_v1/variable_force/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
                model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();models[kind]=model
            for group in c['groups']:
                mode,split=group['mode'],group['split'];root=HERE/f'runs/{NAME}/{method}_s{seed}/{mode}/{split}'
                inputs=initial_inputs(method,seed,mode,split);root.mkdir(parents=True,exist_ok=True)
                ip=root/'initial_inputs.npz'
                if ip.exists():
                    with np.load(ip) as a:
                        for k,v in inputs.items():np.testing.assert_array_equal(a[k],v)
                else:np.savez_compressed(ip,**inputs)
                with np.load(HERE/group['path']) as a:
                    truth=canonical_sequence(a['states'][:,3:]);events=a['events'][:,3:].copy()
                    cf_truth=canonical_sequence(a['counterfactual_states'][:,3:]) if 'counterfactual_states' in a else None
                    cf_events=a['counterfactual_events'][:,3:].copy() if cf_truth is not None else None
                    if cf_truth is not None:np.testing.assert_array_equal(cf_truth[:,0],truth[:,0])
                for kind in PREDICTORS:
                    folder=root/kind
                    if (folder/'metrics.json').exists():
                        m=json.loads((folder/'metrics.json').read_text())
                        for name,sha in m['files'].items():assert r.sha(folder/name)==sha
                        results.append(m);continue
                    fresh_dir(folder)
                    base=autonomous(inputs['initial'],inputs['context'],inputs['action'],kind,models.get(kind))
                    first_error=check_first_step(base,inputs,method,seed,mode,split,kind)
                    keys,values,ev,valid=score_rollout(base,inputs['initial'],inputs['order'],inputs['context'],inputs['action'],
                                                       truth,events,inputs['found'],inputs['overflow'])
                    archive={**base,'predicted_events':ev,'event_valid':valid,'metric_values':values,'metric_keys':np.array(keys)}
                    counterfactual=None;cf_summary=None
                    if cf_truth is not None:
                        cf=autonomous(inputs['initial'],inputs['context'],-inputs['action'],kind,models.get(kind))
                        ck,cv,ce,cevalid=score_rollout(cf,inputs['initial'],inputs['order'],inputs['context'],-inputs['action'],
                                                    cf_truth,cf_events,inputs['found'],inputs['overflow'])
                        response_keys,response_values=score_counterfactual(base,cf,inputs['initial'],inputs['order'],truth,cf_truth,inputs['target_colors'])
                        for key,value in cf.items():archive['cf_'+key]=value
                        archive.update(cf_metric_values=cv,cf_predicted_events=ce,cf_event_valid=cevalid,
                                       response_metric_keys=np.array(response_keys),response_metric_values=response_values)
                        counterfactual=summarize(response_keys,response_values);cf_summary=summarize(ck,cv)
                    np.savez_compressed(folder/'trajectories.npz',**archive)
                    metric={'method':method,'seed':seed,'mode':mode,'split':split,'predictor':kind,'episodes':len(truth),
                        'factual':summarize(keys,values),'counterfactual':cf_summary,'response':counterfactual,
                        'first_step_C04_max_abs_error':first_error,'nonfinite_episodes':int((base['first_nonfinite']>=0).sum()),
                        'input_sha256':r.sha(ip),'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
                        'files':{'trajectories.npz':r.sha(folder/'trajectories.npz')}}
                    dump(folder/'metrics.json',metric);results.append(metric)
                progress={'completed_conditions':len(results),'expected_conditions':expected,'last':[method,seed,mode,split],
                          'elapsed_seconds':time.time()-started}
                dump(HERE/f'reports/{NAME}/progress.json',progress);print(json.dumps(progress),flush=True)
    dump(HERE/f'reports/{NAME}/summary.json',{'results':results,'conditions':len(results),'expected_conditions':expected,
        'base_episodes':1280,'counterfactual_pairs':512,'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'scope':c['information_boundary'],'seconds':time.time()-started})


if __name__=='__main__':main()
