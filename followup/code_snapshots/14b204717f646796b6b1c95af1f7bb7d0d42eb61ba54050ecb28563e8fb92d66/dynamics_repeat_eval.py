"""Per-dataset C04-C06 repeats with explicit new-data/new-checkpoint routing."""
import argparse
import json
import time
import numpy as np
import torch
from common import HERE,dump,r
from dynamics_repeat_data import NAME,freeze as freeze_parent
from dynamics_repeat_observation import load_inputs
from dynamics_observation_models import ObservationEstimator,decode
from dynamics_models import Transition
from dynamics_observation_eval import pack,predict,forecast_stats,INFORMATION
from evaluate_dynamics_rgb_baseline import canonical,state_context_rows,aggregate
from dynamics_autonomous import autonomous,canonical_sequence,unpack_motion,contact_proxy,METHODS,PREDICTORS,LEARNED
from dynamics_autonomous_metrics import score_rollout,score_counterfactual,summarize

OBS_METHODS=['rgb_cnn','measurement_mlp','rgb_analytic','true_positions_analytic']
OBS_PREDICTORS=['interaction','wrong_exact','joint_exact','inertial','force_wall']


def config_name(ds):return f'dynamics_repeat_eval_d{ds}_v1'


def freeze(ds):
    parent=freeze_parent();assert ds in parent['new_data_seeds']
    gate=HERE/f'reports/{NAME}/observation_development_verification.json'
    assert json.loads(gate.read_text())['all_passed']
    # The other dataset may still be training; this dataset must have every one
    # of its 63 state-model audit receipts, not merely checkpoint files.
    checked={}
    for mode in parent['generator']['modes']:
        for seed in parent['initialization_seeds']:
            for kind in LEARNED:
                receipt=HERE/f'reports/{NAME}/state_verification/d{ds}_{mode}_{kind}_s{seed}.json'
                result=json.loads(receipt.read_text());assert result['all_passed']
                assert result['verifier_sha256']==r.sha(HERE/'verify_dynamics_repeat_state.py')
                for path,sha in result['artifacts'].items():assert r.sha(HERE/path)==sha,path
                checked[str(receipt.relative_to(HERE))]=r.sha(receipt)
    path=HERE/f'configs/{config_name(ds)}.json'
    if not path.exists():
        sources=['dynamics_repeat_eval.py','dynamics_repeat_observation.py','dynamics_observation_models.py',
            'dynamics_observation_eval.py','evaluate_dynamics_rgb_baseline.py','dynamics_models.py',
            'dynamics_physical_baselines.py','dynamics_autonomous.py','dynamics_autonomous_metrics.py']
        checkpoints={}
        for seed in range(3):
            for kind in LEARNED:
                p=f'runs/{NAME}/d{ds}/state/variable_force/{kind}_s{seed}/model.pt';checkpoints[p]=r.sha(HERE/p)
            for kind in ['rgb_cnn','measurement_mlp']:
                p=f'runs/{NAME}/d{ds}/observation/{kind}_s{seed}/model.pt';checkpoints[p]=r.sha(HERE/p)
        groups=[g for g in json.loads((HERE/f'data/{NAME}/d{ds}/world/manifest.json').read_text())['groups'] if g['split'] not in ['train','val']]
        dump(path,{'data_seed':ds,'groups':groups,'seeds':[0,1,2],'observation_methods':OBS_METHODS,'observation_predictors':OBS_PREDICTORS,
            'information_controls':INFORMATION,'autonomous_methods':METHODS,'autonomous_predictors':PREDICTORS,'steps':61,
            'checkpoint_sha256':checkpoints,'state_receipt_sha256':checked,'observation_development_verification_sha256':r.sha(gate),
            'parent_config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'sources':{p:r.sha(HERE/p) for p in sources},
            'scope':'Exact C04 information controls and C05/C06 autonomous/paired-action protocol on this independent data seed. Same variable_force transitions for all environments; no actual-mode model selection. Three fixed initialization seeds retained.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for field in ['checkpoint_sha256','state_receipt_sha256','sources']:
        for p,sha in c[field].items():assert r.sha(HERE/p)==sha,p
    assert c['parent_config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    return c


def models_for(ds,seed):
    models={}
    for kind in LEARNED:
        ck=torch.load(HERE/f'runs/{NAME}/d{ds}/state/variable_force/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
        model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();models[kind]=model
    return models


@torch.no_grad()
def estimate(ds,method,seed,mode,split):
    if method in ['rgb_cnn','measurement_mlp']:
        ck=torch.load(HERE/f'runs/{NAME}/d{ds}/observation/{method}_s{seed}/model.pt',map_location='cpu',weights_only=True)
        model=ObservationEstimator(method);model.load_state_dict(ck['state_dict']);model.eval();x=load_inputs(ds,mode,split,method)
        parts=[model({k:v[i:i+32] for k,v in x.items()}) for i in range(0,len(next(iter(x.values()))),32)]
        raw={k:torch.cat([v[k] for v in parts]) for k in ['state','context']};s,ctx=decode(raw)
        return s.numpy().astype(np.float64),ctx.numpy().astype(np.float64),raw['state'][...,5].numpy(),raw
    if method=='rgb_analytic':
        with np.load(HERE/f'data/{NAME}/d{ds}/measurements/{mode}/{split}/predictions.npz') as a:s=a['states'].copy();ctx=a['contexts'].copy()
    else:
        assert method=='true_positions_analytic'
        with np.load(HERE/f'reports/{NAME}/d{ds}/rgb_baseline/{mode}/{split}/controls.npz') as a:s=a['true_positions_analytic_states'].copy();ctx=a['true_positions_analytic_contexts'].copy()
    return s,ctx,s[:,:,6].copy(),{}


def labels_commands(ds,mode,split):
    folder=HERE/f'data/{NAME}/d{ds}/inputs/{mode}/{split}'
    with np.load(folder/'labels.npz') as a:labels={k:a[k].copy() for k in a.files}
    with np.load(folder/'commands.npz') as a:command={k:a[k].copy() for k in a.files}
    return labels,command


@torch.no_grad()
def observation(c):
    ds=c['data_seed'];results=[]
    for method in OBS_METHODS:
        for seed in c['seeds']:
            models=models_for(ds,seed)
            for g in c['groups']:
                mode,split=g['mode'],g['split'];folder=HERE/f'runs/{NAME}/d{ds}/observation_eval/{method}_s{seed}/{mode}/{split}'
                if (folder/'metrics.json').exists():
                    m=json.loads((folder/'metrics.json').read_text())
                    for p,sha in m['files'].items():assert r.sha(folder/p)==sha
                    results.append(m);continue
                folder.mkdir(parents=True,exist_ok=False);state,ctx,score,raw=estimate(ds,method,seed,mode,split)
                labels,command=labels_commands(ds,mode,split);truth=canonical(labels['past_states'][:,-1]);target=canonical(labels['next_states'])
                true_ctx=np.c_[labels['net_acceleration'],labels['drag']];events=labels['past_events']
                rows=state_context_rows(state,ctx,truth,true_ctx,events);r.write_csv(folder/'inference_rows.csv',rows)
                predictions={'estimated_states':state,'estimated_contexts':ctx,'presence_scores':score};forecasts=[];metrics={}
                for info in INFORMATION:
                    ss=truth if info.startswith('true_state_') else state;cc=true_ctx if info.endswith('_true_context') else ctx
                    presence=ss[:,:,6] if info.startswith('true_state_') else score
                    packed,order,action,found,overflow=pack(ss,presence,command['target_color'],command['delta_velocity'])
                    for kind in OBS_PREDICTORS:
                        key=info+'__'+kind;pred=predict(packed,order,cc,action,kind,models.get(kind));predictions[key]=pred
                        rr=state_context_rows(pred,cc,target,true_ctx,events)
                        for i,row in enumerate(rr):row.update(information=info,predictor=kind,command_target_present=bool(found[i]),discarded_present_colors=int(overflow[i]))
                        forecasts.extend(rr);metrics[key]=forecast_stats(rr)
                np.savez_compressed(folder/'predictions.npz',**predictions);torch.save(raw,folder/'raw_estimator.pt');r.write_csv(folder/'forecast_rows.csv',forecasts)
                m={'data_seed':ds,'method':method,'seed':seed,'mode':mode,'split':split,'episodes':len(state),
                   'inference':aggregate(rows),'forecasts':metrics,'files':{p.name:r.sha(p) for p in folder.iterdir() if p.is_file()},
                   'config_sha256':r.sha(HERE/f'configs/{config_name(ds)}.json')}
                dump(folder/'metrics.json',m);results.append(m)
                print('repeat observation eval',ds,method,seed,mode,split,flush=True)
                dump(HERE/f'reports/{NAME}/d{ds}/observation_eval_progress.json',{'completed_conditions':len(results),'expected_conditions':156})
    dump(HERE/f'reports/{NAME}/d{ds}/observation_eval_summary.json',{'results':results,'config_sha256':r.sha(HERE/f'configs/{config_name(ds)}.json'),'scope':c['scope']})


def autonomous_inputs(ds,method,seed,mode,split):
    labels,command=labels_commands(ds,mode,split)
    if method=='oracle':
        state=canonical(labels['past_states'][:,-1]);ctx=np.c_[labels['net_acceleration'],labels['drag']];score=state[:,:,6]
    else:
        with np.load(HERE/f'runs/{NAME}/d{ds}/observation_eval/{method}_s{seed}/{mode}/{split}/predictions.npz') as a:
            state=a['estimated_states'].copy();ctx=a['estimated_contexts'].copy();score=a['presence_scores'].copy()
    initial,order,action,found,overflow=pack(state,score,command['target_color'],command['delta_velocity'])
    return {'initial':initial,'order':order,'context':ctx,'action':action,'target_colors':command['target_color'],'found':found,'overflow':overflow}


@torch.no_grad()
def long_forecasts(c):
    ds=c['data_seed'];results=[];calibration=[]
    for g in c['groups']:
        mode,split=g['mode'],g['split'];path=HERE/f'reports/{NAME}/d{ds}/contact_proxy_calibration/{mode}/{split}.npz'
        with np.load(HERE/g['path']) as a:
            state=a['states'][:,3:].copy();ctx=np.c_[a['contexts'][:,:2]+a['contexts'][:,2:4],a['contexts'][:,4]]
            events=a['events'][:,3:].copy();actions=a['actions'][:,3].copy()
        flags,valid=contact_proxy(state[...,:4],state[:,0,:,4],state[:,0,:,6]>.5,ctx,actions,np.ones(state.shape[:2],bool))
        if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,predicted=flags,true_events=events,valid=valid)
        else:
            with np.load(path) as a:np.testing.assert_array_equal(a['predicted'],flags);np.testing.assert_array_equal(a['true_events'],events)
        calibration.append({'mode':mode,'split':split,'path':str(path.relative_to(HERE)),'sha256':r.sha(path)})
    dump(HERE/f'reports/{NAME}/d{ds}/contact_proxy_calibration/manifest.json',{'groups':calibration})
    for method in METHODS:
        for seed in c['seeds']:
            models=models_for(ds,seed)
            for g in c['groups']:
                mode,split=g['mode'],g['split'];inputs=autonomous_inputs(ds,method,seed,mode,split)
                root=HERE/f'runs/{NAME}/d{ds}/autonomous/{method}_s{seed}/{mode}/{split}';root.mkdir(parents=True,exist_ok=True)
                ip=root/'initial_inputs.npz'
                if not ip.exists():np.savez_compressed(ip,**inputs)
                else:
                    with np.load(ip) as a:
                        for k,value in inputs.items():np.testing.assert_array_equal(a[k],value)
                with np.load(HERE/g['path']) as a:
                    truth=canonical_sequence(a['states'][:,3:]);events=a['events'][:,3:].copy()
                    ct=canonical_sequence(a['counterfactual_states'][:,3:]) if 'counterfactual_states' in a else None
                    ce=a['counterfactual_events'][:,3:].copy() if ct is not None else None
                for kind in PREDICTORS:
                    folder=root/kind
                    if (folder/'metrics.json').exists():
                        m=json.loads((folder/'metrics.json').read_text())
                        for p,sha in m['files'].items():assert r.sha(folder/p)==sha
                        results.append(m);continue
                    folder.mkdir(parents=True,exist_ok=False)
                    base=autonomous(inputs['initial'],inputs['context'],inputs['action'],kind,models.get(kind))
                    first=None
                    if kind in OBS_PREDICTORS:
                        parent='rgb_cnn' if method=='oracle' else method;info='true_state_true_context' if method=='oracle' else 'estimated_state_estimated_context'
                        with np.load(HERE/f'runs/{NAME}/d{ds}/observation_eval/{parent}_s{seed}/{mode}/{split}/predictions.npz') as a:expected=a[info+'__'+kind][...,:4]
                        actual=unpack_motion(base['motion'][:,1:2],inputs['initial'],inputs['order'])[0][:,0]
                        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=1e-5);first=float(abs(actual-expected).max())
                    keys,values,ev,valid=score_rollout(base,inputs['initial'],inputs['order'],inputs['context'],inputs['action'],truth,events,inputs['found'],inputs['overflow'])
                    archive={**base,'metric_keys':np.array(keys),'metric_values':values,'predicted_events':ev,'event_valid':valid}
                    cf_summary=response=None
                    if ct is not None:
                        cf=autonomous(inputs['initial'],inputs['context'],-inputs['action'],kind,models.get(kind))
                        ck,cv,cev,cvalid=score_rollout(cf,inputs['initial'],inputs['order'],inputs['context'],-inputs['action'],ct,ce,inputs['found'],inputs['overflow'])
                        rk,rv=score_counterfactual(base,cf,inputs['initial'],inputs['order'],truth,ct,inputs['target_colors'])
                        archive.update({'cf_'+k:v for k,v in cf.items()});archive.update(cf_metric_values=cv,cf_predicted_events=cev,cf_event_valid=cvalid,response_metric_keys=np.array(rk),response_metric_values=rv)
                        cf_summary=summarize(ck,cv);response=summarize(rk,rv)
                    np.savez_compressed(folder/'trajectories.npz',**archive)
                    m={'data_seed':ds,'method':method,'seed':seed,'mode':mode,'split':split,'predictor':kind,'episodes':len(truth),
                        'factual':summarize(keys,values),'counterfactual':cf_summary,'response':response,'first_step_C04_max_abs_error':first,
                        'nonfinite_episodes':int((base['first_nonfinite']>=0).sum()),'input_sha256':r.sha(ip),
                        'config_sha256':r.sha(HERE/f'configs/{config_name(ds)}.json'),'files':{'trajectories.npz':r.sha(folder/'trajectories.npz')}}
                    dump(folder/'metrics.json',m);results.append(m)
                print('repeat autonomous eval',ds,method,seed,mode,split,len(results),flush=True)
                dump(HERE/f'reports/{NAME}/d{ds}/autonomous_progress.json',{'completed_conditions':len(results),'expected_conditions':1404})
    dump(HERE/f'reports/{NAME}/d{ds}/autonomous_summary.json',{'results':results,'conditions':len(results),'expected_conditions':1404,
        'base_episodes':1280,'counterfactual_pairs':512,'config_sha256':r.sha(HERE/f'configs/{config_name(ds)}.json'),'scope':c['scope']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-seed',type=int,required=True)
    parser.add_argument('--stage',choices=['observation','autonomous','all'],default='all');args=parser.parse_args()
    c=freeze(args.data_seed);torch.set_num_threads(2)
    if args.stage in ['observation','all']:observation(c)
    if args.stage in ['autonomous','all']:long_forecasts(c)


if __name__=='__main__':main()
