"""Held-out four-frame inference and state/context substitution controls."""
import json
import time
import numpy as np
import torch
from common import HERE,dump,fresh_dir,r
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_rgb_baseline import NAME as BASELINE
from dynamics_observation_models import ObservationEstimator,load_inputs,decode
from train_dynamics_observation import NAME as MODELS
from evaluate_dynamics_rgb_baseline import canonical,state_context_rows,aggregate
from dynamics_models import Transition,encode_states
from dynamics_physical_baselines import predict as physical_predict

NAME='dynamics_observation_eval_v1'
METHODS=['rgb_cnn','measurement_mlp','rgb_analytic','true_positions_analytic']
PREDICTORS=['interaction','wrong_exact','joint_exact','inertial','force_wall']
INFORMATION=['true_state_true_context','estimated_state_true_context','true_state_estimated_context','estimated_state_estimated_context']


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        files={}
        for seed in range(3):
            for kind in ['rgb_cnn','measurement_mlp']:
                p=f'runs/{MODELS}/{kind}_s{seed}/model.pt';files[p]=r.sha(HERE/p)
            for kind in PREDICTORS[:3]:
                p=f'runs/dynamics_state_study_v1/variable_force/{kind}_s{seed}/model.pt';files[p]=r.sha(HERE/p)
        dump(path,{'methods':METHODS,'predictors':PREDICTORS,'information_controls':INFORMATION,'seeds':[0,1,2],
            'checkpoint_sha256':files,'input_manifest_sha256':r.sha(HERE/f'data/{INPUTS}/manifest.json'),
            'sources':{p:r.sha(HERE/p) for p in ['dynamics_observation_eval.py','dynamics_observation_models.py','evaluate_dynamics_rgb_baseline.py','dynamics_models.py','dynamics_physical_baselines.py']},
            'data':'All 13 held-out mode/split groups; only t=0..3 observations and the color-addressed impulse feed inference/prediction. No train/val selection here.',
            'transition_choice':'The previously frozen VARIABLE_FORCE interaction, state-only symmetry and joint symmetry models, same seed as each learned observation head. The same transition weights are used for every environment; actual mode is not an input or model-selection oracle.',
            'analytic_references':'Inertial and known-force/drag/wall integration, without disk contacts. Analytic estimators and references are not learned. Estimator/reference copies across transition seeds are correlated, not independent training replicates.',
            'packing':'Top four predicted presence logits, stable tie break by canonical color; restore color order. Analytic score is detected presence. No true count used. Ground-truth-state substitution is explicitly privileged.',
            'action':'Route the given color/delta command only to a present retained slot of that color; missed target is recorded. No GT ID routing in estimated states.',
            'output':'Carry inferred/true radius, color and presence; transition predicts position and velocity. Evaluate by persistent observable color address, including missed/extra colors; no post-hoc nearest-position rematching.',
            'scope':'One initial four-frame window and one subsequent prediction per held-out trajectory. State/context substitution controls, not long autonomous rollout or new data-generation-seed confirmation.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for key in ['checkpoint_sha256','sources']:
        for p,sha in c[key].items():assert r.sha(HERE/p)==sha,p
    assert c['input_manifest_sha256']==r.sha(HERE/f'data/{INPUTS}/manifest.json')
    return c


@torch.no_grad()
def estimate(method,seed,mode,split):
    if method in ['rgb_cnn','measurement_mlp']:
        path=HERE/f'runs/{MODELS}/{method}_s{seed}/model.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
        model=ObservationEstimator(method);model.load_state_dict(ck['state_dict']);model.eval();inputs=load_inputs(mode,split,method);parts=[]
        for i in range(0,len(next(iter(inputs.values()))),32):parts.append(model({k:v[i:i+32] for k,v in inputs.items()}))
        raw={k:torch.cat([v[k] for v in parts]) for k in ['state','context']};states,contexts=decode(raw)
        return states.numpy().astype(np.float64),contexts.numpy().astype(np.float64),raw['state'][...,5].numpy(),raw
    if method=='rgb_analytic':
        with np.load(HERE/f'data/{BASELINE}/{mode}/{split}/predictions.npz') as a:s=a['states'].copy();ctx=a['contexts'].copy()
    elif method=='true_positions_analytic':
        with np.load(HERE/f'reports/{BASELINE}/{mode}/{split}/controls.npz') as a:s=a['true_positions_analytic_states'].copy();ctx=a['true_positions_analytic_contexts'].copy()
    else:raise ValueError(method)
    return s,ctx,s[:,:,6].copy(),{}


def pack(states,score,command_colors,impulses):
    batches=[];orders=[];actions=[];target_found=[];overflow=[]
    for s,p,color,dv in zip(states,score,command_colors,impulses):
        order=sorted(sorted(range(6),key=lambda j:(-float(p[j]),j))[:4]);packed=s[order].copy();action=np.zeros((4,2));found=False
        for i,k in enumerate(order):
            if k==color and packed[i,6]>.5:action[i]=dv;found=True
        batches.append(packed);orders.append(order);actions.append(action);target_found.append(found)
        overflow.append(int((s[:,6]>.5).sum()-(packed[:,6]>.5).sum()))
    return np.stack(batches),np.array(orders),np.stack(actions),np.array(target_found),np.array(overflow)


@torch.no_grad()
def predict(packed,order,context,action,kind,model=None):
    if kind in ['inertial','force_wall']:
        c=np.column_stack([context[:,:2],np.zeros((len(context),2)),context[:,2]])
        motion=physical_predict(packed,c,action,kind)
    else:
        s=encode_states(packed);a=torch.tensor(action,dtype=torch.float32)/3;ctx=torch.tensor(context,dtype=torch.float32)/torch.tensor([.1,.1,.01])
        motion=(model(s,a,ctx)*s.new_tensor([31.5,31.5,3,3])).numpy()
    if not np.isfinite(motion).all():raise FloatingPointError('Nonfinite initial one-step forecast')
    out=np.zeros((len(packed),6,7))
    for i in range(len(packed)):
        for j,color in enumerate(order[i]):
            if packed[i,j,6]>.5:out[i,color]=[*motion[i,j],*packed[i,j,4:]]
    return out


def forecast_stats(rows):
    keys=['all_presence_correct','recall','state_scene_success','zero_filled_true_slot_position_mae','zero_filled_true_slot_velocity_mae',
        'matched_position_mae_pixels','matched_velocity_mae_pixels_per_frame','command_target_present','discarded_present_colors']
    groups={'all':rows,'no_past_contact':[v for v in rows if not v['past_contact']],'past_contact':[v for v in rows if v['past_contact']]}
    return {name:{'episodes':len(rr),'metrics':{key:r.bootstrap([v[key] for v in rr if v[key] is not None]) if any(v[key] is not None for v in rr) else None for key in keys}} for name,rr in groups.items() if rr}


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(2)
    for p in [HERE/f'reports/{MODELS}/development_verification.json',HERE/f'reports/{BASELINE}/verification.json']:
        assert json.loads(p.read_text())['all_passed']
    manifest=json.loads((HERE/f'data/{INPUTS}/manifest.json').read_text());groups=[g for g in manifest['groups'] if g['split'] not in ['train','val']];results=[]
    for method in c['methods']:
        for seed in c['seeds']:
            models={}
            for kind in PREDICTORS[:3]:
                ck=torch.load(HERE/f'runs/dynamics_state_study_v1/variable_force/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
                model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();models[kind]=model
            for group in groups:
                mode,split=group['mode'],group['split'];folder=HERE/f'runs/{NAME}/{method}_s{seed}/{mode}/{split}'
                if (folder/'metrics.json').exists():results.append(json.loads((folder/'metrics.json').read_text()));continue
                fresh_dir(folder);state,ctx,score,raw=estimate(method,seed,mode,split)
                data=HERE/f'data/{INPUTS}/{mode}/{split}'
                with np.load(data/'labels.npz') as a:
                    truth=canonical(a['past_states'][:,-1]);target=canonical(a['next_states']);true_context=np.column_stack([a['net_acceleration'],a['drag']]);events=a['past_events'].copy()
                with np.load(data/'commands.npz') as a:colors=a['target_color'].copy();impulses=a['delta_velocity'].copy()
                inference_rows=state_context_rows(state,ctx,truth,true_context,events);r.write_csv(folder/'inference_rows.csv',inference_rows)
                predictions={'estimated_states':state,'estimated_contexts':ctx,'presence_scores':score};forecasts=[];forecast_metrics={}
                for info in INFORMATION:
                    ss=truth if info.startswith('true_state_') else state;cc=true_context if info.endswith('_true_context') else ctx
                    presence=ss[:,:,6] if info.startswith('true_state_') else score
                    packed,order,actions,found,overflow=pack(ss,presence,colors,impulses)
                    for kind in PREDICTORS:
                        pred=predict(packed,order,cc,actions,kind,models.get(kind));key=info+'__'+kind;predictions[key]=pred
                        rows=state_context_rows(pred,cc,target,true_context,events)
                        for i,row in enumerate(rows):row.update({'information':info,'predictor':kind,'command_target_present':bool(found[i]),'discarded_present_colors':int(overflow[i])})
                        forecasts.extend(rows);forecast_metrics[key]=forecast_stats(rows)
                np.savez_compressed(folder/'predictions.npz',**predictions);torch.save(raw,folder/'raw_estimator.pt');r.write_csv(folder/'forecast_rows.csv',forecasts)
                metrics={'method':method,'seed':seed,'mode':mode,'split':split,'episodes':len(state),'inference':aggregate(inference_rows),'forecasts':forecast_metrics,
                    'files':{p.name:r.sha(p) for p in sorted(folder.iterdir())},'config_sha256':r.sha(HERE/f'configs/{NAME}.json')}
                dump(folder/'metrics.json',metrics);results.append(metrics)
                print('heldout observation and substitution',method,seed,mode,split,len(state),flush=True)
            dump(HERE/f'reports/{NAME}/progress.json',{'completed_conditions':len(results),'expected_conditions':len(groups)*len(METHODS)*3})
    dump(HERE/f'reports/{NAME}/summary.json',{'results':results,'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'scope':c['scope']})


if __name__=='__main__':main()
