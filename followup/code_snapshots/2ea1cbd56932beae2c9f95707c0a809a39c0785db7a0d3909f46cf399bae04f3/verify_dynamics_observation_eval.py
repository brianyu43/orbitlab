"""Replay observation models, public color actions and every substitution forecast."""
import csv
import json
import numpy as np
import torch
from common import HERE,dump,r
from dynamics_observation_eval import NAME,freeze
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_observation_models import ObservationEstimator,load_inputs,decode
from train_dynamics_observation import NAME as MODELS
from dynamics_rgb_baseline import NAME as BASELINE
from dynamics_models import Transition,encode_states


def color_address(raw):
    out=np.zeros((len(raw),6,7))
    for i,scene in enumerate(raw):
        for obj in scene:
            if obj[4]>0:out[i,int(obj[5])]=obj
    return out


def independent_pack(states,scores,colors,delta):
    order=np.sort(np.argsort(-scores,axis=1,kind='stable')[:,:4],axis=1);packed=np.take_along_axis(states,order[:,:,None],axis=1)
    action=np.zeros((*order.shape,2));found=[];discarded=[]
    for i in range(len(order)):
        hit=(order[i]==colors[i])&(packed[i,:,6]>.5);action[i,hit]=delta[i];found.append(bool(hit.any()))
        discarded.append(int((states[i,:,6]>.5).sum()-(packed[i,:,6]>.5).sum()))
    return packed,order,action,np.array(found),np.array(discarded)


def physical_reference(state,context,action,kind):
    xy=state[:,:,:2].copy();v=state[:,:,2:4].copy()+action;live=state[:,:,6]>.5
    if kind=='inertial':xy+=v
    else:
        drag=context[:,2];decay=np.exp(-drag/8);increment=np.full_like(drag,1/8)
        np.divide(-np.expm1(-drag/8),drag,out=increment,where=drag!=0)
        for _ in range(8):
            v=decay[:,None,None]*v+increment[:,None,None]*context[:,None,:2];xy+=v/8
            limit=31.5-state[:,:,4]
            for axis in range(2):
                outside=(np.abs(xy[:,:,axis])>limit)&live;sign=np.sign(xy[:,:,axis])
                xy[:,:,axis]=np.where(outside,sign*(2*limit-np.abs(xy[:,:,axis])),xy[:,:,axis])
                v[:,:,axis]=np.where(outside&(v[:,:,axis]*sign>0),-v[:,:,axis],v[:,:,axis])
    return np.concatenate([xy,v],-1)*live[:,:,None]


def check_rows(rows,state,context,truth,true_context,events,information=None,predictor=None,found=None,discarded=None):
    for i,row in enumerate(rows):
        s,t=state[i],truth[i];live=s[:,6]>.5;gt=t[:,6]>.5;both=live&gt;error=abs(s[:,:5]-t[:,:5])
        values={'episode':i,'past_contact':bool(events[i].any()),'past_pair_contact':bool(events[i,:,0].any()),'past_wall_contact':bool(events[i,:,1].any()),
            'true_count':int(gt.sum()),'predicted_count':int(live.sum()),'count_correct':bool(live.sum()==gt.sum()),'all_presence_correct':bool(np.array_equal(live,gt)),
            'recall':float(both.sum()/gt.sum()),'false_positive_colors':int((live&~gt).sum()),
            'matched_position_mae_pixels':float(error[both,:2].mean()) if both.any() else None,
            'matched_velocity_mae_pixels_per_frame':float(error[both,2:4].mean()) if both.any() else None,
            'matched_radius_mae_pixels':float(error[both,4].mean()) if both.any() else None,
            'zero_filled_true_slot_position_mae':float(error[gt,:2].mean()),'zero_filled_true_slot_velocity_mae':float(error[gt,2:4].mean()),
            'state_scene_success':bool(np.array_equal(live,gt) and (error[gt,:2]<=1).all() and (error[gt,2:4]<=.1).all() and (error[gt,4]<=.5).all()),
            'net_force_mae_pixels_per_frame_squared':float(abs(context[i,:2]-true_context[i,:2]).mean()),'drag_absolute_error':float(abs(context[i,2]-true_context[i,2]))}
        if information is not None:values.update(information=information,predictor=predictor,command_target_present=bool(found[i]),discarded_present_colors=int(discarded[i]))
        assert set(row)==set(values)
        for key,v in values.items():
            if v is None:assert row[key]==''
            elif isinstance(v,bool):assert row[key]==str(v),(key,row[key],v)
            elif isinstance(v,str):assert row[key]==v
            else:assert abs(float(row[key])-v)<1e-9,(key,row[key],v)


def check_aggregates(rows,summary):
    count=0
    for group,info in summary.items():
        rr=[v for v in rows if group=='all' or (v['past_contact']=='False' if group=='no_past_contact' else v[group]=='True')]
        assert len(rr)==info['episodes']
        for key,stat in info['metrics'].items():
            values=[v[key] for v in rr if v[key]!='']
            if not values:assert stat is None;continue
            arr=np.array([float(v=='True') if v in ['True','False'] else float(v) for v in values]);rng=np.random.default_rng(9123)
            boot=arr[rng.integers(len(arr),size=(1000,len(arr)))].mean(1)
            assert abs(arr.mean()-stat['mean'])<1e-10;np.testing.assert_allclose(np.quantile(boot,[.025,.975]),stat['base_scene_ci95'],atol=1e-10,rtol=0);count+=1
    return count


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(2);root=HERE/f'reports/{NAME}';study=json.loads((root/'summary.json').read_text());assert study['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    for p in [HERE/f'reports/{INPUTS}/verification.json',HERE/f'reports/{BASELINE}/verification.json',HERE/f'reports/{MODELS}/development_verification.json']:
        assert json.loads(p.read_text())['all_passed']
    baseline_manifest=json.loads((HERE/f'data/{BASELINE}/manifest.json').read_text());baseline_report=json.loads((HERE/f'reports/{BASELINE}/summary.json').read_text())
    estimator_cache={};transition_cache={};oracle_cache={};inference_count=forecast_count=checks=0
    assert len(study['results'])==156
    for entry in study['results']:
        method,seed,mode,split=(entry[k] for k in ['method','seed','mode','split']);folder=HERE/f'runs/{NAME}/{method}_s{seed}/{mode}/{split}'
        assert entry==json.loads((folder/'metrics.json').read_text()) and entry['config_sha256']==study['config_sha256']
        for file,sha in entry['files'].items():assert sha==r.sha(folder/file)
        with np.load(folder/'predictions.npz') as a:saved={k:a[k].copy() for k in a.files}
        if method in ['rgb_cnn','measurement_mlp']:
            key=(method,seed)
            if key not in estimator_cache:
                ck=torch.load(HERE/f'runs/{MODELS}/{method}_s{seed}/model.pt',map_location='cpu',weights_only=True);model=ObservationEstimator(method);model.load_state_dict(ck['state_dict']);model.eval();estimator_cache[key]=model
            inputs=load_inputs(mode,split,method);out=[]
            for i in range(0,len(next(iter(inputs.values()))),32):out.append(estimator_cache[key]({k:v[i:i+32] for k,v in inputs.items()}))
            raw={k:torch.cat([p[k] for p in out]) for k in ['state','context']};stored=torch.load(folder/'raw_estimator.pt',map_location='cpu',weights_only=True)
            for k in raw:torch.testing.assert_close(raw[k],stored[k],atol=0,rtol=0)
            state,context=decode(raw);state=state.numpy().astype(np.float64);context=context.numpy().astype(np.float64);score=raw['state'][...,5].numpy()
        else:
            source=next(v for v in baseline_manifest['groups'] if (v['mode'],v['split'])==(mode,split))
            if method=='rgb_analytic':
                path=HERE/f'data/{BASELINE}/{mode}/{split}/predictions.npz';assert r.sha(path)==source['prediction_sha256']
                with np.load(path) as a:state=a['states'].copy();context=a['contexts'].copy()
            else:
                path=HERE/f'reports/{BASELINE}/{mode}/{split}/controls.npz';rr=next(v for v in baseline_report['groups'] if (v['mode'],v['split'])==(mode,split))
                assert r.sha(path)==rr['files']['controls.npz']
                with np.load(path) as a:state=a[method+'_states'].copy();context=a[method+'_contexts'].copy()
            score=state[:,:,6]
        for key,value in [('estimated_states',state),('estimated_contexts',context),('presence_scores',score)]:np.testing.assert_array_equal(value,saved[key])
        data=HERE/f'data/{INPUTS}/{mode}/{split}'
        with np.load(data/'labels.npz') as a:truth=color_address(a['past_states'][:,-1]);target=color_address(a['next_states']);true_context=np.column_stack([a['net_acceleration'],a['drag']]);events=a['past_events'].copy()
        with np.load(data/'commands.npz') as a:colors=a['target_color'].copy();delta=a['delta_velocity'].copy()
        rows=list(csv.DictReader((folder/'inference_rows.csv').open()));check_rows(rows,state,context,truth,true_context,events);checks+=check_aggregates(rows,entry['inference']);inference_count+=len(rows)
        forecast_rows=list(csv.DictReader((folder/'forecast_rows.csv').open()))
        for information in c['information_controls']:
            ss=truth if information.startswith('true_state_') else state;ctx=true_context if information.endswith('_true_context') else context
            sc=ss[:,:,6] if information.startswith('true_state_') else score;packed,order,actions,found,discarded=independent_pack(ss,sc,colors,delta)
            for kind in c['predictors']:
                if kind in ['inertial','force_wall']:motion=physical_reference(packed,ctx,actions,kind)
                else:
                    k=(kind,seed)
                    if k not in transition_cache:
                        ck=torch.load(HERE/f'runs/dynamics_state_study_v1/variable_force/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
                        model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();transition_cache[k]=model
                    s=encode_states(packed);action=torch.tensor(actions,dtype=torch.float32)/3;cctx=torch.tensor(ctx,dtype=torch.float32)/torch.tensor([.1,.1,.01])
                    motion=(transition_cache[k](s,action,cctx)*torch.tensor([31.5,31.5,3.,3.])).numpy()
                output=np.zeros_like(state)
                for i in range(len(state)):
                    for j,co in enumerate(order[i]):
                        if packed[i,j,6]>.5:output[i,co]=[*motion[i,j],*packed[i,j,4:]]
                key=information+'__'+kind;np.testing.assert_allclose(output,saved[key],atol=1e-10,rtol=0)
                selected=[v for v in forecast_rows if v['information']==information and v['predictor']==kind];assert len(selected)==len(state)
                # Stored numerical output remains the scoring source; reference
                # replay may differ by final floating-point operation ordering.
                check_rows(selected,saved[key],ctx,target,true_context,events,information,kind,found,discarded)
                checks+=check_aggregates(selected,entry['forecasts'][key]);forecast_count+=len(selected)
                if information=='true_state_true_context':
                    unique=(seed,mode,split,kind)
                    if unique in oracle_cache:np.testing.assert_array_equal(saved[key],oracle_cache[unique])
                    else:oracle_cache[unique]=saved[key]
        print('verified observation substitutions',method,seed,mode,split,flush=True)
    dump(root/'verification.json',{'all_passed':True,'conditions_verified':len(study['results']),'inference_rows_replayed':inference_count,
        'one_step_forecasts_replayed_and_scored':forecast_count,'aggregate_and_interval_checks':checks,
        'true_state_true_context_controls_identical_across_estimators':True,'public_color_action_routing_verified':True,
        'verifier_sha256':r.sha(__file__),'baseline_summary_sha256':r.sha(HERE/f'reports/{BASELINE}/summary.json'),
        'scope':study['scope']})


if __name__=='__main__':main()
