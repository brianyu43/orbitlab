"""Independent recurrence/input/primary-metric checks plus full score replay."""
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import HERE,dump,r
from dynamics_models import Transition
from dynamics_autonomous import HORIZONS, LEARNED
from dynamics_autonomous_metrics import score_rollout,score_counterfactual,summarize
from dynamics_autonomous_study import NAME,freeze


def canonical(raw):
    n,t=raw.shape[:2];out=np.zeros((n,t,6,7))
    for episode in range(n):
        for slot in range(4):
            if raw[episode,0,slot,6]>.5:
                out[episode,:,int(raw[episode,0,slot,5])]=raw[episode,:,slot]
    return out


def independent_inputs(method,seed,mode,split):
    source=HERE/f'data/dynamics_observation_v1/{mode}/{split}'
    with np.load(source/'commands.npz') as a:colors=a['target_color'].copy();dv=a['delta_velocity'].copy()
    if method=='oracle':
        with np.load(source/'labels.npz') as a:
            state=canonical(a['past_states'][:,3:4])[:,0]
            ctx=np.c_[a['net_acceleration'],a['drag']]
        scores=state[:,:,6]
    else:
        with np.load(HERE/f'runs/dynamics_observation_eval_v1/{method}_s{seed}/{mode}/{split}/predictions.npz') as a:
            state=a['estimated_states'].copy();ctx=a['estimated_contexts'].copy();scores=a['presence_scores'].copy()
    order=np.sort(np.argsort(-scores,axis=1,kind='stable')[:,:4],axis=1)
    initial=np.take_along_axis(state,order[:,:,None],axis=1)
    selection=(order==colors[:,None])&(initial[:,:,6]>.5)
    action=selection[:,:,None]*dv[:,None]
    return {'initial':initial,'context':ctx,'order':order,'action':action,'target_colors':colors,
            'found':selection.any(1),'overflow':(state[:,:,6]>.5).sum(1)-(initial[:,:,6]>.5).sum(1)}


@torch.no_grad()
def replay(initial,context,action,kind,model):
    n=len(initial);out=np.full((n,62,4,4),np.nan);out[:,0]=initial[:,:,:4]
    finite=np.ones((n,62),bool);first=np.full(n,-1,dtype=np.int16)
    good=np.ones(n,bool)
    if kind in LEARNED:
        raw=torch.tensor(initial,dtype=torch.float32);live=(raw[:,:,4]>0).float()
        s=torch.cat([raw[:,:,:5]/torch.tensor([31.5,31.5,3,3,8]),
                     F.one_hot(raw[:,:,5].long().clamp(0,5),6)*live[:,:,None],live[:,:,None]],dim=-1)
        ctx=torch.tensor(context,dtype=torch.float32)/torch.tensor([.1,.1,.01])
        a=torch.tensor(action,dtype=torch.float32)/3
    else:
        pos=initial[:,:,:2].copy();vel=initial[:,:,2:4].copy();live=initial[:,:,4]>0
        drag=context[:,2];d=np.exp(-drag/8);g=np.full(n,1/8.)
        np.divide(1-d,drag,out=g,where=drag!=0)
        # Use expm1 for the reference's small-drag precision, derived separately.
        g=np.array([1/8 if gamma==0 else -np.expm1(-gamma/8)/gamma for gamma in drag])
    for t in range(1,62):
        if kind in LEARNED:
            proposed=model(s,a if t==1 else torch.zeros_like(a),ctx)
            value=(proposed*torch.tensor([31.5,31.5,3,3])).numpy()
            ok=np.isfinite(value).all((1,2))&good
            s=s.clone();s[:,:,:4]=proposed;s[~torch.from_numpy(ok)]=0
        else:
            if t==1:vel=vel+action
            if kind=='inertial':pos=pos+vel
            else:
                for substep in range(8):
                    vel=d[:,None,None]*vel+g[:,None,None]*context[:,None,:2]
                    pos=pos+vel/8
                    for axis in [0,1]:
                        x=pos[:,:,axis];v=vel[:,:,axis];bound=31.5-initial[:,:,4]
                        high=(x>bound)&live;low=(x < -bound)&live
                        reflect=(high&(v>0))|(low&(v<0))
                        x[high]=2*bound[high]-x[high]
                        x[low]=-2*bound[low]-x[low]
                        v[reflect]*=-1
            value=np.concatenate([pos,vel],axis=-1)*live[:,:,None]
            ok=np.isfinite(value).all((1,2))&good
            pos[~ok]=0;vel[~ok]=0
        first[good&~ok]=t;good=ok;finite[:,t]=ok;out[ok,t]=value[ok]
    return {'motion':out,'finite':finite,'first_nonfinite':first}


def unpack(result,inputs):
    values=np.zeros((len(inputs['initial']),62,6,4));present=np.zeros((len(values),6),bool);radius=np.zeros_like(present,dtype=float)
    for i,(ss,order) in enumerate(zip(inputs['initial'],inputs['order'])):
        for slot,c in enumerate(order):
            if ss[slot,6]>.5:
                values[i,:,c]=result['motion'][i,:,slot];present[i,c]=True;radius[i,c]=ss[slot,4]
    return values,present,radius


def independent_contact(motion,radius,presence,context,action,finite):
    n,t,slots=motion.shape[:3];flags=np.zeros((n,t-1,2),bool)
    valid=finite[:,:-1]&finite[:,1:]
    for k in range(t-1):
        before=motion[:,k];after=motion[:,k+1]
        v=before[:,:,2:].copy()
        if k==0:v+=action
        gamma=context[:,2];decay=np.exp(-gamma);gain=np.ones(n)
        np.divide(-np.expm1(-gamma),gamma,out=gain,where=gamma!=0)
        expected=v*decay[:,None,None]+gain[:,None,None]*context[:,None,:2]
        impulse=np.sqrt(np.sum((after[:,:,2:]-expected)**2,axis=-1))>.1
        for i in range(slots):
            bound=31.5-radius[:,i]-1.5
            close=np.any((np.abs(before[:,i,:2])>=bound[:,None])|(np.abs(after[:,i,:2])>=bound[:,None]),axis=1)
            flags[:,k,1]|=close&impulse[:,i]&presence[:,i]
            for j in range(i+1,slots):
                x=before[:,i,:2]-before[:,j,:2];y=after[:,i,:2]-after[:,j,:2]
                path=y-x;den=np.einsum('ij,ij->i',path,path);num=-np.einsum('ij,ij->i',x,path)
                u=np.divide(num,den,out=np.zeros(n),where=den!=0);u=np.minimum(1,np.maximum(0,u))
                nearest=x+u[:,None]*path
                near=np.sqrt(np.einsum('ij,ij->i',nearest,nearest))<=radius[:,i]+radius[:,j]+1.5
                flags[:,k,0]|=near&(impulse[:,i]|impulse[:,j])&presence[:,i]&presence[:,j]
    return flags&valid[:,:,None],valid


def independent_primary(result,inputs,truth,keys,values):
    pred,present,radius=unpack(result,inputs);look={k:i for i,k in enumerate(keys)};checks=0
    for i in range(len(pred)):
        colors=np.flatnonzero(truth[i,0,:,6]>.5)
        present_colors=np.flatnonzero(present[i]);same=np.array_equal(colors,present_colors)
        matched=np.intersect1d(colors,present_colors)
        for h_index,h in enumerate(HORIZONS):
            finite=bool(result['finite'][i,h]);actual=truth[i,h,colors]
            errors=np.abs(pred[i,h,colors]-actual[:,:4])
            expected={
                'finite_fraction':float(finite),'all_presence_correct':float(same),
                'recall':len(matched)/len(colors),'command_target_present':float(inputs['found'][i]),
                'discarded_present_colors':float(inputs['overflow'][i]),
                'position_mae_pixels_finite':float(errors[:,:2].mean()) if finite else np.nan,
                'velocity_mae_pixels_per_frame_finite':float(errors[:,2:].mean()) if finite else np.nan,
                'scene_success':float(finite and same and np.all(errors[:,:2]<=1) and np.all(errors[:,2:]<=.1)
                                      and np.all(np.abs(radius[i,colors]-actual[:,4])<=.5))}
            correct=0
            if finite:
                for c in matched:
                    distances=[float(np.sum((pred[i,h,c,:2]-truth[i,h,j,:2])**2)) for j in colors]
                    correct+=colors[int(np.argmin(distances))]==c
            expected['own_color_nearest_position_fraction']=correct/len(colors)
            for name,value in expected.items():
                np.testing.assert_allclose(values[i,h_index,look[name]],value,atol=1e-8,rtol=1e-10,equal_nan=True);checks+=1
    return checks


def independent_response(base,cf,inputs,truth,cf_truth,keys,values):
    p,presence,_=unpack(base,inputs);q,_,_=unpack(cf,inputs);look={k:i for i,k in enumerate(keys)};checks=0
    for i in range(len(p)):
        colors=np.flatnonzero(truth[i,0,:,6]>.5);target=int(inputs['target_colors'][i])
        for ih,h in enumerate(HORIZONS):
            good=bool(base['finite'][i,h] and cf['finite'][i,h])
            # Difference of forecast errors equals error in counterfactual response.
            # fsum avoids losing small true responses when two enormous finite
            # forecasts cancel; the ordinary difference-of-errors is unstable.
            residual=np.array([[math.fsum([float(q[i,h,c,f]),-float(p[i,h,c,f]),
                float(truth[i,h,c,f]),-float(cf_truth[i,h,c,f])]) for f in range(4)] for c in range(6)])
            true_effect=cf_truth[i,h,:,:4]-truth[i,h,:,:4]
            for label,selected in [('target',[target]),('nontarget',[v for v in colors if v!=target])]:
                for feature,sl in [('position',slice(0,2)),('velocity',slice(2,4))]:
                    expected=float(np.abs(residual[selected,sl]).mean()) if good else np.nan
                    np.testing.assert_allclose(values[i,ih,look[label+'_'+feature+'_response_mae_finite']],expected,atol=1e-8,rtol=1e-10,equal_nan=True)
                    null=float(np.abs(true_effect[selected,sl]).mean())
                    np.testing.assert_allclose(values[i,ih,look[label+'_'+feature+'_zero_response_mae']],null,atol=1e-10,rtol=1e-10);checks+=2
    return checks


@torch.no_grad()
def main():
    torch.set_num_threads(2);c=freeze();started=time.time()
    report=HERE/f'reports/{NAME}';summary=json.loads((report/'summary.json').read_text())
    expected={(m,s,g['mode'],g['split'],k) for m in c['methods'] for s in c['seeds'] for g in c['groups'] for k in c['predictors']}
    listed=[(v['method'],v['seed'],v['mode'],v['split'],v['predictor']) for v in summary['results']]
    assert len(listed)==len(set(listed))==len(expected) and set(listed)==expected
    count=frames=cf_frames=primary=response=aggregate=inputs_checked=contact_frames=0
    for g in c['groups']:
        with np.load(HERE/g['path']) as a:
            raw=a['states'][:,3:].copy();events=a['events'][:,3:].copy();action=a['actions'][:,3].copy()
            context=np.c_[a['contexts'][:,:2]+a['contexts'][:,2:4],a['contexts'][:,4]]
        out=report/f"contact_proxy_calibration/{g['mode']}/{g['split']}"
        flags,valid=independent_contact(raw[:,:,:,:4],raw[:,0,:,4],raw[:,0,:,6]>.5,context,action,np.ones(raw.shape[:2],bool))
        cm=json.loads((out/'summary.json').read_text());assert cm['sha256']==r.sha(out/'predictions.npz')
        with np.load(out/'predictions.npz') as a:
            np.testing.assert_array_equal(flags,a['predicted']);np.testing.assert_array_equal(events,a['true_events'])
        for j,kind in enumerate(['pair','wall']):
            yes=events[:,:,j]>0;pred=flags[:,:,j]
            for key,value in [('tp',(yes&pred).sum()),('fp',(~yes&pred).sum()),('fn',(yes&~pred).sum())]:assert cm['metrics'][kind][key]==value
        contact_frames+=len(raw)*61
    models={}
    for seed in c['seeds']:
        for kind in LEARNED:
            ck=torch.load(HERE/f'runs/dynamics_state_study_v1/variable_force/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
            model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();models[seed,kind]=model
    for method in c['methods']:
        for seed in c['seeds']:
            for g in c['groups']:
                mode,split=g['mode'],g['split'];root=HERE/f'runs/{NAME}/{method}_s{seed}/{mode}/{split}'
                inputs=independent_inputs(method,seed,mode,split)
                with np.load(root/'initial_inputs.npz') as a:
                    assert set(a.files)==set(inputs)
                    for name,value in inputs.items():np.testing.assert_array_equal(a[name],value)
                inputs_checked+=len(inputs['initial'])
                with np.load(HERE/g['path']) as a:
                    truth=canonical(a['states'][:,3:]);events=a['events'][:,3:].copy()
                    cf_truth=canonical(a['counterfactual_states'][:,3:]) if 'counterfactual_states' in a else None
                    cf_events=a['counterfactual_events'][:,3:].copy() if cf_truth is not None else None
                for kind in c['predictors']:
                    folder=root/kind;m=json.loads((folder/'metrics.json').read_text())
                    assert m['input_sha256']==r.sha(root/'initial_inputs.npz') and m['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
                    assert m['files']['trajectories.npz']==r.sha(folder/'trajectories.npz')
                    with np.load(folder/'trajectories.npz') as a:stored={k:a[k].copy() for k in a.files}
                    model=models.get((seed,kind));base=None
                    for branch,label,event,sign in [('base',truth,events,1),('cf',cf_truth,cf_events,-1)]:
                        if label is None:continue
                        prefix='' if branch=='base' else 'cf_'
                        result=replay(inputs['initial'],inputs['context'],inputs['action']*sign,kind,model)
                        for key,value in result.items():np.testing.assert_allclose(stored[prefix+key],value,rtol=1e-10,atol=1e-8,equal_nan=True)
                        flags,valid=independent_contact(result['motion'],inputs['initial'][:,:,4],inputs['initial'][:,:,6]>.5,inputs['context'],sign*inputs['action'],result['finite'])
                        np.testing.assert_array_equal(stored[prefix+'predicted_events'],flags);np.testing.assert_array_equal(stored[prefix+'event_valid'],valid)
                        keys,values,_,_=score_rollout(result,inputs['initial'],inputs['order'],inputs['context'],sign*inputs['action'],label,event,inputs['found'],inputs['overflow'])
                        np.testing.assert_array_equal(stored['metric_keys'],keys)
                        np.testing.assert_allclose(stored[prefix+'metric_values'],values,atol=1e-8,rtol=1e-10,equal_nan=True)
                        # Aggregate replay uses the exact saved arrays to avoid irrelevant summation-order drift.
                        assert summarize(keys,stored[prefix+'metric_values'])==m['factual' if branch=='base' else 'counterfactual']
                        aggregate+=len(keys)*len(HORIZONS)
                        primary+=independent_primary(result,inputs,label,keys,stored[prefix+'metric_values'])
                        frames+=len(label)*61 if branch=='base' else 0
                        cf_frames+=len(label)*61 if branch=='cf' else 0
                        if branch=='base':base=result
                        else:
                            keys,values=score_counterfactual(base,result,inputs['initial'],inputs['order'],truth,cf_truth,inputs['target_colors'])
                            np.testing.assert_array_equal(keys,stored['response_metric_keys'])
                            np.testing.assert_allclose(values,stored['response_metric_values'],atol=1e-8,rtol=1e-10,equal_nan=True)
                            assert summarize(keys,stored['response_metric_values'])==m['response'];aggregate+=len(keys)*len(HORIZONS)
                            response+=independent_response(base,result,inputs,truth,cf_truth,keys,stored['response_metric_values'])
                    assert m['nonfinite_episodes']==int((base['first_nonfinite']>=0).sum())
                    count+=1
                dump(report/'verification_progress.json',{'verified_conditions':count,'expected_conditions':len(expected),'elapsed_seconds':time.time()-started})
                print('verified autonomous',method,seed,mode,split,count,flush=True)
    # Full summary is an exact index of verified per-condition outputs.
    for item in summary['results']:
        path=HERE/f"runs/{NAME}/{item['method']}_s{item['seed']}/{item['mode']}/{item['split']}/{item['predictor']}/metrics.json"
        assert item==json.loads(path.read_text())
    dump(report/'verification.json',{'all_passed':True,'conditions_verified':count,'initial_input_episodes_checked':inputs_checked,
        'factual_forecast_frames_replayed':frames,'counterfactual_forecast_frames_replayed':cf_frames,
        'independent_primary_metric_checks':primary,'independent_response_metric_checks':response,
        'aggregate_metric_checks':aggregate,'true_trajectory_contact_proxy_frames_checked':contact_frames,
        'input_packing_and_public_color_actions_independently_verified':True,
        'all_recurrences_replayed_without_future_labels':True,'summary_sha256':r.sha(report/'summary.json'),
        'verifier_sha256':r.sha(__file__),'seconds':time.time()-started,
        'scope':'All saved recurrent predictions and derived contact flags independently replayed. Primary endpoint/response metrics independently recomputed; all remaining metrics replayed with locked scoring code. One original data seed; neither scientific generality nor entire goal completion.'})


if __name__=='__main__':main()
