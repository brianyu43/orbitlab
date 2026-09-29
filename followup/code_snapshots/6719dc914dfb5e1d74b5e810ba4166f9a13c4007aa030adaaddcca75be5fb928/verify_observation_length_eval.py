"""Replay every length-specific inference and one-step information substitution."""
import argparse
import csv
import json
import numpy as np
import torch
from common import HERE,dump,r
from dynamics_observation_length_data import NAME
from dynamics_observation_length_eval import freeze,config_name,state_checkpoint
from dynamics_observation_length_models import load_inputs,LengthEstimator
from dynamics_observation_models import decode
from dynamics_models import Transition,encode_states
from verify_dynamics_observation_eval import color_address,independent_pack,physical_reference,check_rows,check_aggregates


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-seed',type=int,required=True);parser.add_argument('--length',type=int,choices=[1,2,4,8],required=True);args=parser.parse_args()
    ds=args.data_seed;length=args.length;c=freeze(ds,length);torch.set_num_threads(2);root=HERE/f'reports/{NAME}/d{ds}/L{length}'
    path=root/'observation_eval_summary.json';study=json.loads(path.read_text())
    assert study['config_sha256']==r.sha(HERE/f'configs/{config_name(ds,length)}.json')
    wanted={(m,s,g['mode'],g['split']) for m in c['observation_methods'] for s in c['seeds'] for g in c['groups']}
    listed=[(v['method'],v['seed'],v['mode'],v['split']) for v in study['results']]
    assert len(listed)==len(wanted)==156 and set(listed)==wanted
    preparation=json.loads((HERE/f'data/{NAME}/input_manifest.json').read_text())
    for group in c['groups']:
        assert r.sha(HERE/group['path'])==group['files']['trajectories.npz']
        pg=next(x for x in preparation['groups'] if (x['data_seed'],x['mode'],x['split'])==(ds,group['mode'],group['split']))
        for file,sha in pg['files'].items():assert r.sha(HERE/file)==sha,file
    estimator_cache={};transition_cache={};oracle_cache={};inference_count=forecast_count=checks=0
    for entry in study['results']:
        method,seed,mode,split=[entry[k] for k in ['method','seed','mode','split']]
        assert entry['data_seed']==ds and entry['length']==length
        folder=HERE/f'runs/{NAME}/d{ds}/L{length}/observation_eval/{method}_s{seed}/{mode}/{split}'
        assert entry==json.loads((folder/'metrics.json').read_text())
        for file,sha in entry['files'].items():assert r.sha(folder/file)==sha
        with np.load(folder/'predictions.npz') as a:saved={k:a[k].copy() for k in a.files}
        if method in ['rgb_cnn','measurement_mlp']:
            key=method,seed
            if key not in estimator_cache:
                ck=torch.load(HERE/f'runs/{NAME}/d{ds}/L{length}/{method}_s{seed}/model.pt',map_location='cpu',weights_only=True)
                model=LengthEstimator(method);model.load_state_dict(ck['state_dict']);model.eval();estimator_cache[key]=model
            x=load_inputs(ds,mode,split,method,length);parts=[]
            for i in range(0,len(next(iter(x.values()))),32):parts.append(estimator_cache[key]({k:v[i:i+32] for k,v in x.items()}))
            raw={k:torch.cat([p[k] for p in parts]) for k in ['state','context']}
            stored=torch.load(folder/'raw_estimator.pt',map_location='cpu',weights_only=True)
            for k in raw:torch.testing.assert_close(raw[k],stored[k],atol=0,rtol=0)
            state,ctx=decode(raw);state=state.numpy().astype(np.float64);ctx=ctx.numpy().astype(np.float64);scores=raw['state'][...,5].numpy()
        else:
            if method=='rgb_analytic':
                with np.load(HERE/f'data/{NAME}/d{ds}/measurements/{mode}/{split}/predictions.npz') as a:state=a[f'states_L{length}'].copy();ctx=a[f'contexts_L{length}'].copy()
            else:
                with np.load(HERE/f'reports/{NAME}/d{ds}/references/{mode}/{split}/controls.npz') as a:state=a[f'L{length}__'+method+'_states'].copy();ctx=a[f'L{length}__'+method+'_contexts'].copy()
            scores=state[:,:,6]
        for key,value in [('estimated_states',state),('estimated_contexts',ctx),('presence_scores',scores)]:np.testing.assert_array_equal(saved[key],value)
        source=HERE/f'data/{NAME}/d{ds}/inputs/{mode}/{split}'
        with np.load(source/'labels.npz') as a:
            truth=color_address(a['past_states'][:,-1]);target=color_address(a['next_states'])
            true_context=np.c_[a['net_acceleration'],a['drag']];events=a['past_events'][:,8-length:].copy()
        with np.load(source/'commands.npz') as a:colors=a['target_color'].copy();delta=a['delta_velocity'].copy()
        rows=list(csv.DictReader((folder/'inference_rows.csv').open()))
        assert len(rows)==entry['episodes'];check_rows(rows,state,ctx,truth,true_context,events)
        checks+=check_aggregates(rows,entry['inference']);inference_count+=len(rows)
        forecasts=list(csv.DictReader((folder/'forecast_rows.csv').open()))
        assert len(forecasts)==len(state)*len(c['information_controls'])*len(c['observation_predictors'])
        for info in c['information_controls']:
            ss=truth if info.startswith('true_state_') else state;cc=true_context if info.endswith('_true_context') else ctx
            score=ss[:,:,6] if info.startswith('true_state_') else scores
            packed,order,actions,found,discarded=independent_pack(ss,score,colors,delta)
            for kind in c['observation_predictors']:
                if kind in ['inertial','force_wall']:motion=physical_reference(packed,cc,actions,kind)
                else:
                    key=kind,seed
                    if key not in transition_cache:
                        ck=torch.load(state_checkpoint(ds,kind,seed),map_location='cpu',weights_only=True)
                        model=Transition(ck['kind'],ck['hidden']);model.load_state_dict(ck['state_dict']);model.eval();transition_cache[key]=model
                    s=encode_states(packed);aa=torch.tensor(actions,dtype=torch.float32)/3
                    context=torch.tensor(cc,dtype=torch.float32)/torch.tensor([.1,.1,.01])
                    motion=(transition_cache[key](s,aa,context)*torch.tensor([31.5,31.5,3.,3.])).numpy()
                prediction=np.zeros_like(state)
                for i in range(len(state)):
                    for j,color in enumerate(order[i]):
                        if packed[i,j,6]>.5:prediction[i,color]=[*motion[i,j],*packed[i,j,4:]]
                key=info+'__'+kind;np.testing.assert_allclose(prediction,saved[key],atol=1e-10,rtol=0)
                selected=[v for v in forecasts if v['information']==info and v['predictor']==kind];assert len(selected)==len(state)
                check_rows(selected,saved[key],cc,target,true_context,events,info,kind,found,discarded)
                checks+=check_aggregates(selected,entry['forecasts'][key]);forecast_count+=len(selected)
                if info=='true_state_true_context':
                    address=seed,mode,split,kind
                    if address in oracle_cache:np.testing.assert_array_equal(saved[key],oracle_cache[address])
                    else:oracle_cache[address]=saved[key]
        print('verified length C04',ds,method,seed,mode,split,flush=True)
    dump(root/'observation_eval_verification.json',{'all_passed':True,'data_seed':ds,'length':length,'conditions_verified':len(listed),
        'inference_rows_replayed':inference_count,'one_step_forecasts_replayed_and_scored':forecast_count,
        'aggregate_and_interval_checks':checks,'true_state_true_context_identical_across_estimators':True,
        'summary_sha256':r.sha(path),'verifier_sha256':r.sha(__file__),
        'reference_helpers_sha256':r.sha(HERE/'verify_dynamics_observation_eval.py'),
        'scope':'Independent input packing, physical references and every inference/forecast row and interval for this data seed and observation length. The same t=7 target is shared across L; paired cross-length aggregation remains separate.'})


if __name__=='__main__':main()
