"""Replay every RGB measurement and independently check visible-window references."""
import csv
import json
import time
import numpy as np
from common import HERE,dump,r
from dynamics_observation_length_data import NAME
from prepare_observation_length_inputs import CONFIG,freeze
from dynamics_rgb_baseline import measure_frame
from verify_dynamics_observation_eval import color_address,check_rows,check_aggregates


def reference(features):
    length=len(features);xy=features[...,:2];alive=features[...,3]>=.5
    drag=0.;decay=1.;force=np.zeros(2);b=1.
    complete=alive.all(0);den=0.;raw=1.;residual=None
    if length>=4 and complete.any():
        changes=np.diff(xy[:,complete],axis=0)
        x=changes[:-1].reshape(-1,2);y=changes[1:].reshape(-1,2)
        centered=x-x.mean(0);den=float(np.square(centered).sum())
        raw=float((centered*(y-y.mean(0))).sum()/den) if den>1e-12 else 1.
        decay=float(np.clip(raw,np.exp(-.05),1));drag=float(-np.log(decay))
        b=1. if drag<1e-10 else float(-np.expm1(-drag)/drag)
        force=(y-decay*x).mean(0)/b
        residual=float(np.sqrt(np.square(y-decay*x-b*force).mean()))
    A=sum(np.exp(-drag*j/8) for j in range(1,9))/8
    B=sum(j/8 if drag<1e-10 else -np.expm1(-drag*j/8)/drag for j in range(1,9))/8
    state=np.zeros((6,7))
    for co in np.flatnonzero(alive[-1]):
        velocity=np.zeros(2)
        if length>=2:
            change=xy[-1,co]-xy[-2,co] if alive[-2,co] else np.zeros(2)
            velocity=decay*(change-B*force)/A+b*force
        state[co]=[*xy[-1,co],*velocity,features[-1,co,2],co,1]
    info={'complete_tracks':int(complete.sum()),'design_variance':den,'unclipped_decay':raw,
          'residual_rms':residual,'identified_design':bool(den>1e-12),'short_window_default':bool(length<3)}
    return state,np.array([*force,drag]),info


def main():
    freeze();path=HERE/f'data/{NAME}/input_manifest.json';manifest=json.loads(path.read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
    assert len(manifest['groups'])==57
    started=time.time();frames=episodes=estimates=rows_checked=aggregate_checks=0
    for gi,g in enumerate(manifest['groups']):
        ds,mode,split=[g[k] for k in ['data_seed','mode','split']]
        base=HERE/f'data/{NAME}/d{ds}';folder=base/f'inputs/{mode}/{split}'
        measured=base/f'measurements/{mode}/{split}';report=HERE/f'reports/{NAME}/d{ds}/references/{mode}/{split}'
        for file,sha in g['files'].items():assert r.sha(HERE/file)==sha,file
        world=base/f'world/{mode}/{split}/trajectories.npz';assert r.sha(world)==g['world_archive_sha256']
        with np.load(world) as a:raw={k:a[k].copy() for k in ['observations','states','contexts','actions','events','target_ids']}
        with np.load(folder/'observations.npz') as a:assert a.files==['images'];images=a['images'].copy()
        with np.load(folder/'commands.npz') as a:
            assert set(a.files)=={'target_color','delta_velocity'};colors=a['target_color'].copy();dv=a['delta_velocity'].copy()
        with np.load(folder/'labels.npz') as a:labels={k:a[k].copy() for k in a.files}
        assert set(labels)=={'past_states','next_states','net_acceleration','drag','past_events','next_events','target_ids'}
        with np.load(measured/'predictions.npz') as a:pred={k:a[k].copy() for k in a.files}
        assert set(pred)=={'features',*[f'{key}_L{l}' for l in [1,2,4,8] for key in ['states','contexts']]}
        with np.load(report/'controls.npz') as a:controls={k:a[k].copy() for k in a.files}
        info=json.loads((measured/'fit_info.json').read_text());metrics=json.loads((report/'metrics.json').read_text())
        np.testing.assert_array_equal(images,raw['observations']);n=len(images)
        expectations={'past_states':raw['states'][:,:8],'next_states':raw['states'][:,8],
            'net_acceleration':raw['contexts'][:,:2]+raw['contexts'][:,2:4],'drag':raw['contexts'][:,4],
            'past_events':raw['events'][:,:7],'next_events':raw['events'][:,7],'target_ids':raw['target_ids']}
        for key,value in expectations.items():np.testing.assert_array_equal(labels[key],value)
        np.testing.assert_array_equal(colors,raw['states'][np.arange(n),7,raw['target_ids'],5])
        np.testing.assert_array_equal(dv,raw['actions'][np.arange(n),7,raw['target_ids']])
        true_features=np.zeros_like(pred['features']);truth=color_address(raw['states'][:,7])
        for i in range(n):
            for t in range(8):
                np.testing.assert_array_equal(measure_frame(images[i,t]),pred['features'][i,t]);frames+=1
                for obj in raw['states'][i,t]:
                    if obj[4]>0:true_features[i,t,int(obj[5]),:4]=[*obj[:2],obj[4],1]
            for length in [1,2,4,8]:
                for method,features in [('rgb_analytic',pred['features']),('true_positions_analytic',true_features)]:
                    state,ctx,details=reference(features[i,-length:])
                    name=f'L{length}__{method}'
                    np.testing.assert_allclose(state,controls[name+'_states'][i],atol=1e-9,rtol=0)
                    np.testing.assert_allclose(ctx,controls[name+'_contexts'][i],atol=1e-9,rtol=0);estimates+=1
                    if method=='rgb_analytic':
                        np.testing.assert_array_equal(pred[f'states_L{length}'][i],controls[name+'_states'][i])
                        np.testing.assert_array_equal(pred[f'contexts_L{length}'][i],controls[name+'_contexts'][i])
                        for key,value in details.items():
                            got=info[str(length)][i][key]
                            if value is None or isinstance(value,bool):assert got==value
                            else:np.testing.assert_allclose(got,value,atol=1e-9,rtol=1e-12)
        true_context=np.c_[labels['net_acceleration'],labels['drag']]
        assert len(metrics)==8 and len(controls)==16
        for length in [1,2,4,8]:
            for method in ['rgb_analytic','true_positions_analytic']:
                name=f'L{length}__{method}';rows=list(csv.DictReader((report/(name+'_rows.csv')).open()))
                assert len(rows)==n
                events=labels['past_events'][:,8-length:]
                check_rows(rows,controls[name+'_states'],controls[name+'_contexts'],truth,true_context,events)
                expected_groups={'all','no_past_contact'} if length==1 else {'all'}
                for key,index in [('past_contact',None),('past_pair_contact',0),('past_wall_contact',1)]:
                    if (events.any() if index is None else events[...,index].any()):expected_groups.add(key)
                if (~events.any(axis=(1,2))).any():expected_groups.add('no_past_contact')
                assert set(metrics[name])==expected_groups
                aggregate_checks+=check_aggregates(rows,metrics[name]);rows_checked+=n
        episodes+=n
        dump(HERE/f'reports/{NAME}/input_verification_progress.json',{'groups_verified':gi+1,'expected_groups':57,'episodes_verified':episodes,'seconds':time.time()-started})
        print('verified length inputs',ds,mode,split,gi+1,flush=True)
    assert episodes==manifest['episodes']==6720 and frames==53760 and estimates==53760 and rows_checked==53760
    dump(HERE/f'reports/{NAME}/input_verification.json',{'all_passed':True,'groups_verified':57,'episodes_verified':episodes,
        'rgb_measurements_replayed':frames,'visible_window_estimates_independently_recomputed':estimates,
        'state_context_rows_checked':rows_checked,'aggregate_and_interval_checks':aggregate_checks,
        'input_archives_exclude_gt_count_state_context_ids':True,'privileged_controls_separate':True,
        'manifest_sha256':r.sha(path),'verifier_sha256':r.sha(__file__),
        'metric_verifier_sha256':r.sha(HERE/'verify_dynamics_observation_eval.py'),'seconds':time.time()-started,
        'scope':'Paired t=7 inputs and L-specific analytic references. Learned input noninterference and training audits are separate.'})


if __name__=='__main__':main()
