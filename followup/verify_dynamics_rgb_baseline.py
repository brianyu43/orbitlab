"""Replay every RGB fit and independently recompute context and error metrics."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_rgb_baseline import NAME,freeze,measure_frame


def reference_context_velocity(features):
    pos=features[:,:,:2];alive=features[:,:,3]>.5;valid=alive.all(0)
    if valid.any():
        increments=pos[1:,valid]-pos[:-1,valid];x=increments[:-1].reshape(-1,2);y=increments[1:].reshape(-1,2)
        centered=x-x.mean(0);variance=(centered**2).sum()
        d=(centered*(y-y.mean(0))).sum()/variance if variance>1e-12 else 1.
        decay=np.clip(d,np.exp(-.05),1);drag=-np.log(decay);b=1 if drag<1e-10 else (1-decay)/drag
        accel=(y-decay*x).mean(0)/b
    else:decay=1.;drag=0.;accel=np.zeros(2);b=1.
    # Direct sum of per-substep responses, independently of the fitting helper.
    A=sum(np.exp(-drag*j/8) for j in range(1,9))/8
    B=sum((j/8 if drag<1e-10 else -np.expm1(-drag*j/8)/drag) for j in range(1,9))/8
    out=np.zeros((6,7))
    for color in range(6):
        if not alive[-1,color]:continue
        dp=pos[-1,color]-pos[-2,color] if alive[-2,color] else np.zeros(2)
        previous_velocity=(dp-B*accel)/A
        velocity=previous_velocity*decay+b*accel
        out[color]=[*pos[-1,color],*velocity,features[-1,color,2],color,1]
    return out,np.array([*accel,drag])


def main():
    freeze();root=HERE/f'reports/{NAME}';summary=json.loads((root/'summary.json').read_text());manifest=json.loads((HERE/f'data/{NAME}/manifest.json').read_text())
    assert summary['prediction_manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json');assert summary['evaluator_sha256']==r.sha(HERE/'evaluate_dynamics_rgb_baseline.py')
    frames=episodes=rows_checked=aggregates=0
    for g in manifest['groups']:
        mode,split=g['mode'],g['split'];folder=HERE/f'data/{NAME}/{mode}/{split}';score_folder=root/mode/split
        entry=next(v for v in summary['groups'] if (v['mode'],v['split'])==(mode,split))
        for file,sha in entry['files'].items():assert sha==r.sha(score_folder/file)
        assert r.sha(folder/'predictions.npz')==g['prediction_sha256'] and r.sha(folder/'fit_info.json')==g['info_sha256']
        with np.load(HERE/f'data/{INPUTS}/{mode}/{split}/observations.npz') as a:images=a['images'].copy()
        with np.load(folder/'predictions.npz') as a:pred={k:a[k].copy() for k in a.files}
        with np.load(HERE/f'data/{INPUTS}/{mode}/{split}/labels.npz') as a:labels={k:a[k].copy() for k in a.files}
        with np.load(score_folder/'controls.npz') as a:controls={k:a[k].copy() for k in a.files}
        truth=np.zeros((len(images),6,7));features_gt=np.zeros_like(pred['features'])
        for i in range(len(images)):
            for t in range(4):
                actual=measure_frame(images[i,t]);np.testing.assert_array_equal(actual,pred['features'][i,t]);frames+=1
                for obj in labels['past_states'][i,t]:
                    if obj[4]>0:
                        co=int(obj[5]);features_gt[i,t,co,:4]=[*obj[:2],obj[4],1]
                        if t==3:truth[i,co]=obj
            state,context=reference_context_velocity(pred['features'][i]);np.testing.assert_allclose(state,pred['states'][i],atol=1e-9,rtol=0);np.testing.assert_allclose(context,pred['contexts'][i],atol=1e-9,rtol=0)
            state,context=reference_context_velocity(features_gt[i]);np.testing.assert_allclose(state,controls['true_positions_analytic_states'][i],atol=1e-9,rtol=0);np.testing.assert_allclose(context,controls['true_positions_analytic_contexts'][i],atol=1e-9,rtol=0)
            episodes+=1
        np.testing.assert_array_equal(controls['rgb_analytic_states'],pred['states']);np.testing.assert_array_equal(controls['rgb_analytic_contexts'],pred['contexts'])
        for method in entry['metrics']:
            state=controls[method+'_states'];context=controls[method+'_contexts'];rows=list(csv.DictReader((score_folder/f'{method}_rows.csv').open()));assert len(rows)==len(images)
            for i,row in enumerate(rows):
                s=state[i];t=truth[i];p=s[:,6]>.5;gt=t[:,6]>.5;both=p&gt;error=np.abs(s[:,:5]-t[:,:5]);events=labels['past_events'][i]
                values={'episode':i,'past_contact':bool(events.any()),'past_pair_contact':bool(events[:,0].any()),'past_wall_contact':bool(events[:,1].any()),
                    'true_count':int(gt.sum()),'predicted_count':int(p.sum()),'count_correct':bool(p.sum()==gt.sum()),'all_presence_correct':bool(np.array_equal(p,gt)),
                    'recall':float(both.sum()/gt.sum()),'false_positive_colors':int((p&~gt).sum()),
                    'matched_position_mae_pixels':float(error[both,:2].mean()) if both.any() else None,
                    'matched_velocity_mae_pixels_per_frame':float(error[both,2:4].mean()) if both.any() else None,
                    'matched_radius_mae_pixels':float(error[both,4].mean()) if both.any() else None,
                    'zero_filled_true_slot_position_mae':float(error[gt,:2].mean()),'zero_filled_true_slot_velocity_mae':float(error[gt,2:4].mean()),
                    'state_scene_success':bool(np.array_equal(p,gt) and np.max(error[gt,:2])<=1 and np.max(error[gt,2:4])<=.1 and np.max(error[gt,4])<=.5),
                    'net_force_mae_pixels_per_frame_squared':float(abs(context[i,:2]-labels['net_acceleration'][i]).mean()),
                    'drag_absolute_error':float(abs(context[i,2]-labels['drag'][i]))}
                assert set(row)==set(values)
                for key,value in values.items():
                    if isinstance(value,bool):assert row[key]==str(value)
                    elif value is None:assert row[key]==''
                    else:assert abs(float(row[key])-value)<1e-10,(method,key)
                rows_checked+=1
            for group,stats in entry['metrics'][method].items():
                selected=[v for v in rows if group=='all' or (v['past_contact']=='False' if group=='no_past_contact' else v[group]=='True')]
                assert len(selected)==stats['episodes']
                for key,stat in stats['metrics'].items():
                    vv=[v[key] for v in selected if v[key]!='']
                    if not vv:assert stat is None;continue
                    numbers=np.array([float(v=='True') if v in ['True','False'] else float(v) for v in vv]);rng=np.random.default_rng(9123)
                    boot=numbers[rng.integers(len(numbers),size=(1000,len(numbers)))].mean(1)
                    assert abs(numbers.mean()-stat['mean'])<1e-10;np.testing.assert_allclose(np.quantile(boot,[.025,.975]),stat['base_scene_ci95'],atol=1e-10,rtol=0);aggregates+=1
        print('verified RGB/state/context reference',mode,split,len(images),flush=True)
    dump(root/'verification.json',{'all_passed':True,'rgb_frames_replayed':frames,'episode_estimates_independently_recomputed':episodes,
        'state_context_metric_rows_checked':rows_checked,'aggregate_and_interval_checks':aggregates,'verifier_sha256':r.sha(__file__),
        'scope':summary['scope']})


if __name__=='__main__':main()
