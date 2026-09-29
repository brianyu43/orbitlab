"""Check RGB-only input separation, every measurement, and privileged controls."""
import csv
import json
import numpy as np
from common import HERE,dump,r
from dynamics_repeat_data import NAME
from dynamics_repeat_observation_data import CONFIG,freeze
from dynamics_rgb_baseline import measure_frame
from verify_dynamics_rgb_baseline import reference_context_velocity


def main():
    freeze();path=HERE/f'data/{NAME}/observation_manifest.json';manifest=json.loads(path.read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{CONFIG}.json')
    frames=episodes=rows_count=aggregates=0
    for g in manifest['groups']:
        ds,mode,split=g['data_seed'],g['mode'],g['split'];root=HERE/f'data/{NAME}/d{ds}'
        folder=root/f'inputs/{mode}/{split}';measured=root/f'measurements/{mode}/{split}'
        report=HERE/f'reports/{NAME}/d{ds}/rgb_baseline/{mode}/{split}'
        for filename,sha in g['files'].items():assert r.sha(HERE/filename)==sha,filename
        raw_path=root/f'world/{mode}/{split}/trajectories.npz';assert r.sha(raw_path)==g['source_archive_sha256']
        with np.load(raw_path) as a:raw={k:a[k].copy() for k in a.files}
        with np.load(folder/'observations.npz') as a:assert a.files==['images'];images=a['images'].copy()
        with np.load(folder/'commands.npz') as a:
            assert set(a.files)=={'target_color','delta_velocity'};colors=a['target_color'].copy();dv=a['delta_velocity'].copy()
        with np.load(folder/'labels.npz') as a:labels={k:a[k].copy() for k in a.files}
        with np.load(measured/'predictions.npz') as a:
            assert set(a.files)=={'features','states','contexts'};pred={k:a[k].copy() for k in a.files}
        with np.load(report/'controls.npz') as a:controls={k:a[k].copy() for k in a.files}
        np.testing.assert_array_equal(images,raw['observations']);np.testing.assert_array_equal(labels['past_states'],raw['states'][:,:4])
        np.testing.assert_array_equal(labels['next_states'],raw['states'][:,4]);np.testing.assert_array_equal(labels['past_events'],raw['events'][:,:3])
        np.testing.assert_array_equal(labels['next_events'],raw['events'][:,3]);np.testing.assert_array_equal(labels['target_ids'],raw['target_ids'])
        np.testing.assert_array_equal(labels['net_acceleration'],raw['contexts'][:,:2]+raw['contexts'][:,2:4]);np.testing.assert_array_equal(labels['drag'],raw['contexts'][:,4])
        n=len(images);targets=raw['target_ids'];np.testing.assert_array_equal(colors,raw['states'][np.arange(n),3,targets,5])
        np.testing.assert_array_equal(dv,raw['actions'][np.arange(n),3,targets])
        truth=np.zeros((n,6,7));true_features=np.zeros_like(pred['features'])
        for i in range(n):
            for t in range(4):
                np.testing.assert_array_equal(measure_frame(images[i,t]),pred['features'][i,t]);frames+=1
                for obj in labels['past_states'][i,t]:
                    if obj[4]>0:
                        color=int(obj[5]);true_features[i,t,color,:4]=[*obj[:2],obj[4],1]
                        if t==3:truth[i,color]=obj
            s,ctx=reference_context_velocity(pred['features'][i])
            np.testing.assert_allclose(s,pred['states'][i],atol=1e-9,rtol=0);np.testing.assert_allclose(ctx,pred['contexts'][i],atol=1e-9,rtol=0)
            s,ctx=reference_context_velocity(true_features[i])
            np.testing.assert_allclose(s,controls['true_positions_analytic_states'][i],atol=1e-9,rtol=0)
            np.testing.assert_allclose(ctx,controls['true_positions_analytic_contexts'][i],atol=1e-9,rtol=0)
            episodes+=1
        np.testing.assert_array_equal(controls['rgb_analytic_states'],pred['states']);np.testing.assert_array_equal(controls['rgb_analytic_contexts'],pred['contexts'])
        finite_state=pred['states'].copy();finite_state[:,:,2:4]=pred['features'][:,-1,:,:2]-pred['features'][:,-2,:,:2];finite_state*=finite_state[:,:,6:7]
        np.testing.assert_array_equal(finite_state,controls['rgb_finite_difference_zero_context_states']);assert not controls['rgb_finite_difference_zero_context_contexts'].any()
        for method,metrics in g['metrics'].items():
            s=controls[method+'_states'];ctx=controls[method+'_contexts'];rows=list(csv.DictReader((report/f'{method}_rows.csv').open()));assert len(rows)==n
            for i,row in enumerate(rows):
                p=s[i,:,6]>.5;t=truth[i,:,6]>.5;both=p&t;error=abs(s[i,:,:5]-truth[i,:,:5]);events=labels['past_events'][i]
                values={'episode':i,'past_contact':bool(events.any()),'past_pair_contact':bool(events[:,0].any()),'past_wall_contact':bool(events[:,1].any()),
                    'true_count':int(t.sum()),'predicted_count':int(p.sum()),'count_correct':bool(p.sum()==t.sum()),'all_presence_correct':bool(np.array_equal(p,t)),
                    'recall':float(both.sum()/t.sum()),'false_positive_colors':int((p&~t).sum()),
                    'matched_position_mae_pixels':float(error[both,:2].mean()) if both.any() else None,
                    'matched_velocity_mae_pixels_per_frame':float(error[both,2:4].mean()) if both.any() else None,
                    'matched_radius_mae_pixels':float(error[both,4].mean()) if both.any() else None,
                    'zero_filled_true_slot_position_mae':float(error[t,:2].mean()),'zero_filled_true_slot_velocity_mae':float(error[t,2:4].mean()),
                    'state_scene_success':bool(np.array_equal(p,t) and (error[t,:2]<=1).all() and (error[t,2:4]<=.1).all() and (error[t,4]<=.5).all()),
                    'net_force_mae_pixels_per_frame_squared':float(abs(ctx[i,:2]-labels['net_acceleration'][i]).mean()),
                    'drag_absolute_error':float(abs(ctx[i,2]-labels['drag'][i]))}
                assert set(values)==set(row)
                for key,value in values.items():
                    if isinstance(value,bool):assert row[key]==str(value)
                    elif value is None:assert row[key]==''
                    else:assert abs(float(row[key])-value)<1e-10,(ds,mode,split,method,key)
                rows_count+=1
            for category,m in metrics.items():
                selected=[v for v in rows if category=='all' or (v['past_contact']=='False' if category=='no_past_contact' else v[category]=='True')]
                assert len(selected)==m['episodes']
                for key,stat in m['metrics'].items():
                    values=[v[key] for v in selected if v[key]!='']
                    if not values:assert stat is None;continue
                    numbers=np.array([float(v=='True') if v in ['True','False'] else float(v) for v in values])
                    rng=np.random.default_rng(9123);boot=numbers[rng.integers(len(numbers),size=(1000,len(numbers)))].mean(1)
                    assert abs(numbers.mean()-stat['mean'])<1e-10
                    np.testing.assert_allclose(np.quantile(boot,[.025,.975]),stat['base_scene_ci95'],atol=1e-10,rtol=0);aggregates+=1
        print('verified repeat observation inputs',ds,mode,split,n,flush=True)
    assert episodes==manifest['episodes']==4480 and frames==17920
    dump(HERE/f'reports/{NAME}/observation_data_verification.json',{'all_passed':True,'episodes_verified':episodes,'rgb_measurements_replayed':frames,
        'state_context_rows_independently_checked':rows_count,'aggregate_interval_checks':aggregates,
        'input_archives_exclude_gt_count_state_context_and_ids':True,'privileged_controls_separate':True,
        'manifest_sha256':r.sha(path),'verifier_sha256':r.sha(__file__),
        'reference_context_code_sha256':r.sha(HERE/'verify_dynamics_rgb_baseline.py'),'scope':'Repeated frozen RGB measurements and input separation; no learned model result.'})


if __name__=='__main__':main()
