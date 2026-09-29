"""Score image-derived and explicitly privileged position-derived estimates."""
import json
import numpy as np
from common import HERE,dump,fresh_dir,r
from dynamics_observation_inputs import NAME as INPUTS
from dynamics_rgb_baseline import NAME,freeze,infer


def canonical(raw):
    result=np.zeros((*raw.shape[:-2],6,7),np.float64)
    for i in np.ndindex(raw.shape[:-2]):
        for obj in raw[i]:
            if obj[4]>0:result[i+(int(obj[5]),)]=obj
    return result


def state_context_rows(state,context,truth,true_context,past_events):
    rows=[]
    for i,(pred,actual,ctx,ct) in enumerate(zip(state,truth,context,true_context)):
        pg=pred[:,6]>.5;gt=actual[:,6]>.5;matched=pg&gt;error=np.abs(pred[:,:5]-actual[:,:5])
        rows.append({'episode':i,'past_contact':bool(past_events[i].any()),'past_pair_contact':bool(past_events[i,:,0].any()),
            'past_wall_contact':bool(past_events[i,:,1].any()),'true_count':int(gt.sum()),'predicted_count':int(pg.sum()),
            'count_correct':bool(pg.sum()==gt.sum()),'all_presence_correct':bool(np.array_equal(pg,gt)),
            'recall':float(matched.sum()/gt.sum()),'false_positive_colors':int((pg&~gt).sum()),
            'matched_position_mae_pixels':float(error[matched,:2].mean()) if matched.any() else None,
            'matched_velocity_mae_pixels_per_frame':float(error[matched,2:4].mean()) if matched.any() else None,
            'matched_radius_mae_pixels':float(error[matched,4].mean()) if matched.any() else None,
            'zero_filled_true_slot_position_mae':float(error[gt,:2].mean()),'zero_filled_true_slot_velocity_mae':float(error[gt,2:4].mean()),
            'state_scene_success':bool(np.array_equal(pg,gt) and (error[gt,:2]<=1).all() and (error[gt,2:4]<=.1).all() and (error[gt,4]<=.5).all()),
            'net_force_mae_pixels_per_frame_squared':float(np.abs(ctx[:2]-ct[:2]).mean()),'drag_absolute_error':float(abs(ctx[2]-ct[2]))})
    return rows


def aggregate(rows):
    groups={'all':rows,'no_past_contact':[v for v in rows if not v['past_contact']],
        'past_contact':[v for v in rows if v['past_contact']],'past_pair_contact':[v for v in rows if v['past_pair_contact']],
        'past_wall_contact':[v for v in rows if v['past_wall_contact']]}
    exclude={'episode','past_contact','past_pair_contact','past_wall_contact','true_count','predicted_count'}
    return {group:{'episodes':len(rr),'metrics':{key:r.bootstrap([v[key] for v in rr if v[key] is not None]) if any(v[key] is not None for v in rr) else None
        for key in rows[0] if key not in exclude}} for group,rr in groups.items() if rr}


def main():
    freeze();root=fresh_dir(HERE/f'reports/{NAME}');manifest=json.loads((HERE/f'data/{NAME}/manifest.json').read_text());reports=[]
    assert json.loads((HERE/f'reports/{INPUTS}/verification.json').read_text())['all_passed']
    for g in manifest['groups']:
        mode,split=g['mode'],g['split'];folder=fresh_dir(root/mode/split)
        with np.load(HERE/f'data/{NAME}/{mode}/{split}/predictions.npz') as a:features=a['features'].copy();image_state=a['states'].copy();image_context=a['contexts'].copy()
        with np.load(HERE/f'data/{INPUTS}/{mode}/{split}/labels.npz') as a:
            past=canonical(a['past_states']);truth=past[:,-1];true_context=np.concatenate([a['net_acceleration'],a['drag'][:,None]],1);events=a['past_events'].copy()
        true_features=np.zeros_like(features);true_features[:,:,:,:2]=past[:,:,:,:2];true_features[:,:,:,2]=past[:,:,:,4];true_features[:,:,:,3]=past[:,:,:,6]
        privileged=[infer(f) for f in true_features];truepos_state=np.stack([v[0] for v in privileged]);truepos_context=np.stack([v[1] for v in privileged])
        finite_state=image_state.copy();finite_state[:,:,2:4]=features[:,-1,:,:2]-features[:,-2,:,:2];finite_state*=finite_state[:,:,6:7]
        predictions={'rgb_analytic':(image_state,image_context),'true_positions_analytic':(truepos_state,truepos_context),
            'rgb_finite_difference_zero_context':(finite_state,np.zeros_like(image_context))}
        arrays={};all_rows=[];metrics={}
        for method,(s,c) in predictions.items():
            arrays[method+'_states']=s;arrays[method+'_contexts']=c
            rows=state_context_rows(s,c,truth,true_context,events);r.write_csv(folder/f'{method}_rows.csv',rows);metrics[method]=aggregate(rows)
            m=metrics[method]['all']['metrics'];print('RGB baseline score',mode,split,method,
                {key:round(m[key]['mean'],6) if m[key] else None for key in ['count_correct','matched_position_mae_pixels','matched_velocity_mae_pixels_per_frame','net_force_mae_pixels_per_frame_squared','drag_absolute_error']},flush=True)
        np.savez_compressed(folder/'controls.npz',**arrays);dump(folder/'metrics.json',metrics)
        reports.append({'mode':mode,'split':split,'metrics':metrics,'files':{p.name:r.sha(p) for p in sorted(folder.iterdir())}})
    dump(root/'summary.json',{'groups':reports,'prediction_manifest_sha256':r.sha(HERE/f'data/{NAME}/manifest.json'),
        'evaluator_sha256':r.sha(__file__),'scope':'Known-renderer image reference and privileged exact-position control; one four-frame observation window per trajectory. No learned model or future rollout result.'})


if __name__=='__main__':main()
