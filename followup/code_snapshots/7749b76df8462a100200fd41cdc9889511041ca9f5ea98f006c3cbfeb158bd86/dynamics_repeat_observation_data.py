"""Separate new RGB/commands from labels; repeat frozen image measurements."""
import json
import time
import numpy as np
from common import HERE,dump,r
from dynamics_repeat_data import NAME,freeze as freeze_parent
from dynamics_rgb_baseline import measure_frame,infer
from evaluate_dynamics_rgb_baseline import canonical,state_context_rows,aggregate

CONFIG='dynamics_repeat_observation_data_v1'


def freeze():
    parent=freeze_parent();path=HERE/f'configs/{CONFIG}.json'
    verification=HERE/f'reports/{NAME}/data_verification.json';assert json.loads(verification.read_text())['all_passed']
    if not path.exists():dump(path,{'data_seeds':parent['new_data_seeds'],
        'parent_sha256':r.sha(HERE/f'configs/{NAME}.json'),'data_manifest_sha256':r.sha(HERE/f'data/{NAME}/manifest.json'),
        'data_verification_sha256':r.sha(verification),
        'source_sha256':{p:r.sha(HERE/p) for p in ['dynamics_repeat_observation_data.py','dynamics_rgb_baseline.py','evaluate_dynamics_rgb_baseline.py']},
        'input_archives':{'observations.npz':['images'],'commands.npz':['target_color','delta_velocity']},
        'measurement_priors':'Unchanged known palette/circle renderer and free-flight eight-substep kinematics. No true state/count/context/event input. Not learned concepts or physics.',
        'privileged_control':'True-position analytic control saved separately under reports; never included in learned estimator inputs.',
        'normalization':'The original fixed physical scales/clipping, no fitted train/test normalizer.',
        'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for p,sha in c['source_sha256'].items():assert r.sha(HERE/p)==sha,p
    assert c['parent_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    assert c['data_manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json')
    return c


def main():
    c=freeze();manifest=json.loads((HERE/f'data/{NAME}/manifest.json').read_text());groups=[];started=time.time()
    for g in manifest['groups']:
        seed,mode,split=g['data_seed'],g['mode'],g['split'];base=HERE/f'data/{NAME}/d{seed}'
        folder=base/f'inputs/{mode}/{split}';measured=base/f'measurements/{mode}/{split}'
        reports=HERE/f'reports/{NAME}/d{seed}/rgb_baseline/{mode}/{split}'
        if (reports/'group.json').exists():
            report=json.loads((reports/'group.json').read_text())
            for p,sha in report['files'].items():assert r.sha(HERE/p)==sha,p
            groups.append(report);continue
        for p in [folder,measured,reports]:
            if p.exists():raise RuntimeError(f'Incomplete observation preparation retained: {p}')
            p.mkdir(parents=True)
        with np.load(HERE/g['path']) as a:
            images=a['observations'].copy();states=a['states'][:,:4].copy();context=a['contexts'].copy()
            n=len(states);target=a['target_ids'].copy();action=a['actions'][:,3].copy()
            past_events=a['events'][:,:3].copy();next_events=a['events'][:,3].copy();next_states=a['states'][:,4].copy()
        colors=states[np.arange(n),3,target,5].astype(np.int64);dv=action[np.arange(n),target]
        np.savez_compressed(folder/'observations.npz',images=images)
        np.savez_compressed(folder/'commands.npz',target_color=colors,delta_velocity=dv)
        np.savez_compressed(folder/'labels.npz',past_states=states,next_states=next_states,net_acceleration=context[:,:2]+context[:,2:4],
                            drag=context[:,4],past_events=past_events,next_events=next_events,target_ids=target)
        features=np.stack([[measure_frame(frame) for frame in sequence] for sequence in images])
        outputs=[infer(f) for f in features];s=np.stack([v[0] for v in outputs]);ctx=np.stack([v[1] for v in outputs])
        np.savez_compressed(measured/'predictions.npz',features=features,states=s,contexts=ctx)
        dump(measured/'fit_info.json',[v[2] for v in outputs])
        past=canonical(states);truth=past[:,-1];ct=np.column_stack([context[:,:2]+context[:,2:4],context[:,4]])
        true_features=np.zeros_like(features);true_features[...,:2]=past[...,:2]
        true_features[...,2]=past[...,4];true_features[...,3]=past[...,6]
        privileged=[infer(f) for f in true_features]
        finite_state=s.copy();finite_state[:,:,2:4]=features[:,-1,:,:2]-features[:,-2,:,:2];finite_state*=finite_state[:,:,6:7]
        controls={'rgb_analytic':(s,ctx),'true_positions_analytic':(np.stack([v[0] for v in privileged]),np.stack([v[1] for v in privileged])),
                  'rgb_finite_difference_zero_context':(finite_state,np.zeros_like(ctx))}
        arrays={};metrics={}
        for method,(ss,cc) in controls.items():
            arrays[method+'_states']=ss;arrays[method+'_contexts']=cc
            rows=state_context_rows(ss,cc,truth,ct,past_events);metrics[method]=aggregate(rows)
            r.write_csv(reports/f'{method}_rows.csv',rows)
        np.savez_compressed(reports/'controls.npz',**arrays);dump(reports/'metrics.json',metrics)
        files={str(p.relative_to(HERE)):r.sha(p) for parent in [folder,measured,reports] for p in parent.iterdir() if p.is_file()}
        report={'data_seed':seed,'mode':mode,'split':split,'episodes':n,'source_archive_sha256':g['archive_sha256'],
                'files':files,'metrics':metrics}
        dump(reports/'group.json',report);groups.append(report)
        print('repeat observation preparation',seed,mode,split,n,flush=True)
        dump(HERE/f'reports/{NAME}/observation_data_progress.json',{'completed_groups':len(groups),'expected_groups':38,'seconds':time.time()-started})
    dump(HERE/f'data/{NAME}/observation_manifest.json',{'groups':groups,'episodes':sum(g['episodes'] for g in groups),
        'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'seconds':time.time()-started})


if __name__=='__main__':main()
