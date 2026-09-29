"""Public RGB/commands and visible-window analytic estimates; labels remain separate."""
import json
import time
import numpy as np
from common import HERE,dump,r
from dynamics_observation_length_data import NAME,freeze as parent_freeze
from dynamics_observation_length_reference import LENGTHS,infer_window,select_visible
from dynamics_rgb_baseline import measure_frame
from evaluate_dynamics_rgb_baseline import canonical,state_context_rows,aggregate

CONFIG='dynamics_observation_length_inputs_v1'


def freeze():
    parent_freeze();verification=HERE/f'reports/{NAME}/data_verification.json'
    v=json.loads(verification.read_text());assert v['all_passed']
    path=HERE/f'configs/{CONFIG}.json'
    if not path.exists():dump(path,{'parent_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'world_manifest_sha256':r.sha(HERE/f'data/{NAME}/manifest.json'),'world_verification_sha256':r.sha(verification),
        'source_sha256':{f:r.sha(HERE/f) for f in ['prepare_observation_length_inputs.py','dynamics_observation_length_reference.py','dynamics_rgb_baseline.py','evaluate_dynamics_rgb_baseline.py']},
        'input_files':{'observations.npz':['images'],'commands.npz':['target_color','delta_velocity']},
        'lengths':LENGTHS,'known_priors':'Known six colors, circle alpha, eight-substep free-flight equations. Measurements are independent per frame; a shorter reference slices visible measurements before fitting.',
        'stratification':'Past-contact scoring uses the visible L-1 transitions, empty at L=1. All-episode paired length contrasts are primary. No GT event filter enters estimation.',
        'privileged_control':'Exact-position analytic estimates saved only under reports, excluded from learned estimator inputs.',
        'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for f,sha in c['source_sha256'].items():assert r.sha(HERE/f)==sha
    assert c['parent_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    assert c['world_manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json') and c['world_verification_sha256']==r.sha(verification)
    return c


def main():
    c=freeze();groups=[];start=time.time();world=json.loads((HERE/f'data/{NAME}/manifest.json').read_text())
    for g in world['groups']:
        ds,mode,split=[g[k] for k in ['data_seed','mode','split']];base=HERE/f'data/{NAME}/d{ds}'
        inputs=base/f'inputs/{mode}/{split}';measurement=base/f'measurements/{mode}/{split}'
        report=HERE/f'reports/{NAME}/d{ds}/references/{mode}/{split}'
        if (report/'group.json').exists():
            item=json.loads((report/'group.json').read_text())
            for p,sha in item['files'].items():assert r.sha(HERE/p)==sha
            groups.append(item);continue
        for p in [inputs,measurement,report]:
            if p.exists():raise RuntimeError(f'Incomplete inputs retained: {p}')
            p.mkdir(parents=True)
        with np.load(HERE/g['path']) as a:
            images=a['observations'].copy();states=a['states'][:,:8].copy();context=a['contexts'].copy()
            target=a['target_ids'].copy();actions=a['actions'][:,7].copy();next_states=a['states'][:,8].copy()
            events=a['events'][:,:7].copy();next_events=a['events'][:,7].copy()
        n=len(images);colors=states[np.arange(n),7,target,5].astype(np.int64);dv=actions[np.arange(n),target]
        np.savez_compressed(inputs/'observations.npz',images=images)
        np.savez_compressed(inputs/'commands.npz',target_color=colors,delta_velocity=dv)
        np.savez_compressed(inputs/'labels.npz',past_states=states,next_states=next_states,net_acceleration=context[:,:2]+context[:,2:4],
                            drag=context[:,4],past_events=events,next_events=next_events,target_ids=target)
        features=np.stack([[measure_frame(frame) for frame in sequence] for sequence in images])
        past=canonical(states);truth=past[:,-1];true_context=np.c_[context[:,:2]+context[:,2:4],context[:,4]]
        true_features=np.zeros_like(features);true_features[...,:2]=past[...,:2];true_features[...,2]=past[...,4];true_features[...,3]=past[...,6]
        arrays={'features':features};controls={};fit_info={};metrics={}
        for length in LENGTHS:
            fitted=[infer_window(f) for f in select_visible(features,length)]
            ss=np.stack([x[0] for x in fitted]);cc=np.stack([x[1] for x in fitted]);fit_info[str(length)]=[x[2] for x in fitted]
            arrays[f'states_L{length}']=ss;arrays[f'contexts_L{length}']=cc
            privileged=[infer_window(f) for f in select_visible(true_features,length)]
            values={'rgb_analytic':(ss,cc),'true_positions_analytic':(np.stack([x[0] for x in privileged]),np.stack([x[1] for x in privileged]))}
            for method,(state,ctx) in values.items():
                name=f'L{length}__{method}';controls[name+'_states']=state;controls[name+'_contexts']=ctx
                rows=state_context_rows(state,ctx,truth,true_context,events[:,8-length:]);metrics[name]=aggregate(rows)
                r.write_csv(report/(name+'_rows.csv'),rows)
        np.savez_compressed(measurement/'predictions.npz',**arrays);dump(measurement/'fit_info.json',fit_info)
        np.savez_compressed(report/'controls.npz',**controls);dump(report/'metrics.json',metrics)
        files={str(p.relative_to(HERE)):r.sha(p) for folder in [inputs,measurement,report] for p in folder.iterdir() if p.is_file()}
        item={'data_seed':ds,'mode':mode,'split':split,'episodes':n,'world_archive_sha256':r.sha(HERE/g['path']),
              'files':files,'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json')}
        dump(report/'group.json',item);groups.append(item)
        dump(HERE/f'reports/{NAME}/input_progress.json',{'groups_completed':len(groups),'expected_groups':57,'seconds':time.time()-start})
        print('length RGB inputs prepared',ds,mode,split,len(groups),flush=True)
    dump(HERE/f'data/{NAME}/input_manifest.json',{'groups':groups,'episodes':sum(x['episodes'] for x in groups),
        'config_sha256':r.sha(HERE/f'configs/{CONFIG}.json'),'seconds':time.time()-start})


if __name__=='__main__':main()
