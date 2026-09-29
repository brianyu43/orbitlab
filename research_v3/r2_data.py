"""Build full crossed R2 development inputs without choosing model outcomes."""
import argparse,json,time
import numpy as np
import r2_world as w
from v3_common import HERE,sha,dump,lock,event,status

def freeze():
    return lock(w.BASE/'data_protocol.json',{
        'version':1,'sources':{p:sha(HERE/p) for p in ['r2_world.py','r2_data.py','r1_core.py']},
        'development':w.DEVELOPMENT,'confirmation':w.CONFIRMATION,'train_clips':1024,'calibration_clips':128,
        'eval_counts':[2,3,4,6],'clips_per_visual_condition':16,'visual_cross':'same/different RGB x palette/continuous RGB x black/textured background x 0/.25/.5/.75 current-target occlusion',
        'window':'4 frames stored current,t-1,t-2,t-3; both static models see current reference window[0]; temporal model receives only past+current observations, processed oldest-to-current. No future frame.',
        'occluder':'Moving opaque panel; targeted alpha-mass occlusion at current frame, progressively less in earlier frames, no panel at t-3. Records actual visible fractions for every object. Temporal disocclusion opportunity is deliberately favorable and not representative of all occlusions.',
        'objects':'Four polygons, known geometrically symmetric shape3 rendered symmetrically, radii5..8, integer velocity -1..1 per axis, nonoverlapping >.05 object supports in all four frames. Panel may also affect other objects, retained in visibility labels.',
        'labels':'Current per-object state and current amodal/visible masks are stored separately from RGB inputs. All learned readers will get only RGB at inference. Mask supervision is an explicit experiment axis; state supervision is disclosed for all supervised readers.',
        'split':'Whole vector scene/velocity/color C4 family split; backgrounds, occlusion and frames stay with the family. Never count frames as independent scenes.',
        'planned_model_comparison':'Learned center detector+crop, six-slot single-frame reader, six-slot recurrent reader; with/without current-mask auxiliary supervision. Known-palette legacy reader only a domain-limited diagnostic, not an upper bound on continuous/same-color scenes.',
        'scoring_plan':'Object count, miss/merge, click-target selection, current-state and edit success by actual occlusion, non-target factors. Geometric pose equivalence for shape3. Probability-set coverage will be reported with calibration scope and width, not as a proof of recovered hidden state.',
        'rendering_scope':'Known-renderer state-edit diagnostics first. Learned pixel decoder and background/occluder preservation require separate evaluation; no GT background will be quietly passed to a claimed image-only edit model.'})

def prepare_group(seed,split,count,n):
    folder=w.BASE/f'data/d{seed}/{split}/n{count}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text())
        for name,digest in m['files'].items():assert sha(folder/name)==digest
        return
    images=[];masks=[];visible=[];records=[];candidate=0
    while len(records)<n:
        cond=w.CONDITIONS[len(records)%len(w.CONDITIONS)]
        try:im,am,vi,row=w.sample(seed,split,candidate,count,cond)
        except RuntimeError:
            candidate+=1
            if candidate>n*20:raise
            continue
        row['candidate_index']=candidate;row['source_index']=len(records);candidate+=1
        images.append(im);masks.append(np.rint(am*255).astype(np.uint8));visible.append(np.rint(vi*255).astype(np.uint8));records.append(row)
    np.savez_compressed(folder/'rgb.npz',images=np.stack(images))
    np.savez_compressed(folder/'masks.npz',amodal=np.stack(masks),visible=np.stack(visible))
    dump(folder/'labels.json',records)
    files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    dump(folder/'manifest.json',{'seed':seed,'split':split,'count':count,'clips':n,'candidates':candidate,'files':files,'protocol_sha256':sha(w.BASE/'data_protocol.json')})
    print('R2 data',seed,split,count,n,flush=True);event('R2_data_group_complete',seed=seed,split=split,count=count,clips=n)

def prepare(seed):
    freeze();status('R2','data_preparation_running')
    prepare_group(seed,'train',2,1024);prepare_group(seed,'calibration',2,128)
    split='val' if seed==w.DEVELOPMENT else 'test'
    for count in [2,3,4,6]:prepare_group(seed,split,count,16*len(w.CONDITIONS))
    seen=set();groups=[]
    for p in sorted((w.BASE/'data').glob('d*/*/n*/labels.json')):
        rows=json.loads(p.read_text());keys={w.vector_key(r['objects']) for r in rows}
        assert len(keys)==len(rows);assert not keys&seen;seen.update(keys);groups.append(str(p.relative_to(w.BASE)))
    dump(w.BASE/'data_family_audit.json',{'groups':groups,'unique_scene_families':len(seen),'cross_new_split_overlap':0,'rendering_replay':'pending'})
    status('R2','development_data_ready_models_not_yet_trained',data='r2/data/d887101')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=w.DEVELOPMENT);a=p.parse_args();prepare(a.seed)
