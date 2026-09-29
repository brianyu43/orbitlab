"""Replay fresh sources/targets and independently check commands/click annotations."""
import dataclasses
import hashlib
import json
import numpy as np
from common import HERE,dump,r
from object_world import ObjectState,sample_scene,render,rotate_scene,state_slots,overlapping
from object_edit_confirmation import NAME,freeze


def orbit(a):
    return min(hashlib.sha256(np.rot90(a,k).copy(order='C').tobytes()).hexdigest() for k in range(4))


def main():
    c=freeze();root=HERE/f'data/{NAME}';manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['config_sha256']==r.sha(HERE/f'configs/{NAME}.json')
    for f in manifest['files']:assert r.sha(root/f['path'])==f['sha256']
    prior=set()
    for split in ['train','val']+c['splits']:
        with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as a:
            prior.update(orbit(v) for v in a['images'])
        with np.load(HERE/f'data/object_edit_tasks_v1/{split}/tasks.npz') as a:
            prior.update(orbit(v) for v in a['target_images'])
    assert len(prior)==manifest['prior_orbits'];seen=set(prior);ns=nt=nr=0
    for split in c['splits']:
        folder=root/split;scenes=json.loads((folder/'scenes.json').read_text());tasks=json.loads((folder/'tasks.json').read_text())
        with np.load(folder/'rgb.npz') as a:images=a['images'].copy();assert a.files==['images']
        with np.load(folder/'scene_labels.npz') as a:labels={k:a[k].copy() for k in a.files}
        with np.load(folder/'queries.npz') as a:queries={k:a[k].copy() for k in a.files};assert set(a.files)=={'source_indices','commands','click_xy'}
        with np.load(folder/'targets.npz') as a:targets={k:a[k].copy() for k in a.files}
        assert len(scenes)==c['evaluation_n'];count={'count3':3,'count4':4}.get(split,2)
        for i,rec in enumerate(scenes):
            ss=tuple(ObjectState(**v) for v in rec['objects'])
            assert ss==sample_scene(rec['candidate_index'],c['seed'],split,count,split=='occlusion')
            assert len(ss)==count and len({v.color for v in ss})==count
            for j,obj in enumerate(ss):assert (obj.shape==obj.color)==(split=='ood' and j==0)
            rr=render(ss);np.testing.assert_array_equal(rr['image'],images[i]);np.testing.assert_array_equal(state_slots(ss),labels['states'][i])
            np.testing.assert_array_equal(rr['segmentation'],labels['segmentation'][i])
            for key,stored in [('visible','visible_masks'),('amodal','amodal_masks')]:
                np.testing.assert_array_equal(np.rint(rr[key]*255).astype(np.uint8),labels[stored][i])
            if split!='occlusion':assert not overlapping(ss)
            for k in range(4):
                rotated=render(rotate_scene(ss,k));np.testing.assert_array_equal(rotated['image'],np.rot90(images[i],k));nr+=1
            h=orbit(images[i]);assert h==rec['orbit_sha256'];family={h};assigned=[t for t in tasks if t['source_index']==i]
            assert {'color','rotate','translate','color_then_rotate'}<=set(t['operation'] for t in assigned)
            tid=i%count;obj=ss[tid];choices=[x for x in range(6) if x not in {v.color for v in ss if v.id!=tid} and x not in [obj.color,obj.shape]]
            recolor=choices[i%len(choices)]
            expected_ops=4+int(obj.shape!=obj.color and obj.shape not in {v.color for v in ss if v.id!=tid});assert len(assigned)==expected_ops
            for task in assigned:
                j=task['task_index'];assert task['target_object_id']==tid and queries['source_indices'][j]==i
                target=list(ss);changed=dataclasses.asdict(obj);cmd=np.zeros((2,13),np.float32);op=task['operation']
                if op in ['color','novel_color_pair','color_then_rotate']:
                    color=obj.shape if op=='novel_color_pair' else recolor;changed['color']=color;cmd[0,0]=1;cmd[0,3+color]=1
                if op in ['rotate','color_then_rotate']:
                    changed['pose']=(obj.pose+1)%4;step=1 if op=='color_then_rotate' else 0;cmd[step,1]=1;cmd[step,9:11]=[0,-1]
                if op=='translate':
                    dx=5 if obj.x<=47 else -5;changed['x']+=dx;cmd[0,2]=1;cmd[0,11]=dx/31.5
                target[tid]=ObjectState(**changed);target=tuple(target)
                assert target==tuple(ObjectState(**v) for v in task['target_objects'])
                np.testing.assert_allclose(queries['commands'][j],cmd,atol=1e-7,rtol=0)
                np.testing.assert_array_equal(state_slots(target),targets['states'][j])
                tr=render(target);np.testing.assert_array_equal(tr['image'],targets['images'][j])
                if split!='occlusion':assert not overlapping(target)
                for other in range(count):
                    if other!=tid:
                        assert target[other]==ss[other]
                        np.testing.assert_array_equal(tr['amodal'][other],rr['amodal'][other])
                        if split!='occlusion':np.testing.assert_array_equal(tr['visible'][other],rr['visible'][other])
                yy,xx=np.where(rr['segmentation']==tid+1);assert len(xx)
                distance=(xx-xx.mean())**2+(yy-yy.mean())**2;ix=int(distance.argmin());click=[int(xx[ix]),int(yy[ix])]
                assert task['click_xy']==click;np.testing.assert_array_equal(queries['click_xy'][j],click)
                th=orbit(tr['image']);assert th==task['target_orbit_sha256'];family.add(th);nt+=1
            assert not family&seen;seen.update(family);ns+=1
        assert manifest['splits'][split]['tasks']==len(tasks)
        print('verified fresh edit data',split,len(scenes),len(tasks),flush=True)
    assert len(seen)-len(prior)==manifest['new_orbits']
    dump(HERE/f'reports/{NAME}/data_verification.json',{'all_passed':True,'source_scenes_replayed':ns,'target_edits_replayed':nt,
        'global_source_rotation_replays':nr,'prior_orbits_recomputed':len(prior),'new_orbits_recomputed':len(seen)-len(prior),
        'cross_prior_and_new_source_family_overlap':0,'commands_and_visible_clicks_independently_checked':True,
        'ground_truth_excluded_from_rgb_and_query_archives':True,'verifier_sha256':r.sha(__file__)})


if __name__=='__main__':main()
