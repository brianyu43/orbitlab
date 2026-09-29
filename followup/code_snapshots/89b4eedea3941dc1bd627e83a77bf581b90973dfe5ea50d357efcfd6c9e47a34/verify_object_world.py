"""Replay every stored scene and intervention, independently checking split contracts."""
import dataclasses
import json
from pathlib import Path
from common import HERE, dump, np, r, update_status
from object_world import ObjectState, render, rotate_scene, state_slots


def scene(items):
    return tuple(ObjectState(**v) for v in items)


def main():
    root=HERE/'data/object_world_v1';manifest=json.loads((root/'manifest.json').read_text())
    cfg=json.loads((HERE/'configs/object_world_v1.json').read_text());seen=set();counts={};files=[]
    assert r.sha(HERE/'configs/object_world_v1.json')==manifest['config_sha256']
    total=0;pairs=0;rotations=0
    for split in cfg['splits']:
        folder=root/split;records=json.loads((folder/'scenes.json').read_text());a=np.load(folder/'scenes.npz')
        assert r.sha(folder/'scenes.npz')==manifest['splits'][split]['scene_sha256']
        assert r.sha(folder/'scenes.json')==manifest['splits'][split]['metadata_sha256']
        assert len(records)==len(a['images'])==manifest['splits'][split]['n']
        local=set();visibility=[]
        for i,record in enumerate(records):
            ss=scene(record['objects']);rr=render(ss)
            np.testing.assert_array_equal(a['images'][i],rr['image'])
            for saved,key in [('amodal_masks','amodal'),('visible_masks','visible')]:
                np.testing.assert_array_equal(a[saved][i],(rr[key]*255).round().astype(np.uint8))
            np.testing.assert_array_equal(a['segmentation'][i],rr['segmentation'])
            np.testing.assert_array_equal(a['state_slots'][i],state_slots(ss))
            h=r.orbit_hash(a['images'][i]);assert h==record['orbit_sha256'];local.add(h)
            assert len(ss)==manifest['splits'][split]['objects_per_scene']
            assert len({v.color for v in ss})==len(ss)
            assert all((v.shape==v.color)==(split=='ood' and v.id==0) for v in ss)
            actual=[float(rr['visible'][v.id].sum()/rr['amodal'][v.id].sum()) for v in ss]
            np.testing.assert_allclose(actual,record['visible_fraction'],rtol=0,atol=1e-7)
            visibility.extend(actual)
            if split!='occlusion':assert all(v==1. for v in actual)
            for k in range(4):
                rot=render(rotate_scene(ss,k))
                np.testing.assert_array_equal(rot['image'],np.rot90(rr['image'],k))
                np.testing.assert_array_equal(rot['visible'],np.rot90(rr['visible'],k,axes=(1,2)))
                rotations+=1
        edits=[]
        if (folder/'edits.json').exists():
            edits=json.loads((folder/'edits.json').read_text());targets=np.load(folder/'edits.npz')['images']
            assert len(targets)==len(edits)==manifest['splits'][split]['edits']
            for i,e in enumerate(edits):
                assert e['target_index']==i
                source=scene(records[e['source_index']]['objects']);target=scene(e['target_objects']);tid=e['target_object_id']
                rr=render(target);np.testing.assert_array_equal(targets[i],rr['image'])
                h=r.orbit_hash(targets[i]);assert h==e['target_orbit_sha256'];local.add(h)
                before,after=dataclasses.asdict(source[tid]),dataclasses.asdict(target[tid])
                changed={k for k in before if before[k]!=after[k]}
                allowed={'color':{'color'},'novel_color_pair':{'color'},'rotate':{'pose'},
                    'translate':{'x','y'},'color_then_rotate':{'color','pose'}}[e['operation']]
                assert changed and changed<=allowed
                for j,(s,t) in enumerate(zip(source,target)):
                    if j!=tid:
                        assert s==t
                        np.testing.assert_array_equal(render(source)['visible'][j],rr['visible'][j])
                if e['operation'] in ['color','novel_color_pair']:assert after['color']==e['value']
                if e['operation']=='rotate':assert after['pose']==(before['pose']+e['value'])%4
                if e['operation']=='translate':assert [after['x']-before['x'],after['y']-before['y']]==e['value']
                if e['operation']=='color_then_rotate':
                    assert after['color']==e['value']['color'] and after['pose']==(before['pose']+1)%4
                if split=='train':assert e['operation'] in ['color','rotate','translate'] and all(v.shape!=v.color for v in target)
                if e['operation']=='novel_color_pair':assert target[tid].shape==target[tid].color
        assert not seen.intersection(local);seen.update(local)
        counts[split]={'scenes':len(records),'edits':len(edits),'unique_source_and_target_orbits':len(local),
            'objects_with_some_occlusion':sum(v<.99999 for v in visibility),'mean_visible_fraction':float(np.mean(visibility))}
        total+=len(records);pairs+=len(edits)
        for p in sorted(folder.iterdir()):files.append({'path':str(p.relative_to(root)),'sha256':r.sha(p),'bytes':p.stat().st_size})
    assert total==manifest['source_scenes']==1792 and pairs==manifest['edit_pairs']
    report={'all_passed':True,'source_scenes':total,'edit_pairs':pairs,'rotation_replays':rotations,
        'cross_split_source_and_target_orbit_overlap':0,'split_checks':counts,'files':files,
        'verifier_sha256':r.sha(__file__),'renderer_sha256':r.sha(HERE/'object_world.py'),
        'claim_boundary':'Data correctness only. No learned object recognition or intervention performance is established.'}
    out=HERE/'reports/object_world_v1/verification.json';dump(out,report)
    for task in ['B01','B02']:update_status(task,'complete',['data/object_world_v1/manifest.json','reports/object_world_v1/verification.json'])
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))


if __name__=='__main__':main()
