"""Fresh frozen RGB/click edit confirmation, with evaluation labels stored apart."""
import argparse
import dataclasses
import json
import time
import numpy as np
from common import HERE, dump, fresh_dir, r
from object_world import sample_scene, render, edit, state_slots, overlapping
from object_tasks import command, operations

NAME='object_image_edit_v1'
SPLITS=['test','ood','count3','count4','occlusion']


def freeze():
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        checkpoints={}
        for kind in ['flat','slot']:
            rel=f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt';checkpoints[rel]=r.sha(HERE/rel)
            for seed in [0,1,2]:
                rel=f'runs/object_readout_v1/{kind}_s{seed}/model.pt';checkpoints[rel]=r.sha(HERE/rel)
        for seed in [0,1,2]:
            rel=f'runs/object_oracle_study_v1/object_s{seed}/model.pt';checkpoints[rel]=r.sha(HERE/rel)
        sources=['object_edit_confirmation.py','object_image_edit.py','object_world.py','object_tasks.py',
                 'object_state_readout.py','object_oracle_study.py','object_perception.py','object_perception_v2.py','cache_object_slots.py']
        dump(path,{'seed':995031,'evaluation_n':128,'splits':SPLITS,'kinds':['flat','slot'],'head_seeds':[0,1,2],
            'epsilon_seed':995100,'batch':16,'checkpoints':checkpoints,'sources':{p:r.sha(HERE/p) for p in sources},
            'prior_manifests':{p:r.sha(HERE/p) for p in ['data/object_world_v1/manifest.json','data/object_edit_tasks_v1/manifest.json']},
            'head_selection':'Frozen final 6000-step supervised heads; one fixed encoder per architecture. No new training or test selection.',
            'source_selection':'Fresh deterministic candidates. Every object must have a visible hard pixel. Reject source/target C4 orbit overlap with all prior object scenes/edits and any other new source family.',
            'commands':'For source i, target i modulo object count. Seen recolor, +90 degree rotation, 5px horizontal translation (+x if x<=47 else -x), sequential recolor then rotation. Novel recolor if it changes color and no other object uses that color.',
            'layout':'All source and edited scenes except occlusion are nonoverlapping. Occlusion edits may legitimately change visibility; non-target factors remain identical.',
            'click':'Hard-visible target pixel nearest the centroid of its hard-visible pixels; row-major tie break. Annotation supplies only (x,y) pixel, no center, mask, ID or count to the RGB model.',
            'capacity':'Decode five slots. Keep top four by predicted presence logit, stable ties by original slot index, then restore original slot order. Threshold presence sigmoid>=0.5. Record discarded positive detections.',
            'selection':'Closest present predicted center to the click; ties follow retained original slot order. No present slot: leave state unchanged and score target selection as failure.',
            'controllers':['identity','analytic','learned_object_same_seed'],
            'oracle_controls':['true_state_true_id_analytic','true_state_true_id_object_seed_0_1_2','true_state_click_analytic'],
            'composition':'Reuse selected predicted slot across commands, no intermediate truth or rematching.',
            'rendering':'Render predicted states only. Painter order is retained predicted slot order, never reordered by evaluation matching. Round/clamp only for raster output: center -32..95, radius 1..12. Raw states remain scored.',
            'metrics':'Initial geometry-only matching fixed after edits. Separate GT target/untouched/non-target/count/full-state success from command-following and preservation relative to the predicted source. End-to-end success additionally requires correct selection. Image errors also reported against literal input-image passthrough.',
            'tolerances':{'xy_each_axis_pixels':1,'radius_pixels':0.5,'presence_threshold':0.5},
            'uncertainty':'Base source scenes, not individual commands, are bootstrap units. Three head/controller seeds on a single new data-generation seed and single encoder seed per kind.',
            'frozen_unix':time.time()})
    c=json.loads(path.read_text())
    for group in ['checkpoints','sources','prior_manifests']:
        for rel,sha in c[group].items():assert r.sha(HERE/rel)==sha,rel
    return c


def previous_orbits():
    found=set()
    for split in ['train','val']+SPLITS:
        meta=json.loads((HERE/f'data/object_world_v1/{split}/scenes.json').read_text())
        records=json.loads((HERE/f'data/object_edit_tasks_v1/{split}/records.json').read_text())
        found.update(v['orbit_sha256'] for v in meta)
        found.update(v['target_orbit_sha256'] for v in records)
    return found


def visible_click(segmentation,tid):
    y,x=np.nonzero(segmentation==tid+1)
    if not len(x):raise ValueError('Target has no hard-visible pixel')
    j=int(np.argmin((x-x.mean())**2+(y-y.mean())**2))
    return [int(x[j]),int(y[j])]


def planned_tasks(scene,source_index):
    tid=source_index%len(scene);obj=scene[tid];others={v.color for v in scene if v.id!=tid}
    colors=[c for c in range(6) if c not in others and c!=obj.color and c!=obj.shape]
    color=colors[source_index%len(colors)]
    delta=[5 if obj.x<=47 else -5,0]
    choices=[('color',color),('rotate',1),('translate',delta),('color_then_rotate',{'color':color,'rotation':1})]
    if obj.shape!=obj.color and obj.shape not in others:choices.append(('novel_color_pair',obj.shape))
    result=[]
    for op,value in choices:
        record={'source_index':source_index,'target_object_id':tid,'operation':op,'value':value}
        target=scene;cmds=np.zeros((2,13),np.float32)
        for j,(action,v) in enumerate(operations(record)):
            cmds[j]=command(action,v);target=edit(target,tid,action,v)
        record.update({'source_target_pair_is_unseen':obj.shape==obj.color,
            'target_pair_is_unseen':target[tid].shape==target[tid].color,
            'target_objects':[dataclasses.asdict(v) for v in target]})
        result.append((record,target,cmds))
    return result


def build(c):
    root=fresh_dir(HERE/f'data/{NAME}');prior=previous_orbits();seen=set(prior);summary={};files=[]
    for split in c['splits']:
        folder=fresh_dir(root/split);count={'count3':3,'count4':4}.get(split,2)
        images=[];states=[];segments=[];visible=[];amodal=[];scenes=[];records=[];targets=[];ys=[];commands=[];clicks=[];indices=[]
        candidate=0;reasons={'invisible':0,'edited_overlap':0,'orbit_overlap':0}
        while len(images)<c['evaluation_n']:
            idx=candidate;candidate+=1;scene=sample_scene(idx,c['seed'],split,count,split=='occlusion');source=render(scene)
            if any(not np.any(source['segmentation']==v.id+1) for v in scene):reasons['invisible']+=1;continue
            tasks=planned_tasks(scene,len(images))
            if split!='occlusion' and any(overlapping(t) for _,t,_ in tasks):reasons['edited_overlap']+=1;continue
            rendered=[render(t) for _,t,_ in tasks];h=r.orbit_hash(source['image']);th=[r.orbit_hash(t['image']) for t in rendered];family={h,*th}
            if family&seen:reasons['orbit_overlap']+=1;continue
            seen.update(family);i=len(images);images.append(source['image']);states.append(state_slots(scene));segments.append(source['segmentation'])
            visible.append(np.rint(source['visible']*255).astype(np.uint8));amodal.append(np.rint(source['amodal']*255).astype(np.uint8))
            scenes.append({'source_index':i,'candidate_index':idx,'orbit_sha256':h,'objects':[dataclasses.asdict(v) for v in scene],
                'visible_fraction':[float(source['visible'][v.id].sum()/source['amodal'][v.id].sum()) for v in scene]})
            for (record,target,cmd),rt,target_hash in zip(tasks,rendered,th):
                tid=record['target_object_id'];click=visible_click(source['segmentation'],tid)
                records.append({**record,'task_index':len(records),'target_orbit_sha256':target_hash,'click_xy':click})
                targets.append(rt['image']);ys.append(state_slots(target));commands.append(cmd);clicks.append(click);indices.append(i)
        np.savez_compressed(folder/'rgb.npz',images=np.stack(images))
        np.savez_compressed(folder/'scene_labels.npz',states=np.stack(states),segmentation=np.stack(segments),visible_masks=np.stack(visible),amodal_masks=np.stack(amodal))
        np.savez_compressed(folder/'queries.npz',source_indices=np.array(indices),commands=np.stack(commands),click_xy=np.array(clicks))
        np.savez_compressed(folder/'targets.npz',states=np.stack(ys),images=np.stack(targets))
        dump(folder/'scenes.json',scenes);dump(folder/'tasks.json',records)
        summary[split]={'scenes':len(images),'tasks':len(records),'candidates':candidate,'rejections':reasons}
        for path in sorted(folder.iterdir()):files.append({'path':str(path.relative_to(root)),'sha256':r.sha(path)})
        print('fresh edit data',split,summary[split],flush=True)
    dump(root/'manifest.json',{'splits':summary,'files':files,'prior_orbits':len(prior),'new_orbits':len(seen)-len(prior),
        'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'generator_sha256':r.sha(__file__),
        'cross_prior_and_new_source_family_orbit_overlap':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args();config=freeze()
    if not args.freeze_only:build(config)
