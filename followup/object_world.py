"""Small multi-object world with explicit state and exact pixel C4 rotations.

Object-local pose edits and changes of the global coordinate frame are distinct.
Masks and factors are ground truth for oracle training/evaluation only; image-only
models must not receive them as inputs.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import time
from functools import lru_cache
from PIL import Image, ImageDraw
from common import HERE, dump, fresh_dir, update_status, np, torch, o, r

SIZE=64
MAX_OBJECTS=4


@dataclasses.dataclass(frozen=True)
class ObjectState:
    id: int
    shape: int
    color: int
    x: int
    y: int
    radius: int
    pose: int


@lru_cache(maxsize=128)
def local_alpha(shape,radius,pose):
    side=33;ss=4
    p=Image.new('L',(side*ss,side*ss))
    vertices=[((side/2+x*radius)*ss,(side/2+y*radius)*ss) for x,y in o.POLYGONS[shape]]
    ImageDraw.Draw(p).polygon(vertices,fill=255)
    a=np.asarray(p.resize((side,side),Image.Resampling.LANCZOS)).astype(np.float32)/255
    return np.ascontiguousarray(np.rot90(a,pose%4))


def object_alpha(obj):
    patch=local_alpha(obj.shape,obj.radius,obj.pose);out=np.zeros((SIZE,SIZE),dtype=np.float32)
    x0,y0=obj.x-16,obj.y-16;x1,y1=x0+33,y0+33
    left,top=max(0,x0),max(0,y0);right,bottom=min(SIZE,x1),min(SIZE,y1)
    if right>left and bottom>top:out[top:bottom,left:right]=patch[top-y0:bottom-y0,left-x0:right-x0]
    return out


def render(scene):
    scene=tuple(sorted(scene,key=lambda v:v.id))
    if not 1<=len(scene)<=MAX_OBJECTS or [v.id for v in scene]!=list(range(len(scene))):
        raise ValueError('Require consecutive persistent object IDs and 1..4 objects')
    masks=np.zeros((MAX_OBJECTS,SIZE,SIZE),np.float32)
    rgb=np.zeros((SIZE,SIZE,3),np.float32)
    for obj in scene:
        if obj.shape not in range(4) or obj.color not in range(6):raise ValueError('Unknown factor')
        masks[obj.id]=object_alpha(obj);a=masks[obj.id,:,:,None]
        rgb=rgb*(1-a)+np.asarray(o.PALETTE[obj.color],dtype=np.float32)[None,None]*a
    visible=masks.copy();transmittance=np.ones((SIZE,SIZE),np.float32)
    for obj in reversed(scene):
        visible[obj.id]*=transmittance;transmittance*=1-masks[obj.id]
    weights=np.concatenate([transmittance[None],visible],0)
    segmentation=weights.argmax(0).astype(np.uint8)
    return {'image':rgb.round().clip(0,255).astype(np.uint8),'amodal':masks,'visible':visible,'segmentation':segmentation}


def rotate_scene(scene,k):
    result=tuple(scene)
    for _ in range(k%4):
        result=tuple(dataclasses.replace(v,x=v.y,y=SIZE-1-v.x,pose=(v.pose+1)%4) for v in result)
    return result


def translate_scene(scene,dx,dy):
    return tuple(dataclasses.replace(v,x=v.x+dx,y=v.y+dy) for v in scene)


def overlapping(scene):
    masks=render(scene)['amodal']
    return bool(((masks>0).sum(0)>1).any())


def sample_scene(index,seed,split,n_objects=2,occluded=False):
    split_id={'train':0,'val':1,'test':2,'ood':3,'count3':4,'count4':5,'occlusion':6}[split]
    rng=np.random.default_rng(np.random.SeedSequence([seed,split_id,index,n_objects,int(occluded)]))
    scene=[];used_colors=set()
    for object_id in range(n_objects):
        held_out=split=='ood' and object_id==0
        pairs=[(k,c) for k in range(4) for c in range(6) if (k==c)==held_out and c not in used_colors]
        shape,color=pairs[int(rng.integers(len(pairs)))];used_colors.add(color)
        radius=int(rng.integers(5,9));pose=int(rng.integers(4))
        found=False
        for _ in range(5000):
            if occluded and object_id==1:
                x=scene[0].x+int(rng.integers(-6,7));y=scene[0].y+int(rng.integers(-6,7))
                x=int(np.clip(x,12,51));y=int(np.clip(y,12,51))
            else:x,y=(int(v) for v in rng.integers(12,52,size=2))
            candidate=tuple(scene+[ObjectState(object_id,shape,color,x,y,radius,pose)])
            if occluded or not overlapping(candidate):
                scene=list(candidate);found=True;break
        if not found:raise RuntimeError('Scene placement exhausted; retain failure instead of relaxing overlap rule')
    return tuple(scene)


def edit(scene,target_id,operation,value):
    if target_id not in [v.id for v in scene]:raise ValueError('Missing target')
    result=[]
    for obj in scene:
        if obj.id!=target_id:result.append(obj);continue
        if operation=='color':obj=dataclasses.replace(obj,color=int(value))
        elif operation=='rotate':obj=dataclasses.replace(obj,pose=(obj.pose+int(value))%4)
        elif operation=='translate':obj=dataclasses.replace(obj,x=obj.x+int(value[0]),y=obj.y+int(value[1]))
        else:raise ValueError(operation)
        result.append(obj)
    return tuple(result)


def safe_translation(scene,target_id):
    for delta in [(5,0),(-5,0),(0,5),(0,-5),(3,3),(-3,-3)]:
        candidate=edit(scene,target_id,'translate',delta)
        moved=candidate[target_id]
        if 11<=moved.x<=52 and 11<=moved.y<=52 and not overlapping(candidate):return delta
    raise ValueError('No non-occluding translation for this base scene')


def state_slots(scene):
    slots=np.zeros((MAX_OBJECTS,16),np.float32)
    for v in scene:
        slots[v.id,v.shape]=1;slots[v.id,4+v.color]=1
        slots[v.id,10:13]=[(v.x-31.5)/31.5,(v.y-31.5)/31.5,v.radius/8]
        angle=-v.pose*np.pi/2;slots[v.id,13:15]=[np.cos(angle),np.sin(angle)];slots[v.id,15]=1
    return slots


def build():
    cfg={'seed':942301,'train_n':1024,'evaluation_n':128,'max_objects':4,
         'splits':['train','val','test','ood','count3','count4','occlusion'],
         'held_out_pairs':[[k,k] for k in range(4)],
         'layout':'Integer centers in 12..51, radius 5..8, exact local C4 poses. Distinct colors per scene before intervention.',
         'depth':'Persistent ID ascending is painter order. Higher IDs may occlude lower IDs.',
         'masks':'Amodal and visible alpha masks plus segmentation; privileged labels, not image-only model inputs.',
         'actions':'One target at a time. Color/pose/translation; composed color+pose is reserved for evaluation.',
         'ood':'First object uses a held-out shape-color pair; others seen. count3/count4 and occlusion use seen pairs.',
         'selection':'Deterministic candidates; reject cross-split source and target orbit duplicates.',
         'frozen_unix':time.time()}
    cp=HERE/'configs/object_world_v1.json'
    if not cp.exists():dump(cp,cfg)
    cfg=json.loads(cp.read_text());root=fresh_dir(HERE/'data/object_world_v1')
    global_orbits=set();split_records={};source_count=0;edit_count=0;rotation_tests=0
    for split in cfg['splits']:
        folder=root/split;folder.mkdir();n=cfg['train_n'] if split=='train' else cfg['evaluation_n']
        count=3 if split=='count3' else (4 if split=='count4' else 2)
        images=[];masks=[];visible=[];seg=[];states=[];records=[];edits=[];targets=[];local_orbits=set();candidate=0
        while len(images)<n:
            scene=sample_scene(candidate,cfg['seed'],split,count,split=='occlusion');candidate+=1
            rendered=render(scene);h=r.orbit_hash(rendered['image'])
            if h in global_orbits or h in local_orbits:continue
            planned=[]
            if split in ['train','val','test','ood']:
                target_id=len(images)%2;target=scene[target_id]
                colors=[c for c in range(6) if c!=target.color and c!=target.shape and c not in [v.color for v in scene if v.id!=target_id]]
                color=colors[(candidate-1)%len(colors)]
                try:delta=safe_translation(scene,target_id)
                except ValueError:continue
                actions=[('color',color),('rotate',1),('translate',delta)]
                for action,value in actions:planned.append((action,value,edit(scene,target_id,action,value)))
                if split!='train':
                    composed=edit(edit(scene,target_id,'color',color),target_id,'rotate',1)
                    planned.append(('color_then_rotate',{'color':color,'rotation':1},composed))
                # If source already OOD, do not describe recoloring back into seen pairs as novel-target creation.
                if split in ['val','test'] and target.shape not in [v.color for v in scene if v.id!=target_id]:
                    planned.append(('novel_color_pair',target.shape,edit(scene,target_id,'color',target.shape)))
                if any(overlapping(v[2]) for v in planned):continue
                rendered_targets=[render(v[2]) for v in planned]
                target_hashes={r.orbit_hash(v['image']) for v in rendered_targets}
                if target_hashes&global_orbits:continue
            else:rendered_targets=[];target_hashes=set()
            i=len(images);local_orbits.add(h);local_orbits.update(target_hashes)
            for k in range(4):
                rr=render(rotate_scene(scene,k))
                assert np.array_equal(rr['image'],np.rot90(rendered['image'],k))
                assert np.array_equal(rr['segmentation'],np.rot90(rendered['segmentation'],k))
                assert np.array_equal(rr['amodal'],np.rot90(rendered['amodal'],k,axes=(1,2)))
                rotation_tests+=1
            images.append(rendered['image']);masks.append((rendered['amodal']*255).round().astype(np.uint8))
            visible.append((rendered['visible']*255).round().astype(np.uint8));seg.append(rendered['segmentation']);states.append(state_slots(scene))
            records.append({'base_id':f'{cfg["seed"]}:{split}:{candidate-1}','orbit_sha256':h,'objects':[dataclasses.asdict(v) for v in scene],
                'visible_fraction':[float(rendered['visible'][v.id].sum()/max(rendered['amodal'][v.id].sum(),1e-8)) for v in scene]})
            for (action,value,changed),rt in zip(planned,rendered_targets):
                # Non-target object layers remain identical; visibility may change only in overlapping scenes (not in this edit set).
                for v in scene:
                    if v.id!=target_id:
                        assert v==changed[v.id]
                        assert np.array_equal(rt['amodal'][v.id],rendered['amodal'][v.id])
                        assert np.array_equal(rt['visible'][v.id],rendered['visible'][v.id])
                target_index=len(targets);targets.append(rt['image'])
                edits.append({'source_index':i,'target_index':target_index,'target_object_id':target_id,'operation':action,'value':value,
                    'target_orbit_sha256':r.orbit_hash(rt['image']),'target_objects':[dataclasses.asdict(v) for v in changed],
                    'creates_unseen_pair':action=='novel_color_pair'})
        global_orbits.update(local_orbits)
        np.savez_compressed(folder/'scenes.npz',images=np.stack(images),amodal_masks=np.stack(masks),visible_masks=np.stack(visible),segmentation=np.stack(seg),state_slots=np.stack(states))
        dump(folder/'scenes.json',records)
        if targets:
            np.savez_compressed(folder/'edits.npz',images=np.stack(targets));dump(folder/'edits.json',edits)
        o.grid(torch.from_numpy(np.stack(images[:32])).permute(0,3,1,2).float()/255,folder/'examples.png',8)
        if targets:
            example=[]
            for e in edits[:12]:example.extend([images[e['source_index']],targets[e['target_index']]])
            o.grid(torch.from_numpy(np.stack(example)).permute(0,3,1,2).float()/255,folder/'edit_examples.png',6)
        split_records[split]={'n':n,'objects_per_scene':count,'candidate_count':candidate,'edits':len(edits),
            'scene_sha256':r.sha(folder/'scenes.npz'),'metadata_sha256':r.sha(folder/'scenes.json'),
            'mean_visible_fraction':float(np.mean([f for v in records for f in v['visible_fraction']]))}
        source_count+=n;edit_count+=len(edits);print('object world',split,split_records[split],flush=True)
    report={'splits':split_records,'source_scenes':source_count,'edit_pairs':edit_count,'global_rotation_checks':rotation_tests,
        'cross_split_source_and_target_orbits_disjoint':True,'config_sha256':r.sha(cp),'code_sha256':r.sha(__file__)}
    dump(root/'manifest.json',report)
    update_status('B01','complete',['data/object_world_v1/manifest.json','configs/object_world_v1.json'])
    update_status('B02','complete',['data/object_world_v1/manifest.json','data/object_world_v1/test/edits.json'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');a=p.parse_args()
    if a.build:torch.set_num_threads(2);build()
