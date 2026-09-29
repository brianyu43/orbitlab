"""Decoded dSprites semantics, exact C4 image-family audit and train-only split.

No model scores are read; official test membership is never modified.
"""
import argparse,hashlib,io,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from v3_common import ROOT,HERE,sha,dump,lock
from r5_intake import BASE,Store,TASKS

SHAPES=['circle','triangle','square','star_4']
COLORS=[[0,255,0],[255,0,255],[0,127,255],[255,127,0]]
SIZES=[.125,.225,.325,.425]
SEED=890101
UPSTREAM=ROOT/'followup/external/svib/data_creation/dsprites'

def freeze():
    return lock(BASE/'dsprites_audit_protocol.json',{'version':3,'source_sha256':sha(__file__),'intake_manifest_sha256':sha(BASE/'data/dsprites_hard/manifest.json'),
        'upstream_sources':{n:sha(UPSTREAM/n) for n in ['macros.py','rule.py','utils.py','create_data.py','spriteworld/renderers/pil_renderer.py']},
        'tasks':list(TASKS),'shapes':SHAPES,'metadata_colors':COLORS,'sizes':SIZES,
        'pixel_policy':'Keep native PNG RGB unchanged; cv2write yields channel-reversed metadata palette. Maskobject0isred and1isblue in stored RGB. Verify nearest palette identity of median opaque-interior RGB; retain channel deviations>2 as antialias diagnostics, not automatic corruption. Never recolor pixels.',
        'preflight_amendments':['audit_preflight_v1/AMENDMENT.json','audit_preflight_v2/AMENDMENT.json'],
        'mask_visibility_caveat':'Some MultipleAtomic target masks use initial-source-color overdraw while targetRGBusesnewcolors. Target masked palette identities are diagnostic, not a clean-visibility assertion. Re-render everyflaggedtarget; do not modifydata or useprovidedmasks as unquestionedvisiblelabels.',
        'renderer_diagnostic_sha256':sha(HERE/'r5_renderer_diagnosis.py'),
        'split_seed':SEED,'training':57600,'validation':6400,'official_test':8000,
        'family_policy':'Whole C4pixel orbit hashes of both source and target; connect any shared image orbit. Same connected component cannot cross new train/validation. Official test retained, overlaps reported.',
        'split_selection':'Deterministic shuffled source/target-orbit connected components using890101. Allocate complete components to validation up to6400; require exactly6400. No image outcomes, labels or scores used to select membership.',
        'exposure':'Original project previously used allSingleAtomic64000train and8000test; new models train fresh on57600only. Test is not previously unseen at project level. Other tasks not previously trained.',
        'rules':'Match published source: ShAASh,MMix2,NShAC,MNMix2. MultipleNonAtomic size uses OTHERobject color+OTHERobject quadrant in the code; also count disagreement with literal paper self-quadrant size formula.',
        'rotation_audit':'Hypothetical world-space quarter-turn about(.5,.5), not re-rendered official test augmentation. Compare semantic shape/color/size rule outputs; isolate quadrant dependence. Does not claim pixel symmetry of 3Dtasks.'})

def factors(objects):
    return np.array([[SHAPES.index(v['shape']),COLORS.index(list(v['color'])),SIZES.index(v['size'])] for v in objects],np.int16)

def quadrant(xy):
    x,y=xy[...,0],xy[...,1]
    return np.where(y<.5,np.where(x<.5,3,4),np.where(x<.5,2,1)).astype(np.int16)

def rule(attrs,xy,task,paper=False):
    result=attrs.copy();shape,color,size=attrs[...,0],attrs[...,1],attrs[...,2]
    if task=='Single_Atomic':result[...,0]=np.flip(shape,axis=-1)
    elif task=='Multiple_Atomic':result[...,1]=shape;result[...,2]=np.flip(color,axis=-1)
    elif task=='Single_Non-Atomic':result[...,0]=(shape+np.flip(shape,axis=-1))%4
    elif task=='Multiple_Non-Atomic':
        q=quadrant(xy);result[...,1]=(shape+q)%4
        result[...,2]=(np.flip(color,axis=-1)+(q if paper else np.flip(q,axis=-1)))%4
    else:raise ValueError(task)
    return result

def orbit(image):
    return min(hashlib.sha256(np.rot90(image,k).tobytes()).digest() for k in range(4))

def array_digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def make_split(hashes):
    parent=np.arange(64000)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    seen={}
    for i in range(64000):
        for role in range(2):
            key=hashes[i,role].tobytes()
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    groups={}
    for i in range(64000):groups.setdefault(find(i),[]).append(i)
    ordered=list(groups.values());permutation=np.random.default_rng(SEED).permutation(len(ordered));val=[];train=[]
    for j in permutation:
        g=ordered[j]
        if len(val)+len(g)<=6400:val.extend(g)
        else:train.extend(g)
    assert len(val)==6400 and len(train)==57600
    val=np.sort(np.array(val,np.int64));train=np.sort(np.array(train,np.int64))
    keys=lambda ids:{hashes[i,r].tobytes() for i in ids for r in range(2)}
    a,b,c=keys(train),keys(val),keys(range(64000,72000));assert not (a&b)
    return train,val,{'connected_components':len(groups),'largest_component':max(map(len,groups.values())),'train_val_orbit_overlap':0,'train_test_any_side_orbit_overlap':len(a&c),'val_test_any_side_orbit_overlap':len(b&c)}

def audit(task):
    freeze();out=BASE/'audits/dsprites_hard'/task
    if (out/'summary.json').exists():
        d=json.loads((out/'summary.json').read_text());assert d['protocol_sha256']==sha(BASE/'dsprites_audit_protocol.json')
        for n,h in d['files'].items():assert sha(out/n)==h
        return
    out.mkdir(parents=True,exist_ok=True);store=Store(task);started=time.perf_counter()
    source=np.zeros((72000,2,3),np.int16);target=np.zeros_like(source);centers=np.zeros((72000,2,2),np.float64)
    hashes=np.zeros((72000,2,32),np.uint8);changes=np.zeros(72000,np.int32);mse=np.zeros(72000,np.float64)
    errors=[];pixel_errors=[];identity_errors=[];mask_replays=[];vocab={s:{r:set() for r in ['source','target']} for s in ['train','test']};mask_counts=np.zeros((72000,2,2),np.int32)
    for index in range(72000):
        split='train' if index<64000 else 'test';i=index if index<64000 else index-64000
        objects=[store.metadata(split,i,role)['objects'] for role in ['source','target']];assert all(len(v)==2 for v in objects)
        source[index],target[index]=[factors(v) for v in objects]
        centers[index]=np.array([v['2d_coords'] for v in objects[0]])
        for j in range(2):
            assert objects[0][j]['2d_coords']==objects[1][j]['2d_coords']
            assert objects[0][j]['rotation']==objects[1][j]['rotation']==0
        expected=rule(source[index],centers[index],task)
        if not np.array_equal(expected,target[index]):errors.append({'split':split,'index':i,'source':source[index].tolist(),'target':target[index].tolist(),'expected':expected.tolist()})
        images=[]
        for ridx,role in enumerate(['source','target']):
            image=store.image(split,i,role);assert image.shape==(128,128,3) and image.dtype==np.uint8
            mask=np.asarray(Image.open(io.BytesIO(store.read(split,i,role+'_mask.png')[0])).convert('RGB'));assert mask.shape==image.shape
            hashes[index,ridx]=np.frombuffer(orbit(image),np.uint8);images.append(image)
            vocab[split][role].update(tuple(int(t) for t in v) for v in (source[index] if ridx==0 else target[index]))
            for obj in range(2):
                channel=0 if obj==0 else 2;other=2-channel
                interior=(mask[...,channel]==255)&(mask[...,other]==0)&(mask[...,1]==0)
                mask_counts[index,ridx,obj]=int(interior.sum())
                if interior.any():
                    expected_rgb=np.array(objects[ridx][obj]['color'][::-1],np.int16)
                    delta=np.abs(image[interior].astype(np.int16)-expected_rgb).max(axis=1)
                    median=np.median(image[interior],axis=0)
                    predicted=int(((np.array(COLORS)[:,::-1]-median)**2).sum(1).argmin())
                    if predicted!=COLORS.index(objects[ridx][obj]['color']):
                        identity_errors.append([split,i,role,obj,predicted,median.tolist()])
                        if role=='target':
                            from r5_renderer_diagnosis import render,compare
                            mask_replays.append({'split':split,'index':i,'object':obj,
                                'RGB_target_order':compare(image,render(objects[1])),
                                'mask_source_order':compare(mask,render(objects[1],sort_objects=objects[0],mask=True)),
                                'mask_target_order':compare(mask,render(objects[1],mask=True))})
                    # Lanczos ringing/overlap can perturb boundary-adjacent opaque pixels.
                    if np.median(delta)>2:pixel_errors.append([split,i,role,obj,float(np.median(delta))])
        different=np.any(images[0]!=images[1],axis=-1);changes[index]=different.sum();mse[index]=np.mean(((images[0].astype(float)-images[1])/255)**2)
        if (index+1)%4000==0:print('R5 decoded audit',task,index+1,'rule_errors',len(errors),'palette_errors',len(pixel_errors),flush=True)
    store.close()
    dump(out/'rule_errors.json',errors);dump(out/'palette_channel_deviations.json',pixel_errors);dump(out/'palette_identity_discrepancies.json',identity_errors);dump(out/'mask_order_replays.json',mask_replays)
    # Preserve complete observations even if semantic assumptions need an amendment.
    np.savez_compressed(out/'factors.npz',source=source,target=target,centers=centers,orbit_hashes=hashes,changed_pixels=changes,copy_mse=mse,opaque_object_pixels=mask_counts)
    assert not errors,'Distributed data disagrees with source rule; inspect before training'
    assert not [v for v in identity_errors if v[2]=='source'],'Source RGB palette identity disagrees; inspect before training'
    train,val,split_stats=make_split(hashes);np.savez_compressed(out/'partition.npz',train=train,val=val,test=np.arange(8000,dtype=np.int64))
    rotation=[];xy=centers.copy()
    for k in range(1,4):
        xy=np.stack([1-xy[...,1],xy[...,0]],-1);changed=np.any(rule(source,xy,task)!=target,axis=(1,2))
        rotation.append({'world_quarter_turn':k,'changed_semantic_target_fraction':float(changed.mean()),'n':72000})
    paper_mismatch=float(np.any(rule(source,centers,task,paper=True)!=target,axis=(1,2)).mean())
    groups={}
    for s,ids in [('train',train),('val',val),('official_test',np.arange(64000,72000))]:
        groups[s]={'n':len(ids),'changed_scenes':int((changes[ids]>0).sum()),'unchanged_scenes':int((changes[ids]==0).sum()),'copy_image_mse':float(mse[ids].mean()),'missing_opaque_mask_objects':int((mask_counts[ids]==0).sum())}
    v={s:{r:sorted(map(list,b)) for r,b in roles.items()} for s,roles in vocab.items()}
    result={'protocol_sha256':sha(BASE/'dsprites_audit_protocol.json'),'task':task,'all_metadata_pairs':72000,'all_RGB_and_mask_images_decoded':288000,'rule_errors':0,'source_palette_identity_errors':0,'target_mask_palette_discrepancies':len(identity_errors),'palette_channel_deviations':len(pixel_errors),
        'provided_target_masks_assumed_visibility_correct':False,'mask_order_replays':len(mask_replays),
        'groups':groups,'partition':split_stats,'vocabularies':v,'source_test_unseen_factor_bindings':len(vocab['test']['source']-vocab['train']['source']),
        'semantic_quarter_turn_audit':rotation,'literal_paper_self_quadrant_size_disagreement_fraction':paper_mismatch,'seconds':time.perf_counter()-started,
        'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},'scope':'Source-semantic and full-image orbit audit, no model performance. Hypothetical world rotation, not extra benchmark data.'}
    dump(out/'summary.json',result);print('R5 audit complete',task,groups,rotation,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS);a=p.parse_args()
    for task in ([a.task] if a.task else TASKS):audit(task)
