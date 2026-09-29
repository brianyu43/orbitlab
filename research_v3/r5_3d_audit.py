"""Full decoded Single Atomic 3D audit before any 3D model training."""
import argparse,ast,io,json,time
from collections import Counter
import numpy as np
from PIL import Image
from v3_common import HERE,ROOT,sha,dump,lock
from r5_intake import BASE
from r5_3d_data import Pairs3D
from r5_audit import orbit,make_split

def schema(family):
    path=ROOT/'followup/external/svib/data_creation'/family/'macros.py'
    constants={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            constants[node.targets[0].id]=ast.literal_eval(node.value)
    names=['shape','color','size','material'] if family=='clevr' else ['shape','size','material']
    fields={'shape':'SHAPES','color':'COLORS','size':'SIZES','material':'MATERIALS'}
    return names,{name:constants[fields[name]] for name in names}

def freeze(family):
    upstream=ROOT/'followup/external/svib/data_creation'/family;data=BASE/'data'/f'{family}_hard'/'Single_Atomic'
    names,vocab=schema(family)
    return lock(BASE/f'{family}_audit_protocol.json',{'source_sha256':sha(__file__),'loader_source_sha256':sha(HERE/'r5_3d_data.py'),
        'intake_manifest_sha256':sha(data/'manifest.json'),'intake_verification_sha256':sha(data/'verification.json'),
        'upstream_sources':{n:sha(upstream/n) for n in ['macros.py','rule.py','create_data.py','utils.py']},'family':family,'task':'Single_Atomic',
        'native_pixels':'PIL convertRGB from original128x128RGBA without resize or recolor; count alpha-channel values; decode all RGB and masks.',
        'semantic_rule':'Swap the two objects shape labels; retain other attributes, world XY and rotation. World Z and projected pixel centers may change with object mesh height; record instead of assuming invariance.',
        'attribute_names':names,'vocabularies':vocab,'split_seed':890101,'train_n':57600,'val_n':6400,'test_n':8000,
        'family_policy':'Keep connected components sharing source OR target exact C4 RGB image-orbit hashes in one train/val split. Conservative duplicate detection, not a claim that 3D task commutes with image rotation.',
        'official_test':'Membership preserved, overlaps reported; no previous project training on these3Dtasks.',
        'mask_policy':'Record opaque dominant-red/green pixels of provided object masks as visibility diagnostics only. No primary model receives masks. No assumption that masks fully validate photometric visibility.',
        'changed_region':'Exact source-target RGB difference; report semantic-unchanged scenes and background/unchanged variation so rendering noise cannot be mistaken for a semantic edit.'})

def labels(objects,names,vocab):
    def index(name,value):
        if name=='color':return [list(x) for x in vocab[name]].index(list(value))
        return vocab[name].index(value)
    return np.array([[index(name,ob[name]) for name in names] for ob in objects],np.int16)

def audit(family):
    freeze(family);out=BASE/'audits'/f'{family}_hard'/'Single_Atomic'
    if (out/'summary.json').exists():
        saved=json.loads((out/'summary.json').read_text());assert saved['protocol_sha256']==sha(BASE/f'{family}_audit_protocol.json')
        for n,h in saved['files'].items():assert sha(out/n)==h
        return
    out.mkdir(parents=True,exist_ok=True);store=Pairs3D(family);names,vocab=schema(family);started=time.perf_counter()
    source=np.zeros((72000,2,len(names)),np.int16);target=np.zeros_like(source)
    centers=np.zeros((72000,2,2,2),np.float64);world=np.zeros((72000,2,2,3),np.float64)
    hashes=np.zeros((72000,2,32),np.uint8);changed=np.zeros(72000,np.int32);mse=np.zeros(72000,np.float64);mask_counts=np.zeros((72000,2,2),np.int32)
    alpha=Counter();errors=[];vocabularies={split:{role:set() for role in ('source','target')} for split in ('train','test')}
    for index in range(72000):
        split='train' if index<64000 else 'test';i=index%64000;metas=[store.metadata(split,i,role) for role in ('source','target')]
        assert all(len(m['objects'])==2 for m in metas)
        source[index],target[index]=[labels(m['objects'],names,vocab) for m in metas]
        expected=source[index].copy();expected[:,0]=expected[::-1,0]
        if not np.array_equal(expected,target[index]):errors.append({'split':split,'index':i,'expected':expected.tolist(),'target':target[index].tolist()})
        for field in ('Camera','Lamp_Key','Lamp_Fill','Lamp_Back'):
            if field in metas[0]:assert metas[0][field]==metas[1][field],(split,i,field)
        images=[]
        for r,role in enumerate(('source','target')):
            objects=metas[r]['objects'];centers[index,r]=[ob['pixel_coords'][:2] for ob in objects];world[index,r]=[ob['3d_coords'] for ob in objects]
            for ob in objects:assert ob['rotation']==0
            raw=Image.open(io.BytesIO(store.raw(split,i,role+'.png')));raw.load();assert raw.size==(128,128) and raw.mode=='RGBA'
            a=np.asarray(raw);alpha.update(dict(zip(*[v.tolist() for v in np.unique(a[:,:,3],return_counts=True)])))
            image=np.asarray(raw.convert('RGB'));hashes[index,r]=np.frombuffer(orbit(image),np.uint8);images.append(image)
            mask=np.asarray(Image.open(io.BytesIO(store.raw(split,i,role+'_mask.png'))).convert('RGB'));assert mask.shape==(128,128,3)
            for obj in range(2):
                other=[j for j in range(3) if j!=obj];mask_counts[index,r,obj]=int(((mask[:,:,obj]>=230)&(mask[:,:,other].max(-1)<=25)).sum())
            vocabularies[split][role].update(tuple(int(x) for x in row) for row in (source[index] if r==0 else target[index]))
        assert np.array_equal(world[index,0,:,:2],world[index,1,:,:2]),(split,i,'worldXY')
        assert np.isfinite(centers[index]).all() and np.isfinite(world[index]).all()
        changed[index]=np.any(images[0]!=images[1],axis=-1).sum();mse[index]=np.square((images[0].astype(np.float64)-images[1])/255).mean()
        if (index+1)%4000==0:print('R5 3Ddecoded',family,index+1,'rule_errors',len(errors),flush=True)
    store.close();dump(out/'rule_errors.json',errors)
    np.savez_compressed(out/'factors.npz',source=source,target=target,centers=centers,world_coords=world,orbit_hashes=hashes,changed_pixels=changed,copy_mse=mse,opaque_object_pixels=mask_counts)
    assert not errors,'Official metadata and inspected source differ; inspect before training'
    train,val,stats=make_split(hashes);np.savez_compressed(out/'partition.npz',train=train,val=val,test=np.arange(8000,dtype=np.int64))
    groups={}
    for split,ids in [('train',train),('val',val),('official_test',np.arange(64000,72000))]:
        unchanged=np.all(source[ids]==target[ids],axis=(1,2));u=ids[unchanged]
        groups[split]={'n':len(ids),'changed_scenes':int((changed[ids]>0).sum()),'semantic_unchanged_scenes':int(unchanged.sum()),
            'semantic_unchanged_with_pixel_changes':int((changed[u]>0).sum()),'semantic_unchanged_copy_MSE':float(mse[u].mean()) if len(u) else None,
            'copy_image_mse':float(mse[ids].mean()),'changed_pixel_fraction_quantiles':np.quantile(changed[ids]/16384,[0,.25,.5,.75,1]).tolist(),
            'provided_masks_without_opaque_pixels':int((mask_counts[ids]==0).sum())}
    center_delta=np.linalg.norm(centers[:,1]-centers[:,0],axis=-1)
    dump(out/'summary.json',{'protocol_sha256':sha(BASE/f'{family}_audit_protocol.json'),'family':family,'task':'Single_Atomic',
        'all_metadata_pairs':72000,'all_RGB_and_mask_images_decoded':288000,'rule_errors':0,'attribute_names':names,'vocabularies':vocab,
        'factor_bindings':{s:{r:sorted(map(list,v)) for r,v in roles.items()} for s,roles in vocabularies.items()},
        'source_test_unseen_factor_bindings':len(vocabularies['test']['source']-vocabularies['train']['source']),
        'groups':groups,'partition':stats,'RGB_alpha_counts':dict(alpha),'projected_center_change_max_px':float(center_delta.max()),'projected_center_changed_objects':int((center_delta>0).sum()),
        'seconds':time.perf_counter()-started,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'scope':'Data/provenance audit only; no model performance or physical C4 equivariance claim.'})
    print('R5 3Daudit complete',family,groups,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('family',choices=['clevr','clevrtex']);a=p.parse_args();audit(a.family)
