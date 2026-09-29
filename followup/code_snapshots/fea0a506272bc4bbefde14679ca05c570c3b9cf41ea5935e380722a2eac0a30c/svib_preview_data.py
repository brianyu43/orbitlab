"""Image-only SVIB preview adapter, preserving original PNG channels and folds."""
import collections
import json
import subprocess
import time
import numpy as np
from PIL import Image
from common import HERE,dump,r

NAME='svib_preview_shape_swap_v1'
ALPHAS=['0.0','0.2','0.4','0.6']
SOURCE=HERE/'external/svib-samples/samples/dSprites/Single_Atomic'


def alpha_name(alpha):return 'alpha_'+alpha.replace('.','p')


def freeze():
    intake=HERE/'reports/svib_intake_v1/manifest.json';original=json.loads(intake.read_text());assert len(original['files'])==2000
    for name,sha in original['commits'].items():
        actual=subprocess.check_output(['git','-C',str(HERE/'external'/name),'rev-parse','HEAD'],text=True).strip();assert actual==sha
    for item in original['files']:assert r.sha(SOURCE/item['path'])==item['sha256'],item['path']
    path=HERE/f'configs/{NAME}.json'
    if not path.exists():
        dump(path,{'alphas':ALPHAS,'train_per_alpha':70,'val_per_alpha':15,'id_test_per_alpha':15,'external_test':100,
            'resolution':[128,128],'input':'Unmodified PIL RGB source PNG only; float conversion divides by 255.',
            'split':'Sorted episode directories, first 70 train / next 15 validation / last 15 ID test; official preview Test is shared across alpha.',
            'intake_sha256':r.sha(intake),'commits':original['commits'],
            'sources':{f:r.sha(HERE/f) for f in ['svib_preview_data.py','planning_B06_SVIB_PREVIEW_KO.md',
                'external/svib/tasks/image_to_image/data.py','external/svib/tasks/image_to_image/eval.py','external/svib/data_creation/dsprites/create_data.py']},
            'frozen_unix':time.time(),'scope':'500 published preview pairs, no full SVIB benchmark claim or source checkpoint transfer.'})
    c=json.loads(path.read_text());assert c['intake_sha256']==r.sha(intake)
    for file,sha in c['sources'].items():assert r.sha(HERE/file)==sha,file
    return c


def binding(obj):return (obj['shape'],tuple(obj['color']),obj['size'])


def read_pair(folder):
    source=np.asarray(Image.open(folder/'source.png').convert('RGB')).copy()
    target=np.asarray(Image.open(folder/'target.png').convert('RGB')).copy();assert source.shape==target.shape==(128,128,3)
    s=json.loads((folder/'source.json').read_text());t=json.loads((folder/'target.json').read_text())
    assert len(s['objects'])==len(t['objects'])==2
    assert [x['shape'] for x in s['objects']][::-1]==[x['shape'] for x in t['objects']]
    for x,y in zip(s['objects'],t['objects']):
        for key in ['color','size','rotation','2d_coords']:assert x[key]==y[key]
    return source,target,s,t


def main():
    c=freeze();out=HERE/f'data/{NAME}';report=HERE/f'reports/{NAME}';groups=[];all_records=[];bindings={}
    tasks=[('heldout','Test',sorted(p for p in (SOURCE/'Test').iterdir() if p.is_dir()))]
    for alpha in ALPHAS:
        folders=sorted(p for p in (SOURCE/f'Train/alpha-{alpha}').iterdir() if p.is_dir());assert len(folders)==100
        for split,part in [('train',folders[:70]),('val',folders[70:85]),('test_id',folders[85:])]:tasks.append((f'{alpha_name(alpha)}/{split}',f'Train/alpha-{alpha}',part))
    assert len(tasks[0][2])==100
    for address,source_split,folders in tasks:
        destination=out/address
        if destination.exists():raise RuntimeError(f'Existing preview adapter output retained: {destination}')
        destination.mkdir(parents=True);images=[];targets=[];records=[];bs=set();bt=set()
        for i,folder in enumerate(folders):
            source,target,s,t=read_pair(folder);images.append(source);targets.append(target)
            bs.update(binding(x) for x in s['objects']);bt.update(binding(x) for x in t['objects'])
            record={'index':i,'source_episode':str(folder.relative_to(SOURCE)),'source_split':source_split,
                'source_image_sha256':r.sha(folder/'source.png'),'target_image_sha256':r.sha(folder/'target.png'),
                'source_orbit_sha256':r.orbit_hash(source),'target_orbit_sha256':r.orbit_hash(target),
                'identical_pair':bool(np.array_equal(source,target)),'source_objects':s['objects'],'target_objects':t['objects']}
            records.append(record);all_records.append({'adapter_split':address,**record})
        images=np.stack(images);targets=np.stack(targets)
        changed=np.any(images!=targets,axis=-1);foreground=np.any(images>0,axis=-1)|np.any(targets>0,axis=-1)
        np.savez_compressed(destination/'inputs.npz',source_rgb=images)
        np.savez_compressed(destination/'labels.npz',target_rgb=targets,changed_mask=changed,foreground_mask=foreground)
        dump(destination/'episodes.json',records);bindings[address]={'source':bs,'target':bt}
        groups.append({'address':address,'episodes':len(records),'identical_pairs':sum(x['identical_pair'] for x in records),
            'files':{str(p.relative_to(HERE)):r.sha(p) for p in destination.iterdir() if p.is_file()}})
    assert len(all_records)==500
    # Actual image/orbit overlap is distinct from sharing factor combinations.
    overlaps=[]
    for alpha in ALPHAS:
        train=[x for x in all_records if x['adapter_split']==f'{alpha_name(alpha)}/train']
        for split in [f'{alpha_name(alpha)}/val',f'{alpha_name(alpha)}/test_id','heldout']:
            evaluation=[x for x in all_records if x['adapter_split']==split]
            for a in ['source','target']:
                for b in ['source','target']:
                    overlap={x[f'{a}_orbit_sha256'] for x in train}&{x[f'{b}_orbit_sha256'] for x in evaluation}
                    overlaps.append({'alpha':alpha,'evaluation':split,'train_side':a,'evaluation_side':b,'rotation_orbit_overlap':len(overlap)})
    assert not any(x['rotation_orbit_overlap'] for x in overlaps),'Train/heldout image orbit overlap; preserve output and review protocol before training.'
    exposure={address:{'source_bindings':len(value['source']),'target_bindings':len(value['target']),
        'source_overlap_external_input_bindings':len(value['source']&bindings['heldout']['source']),
        'target_overlap_external_input_bindings':len(value['target']&bindings['heldout']['source'])} for address,value in bindings.items()}
    dump(out/'manifest.json',{'groups':groups,'episodes':500,'alpha_sources':ALPHAS,'image_orbit_overlap_checks':overlaps,
        'binding_exposure':exposure,'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),'intake_sha256':c['intake_sha256'],
        'scope':c['scope']})
    dump(report/'data_preparation.json',{'prepared':True,'groups':13,'episodes':500,'train_episodes_per_alpha':70,
        'shared_external_test_episodes':100,'masks_only_in_labels':True,'cross_split_train_image_orbit_overlap':False,
        'manifest_sha256':r.sha(out/'manifest.json'),'source_sha256':r.sha(__file__)})
    print('SVIB preview adapter prepared 500 pairs in 13 partitions; independent audit required.',flush=True)


if __name__=='__main__':main()
