"""Fresh single-object families and paired generation noise for R4.

No old generator, dataset, threshold or diversity-reader file is modified.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from v3_common import ROOT,HERE,sha,dump,lock,event,status
import orbitlab as o
import research as r
BASE=HERE/'r4';DEVELOPMENT=889101;CONFIRMATION=[889201,889202,889203]
SPLITS={'train':1024,'val':256,'test':256,'ood':256}
THRESHOLD=ROOT/'followup/reports/evaluator_calibration_v1/locked_threshold.json'
READER=ROOT/'generation_recovery/reports/p0_diversity_reader_lock_v1.json'

def freeze():
    return lock(BASE/'data_protocol.json',{'version':1,'source_sha256':sha(__file__),'renderer_sha256':sha(ROOT/'work/orbitlab.py'),'orbit_hash_source_sha256':sha(ROOT/'work/research.py'),
        'development':DEVELOPMENT,'conditional_confirmation':CONFIRMATION,'split_sizes':SPLITS,'image_size':64,'latent_generation_noise_shape':[24*128,4,16],
        'noise_seed_rule':'data_seed+7200000; same stored float32normal noise for every decoder/representation and initialization','samples_per_pair':128,'reference_per_pair':384,
        'reference_seed_rule':'data_seed+10000; all24shape/color pairs;384distinct scene families each, quarter turns cycling per pair','split_rule':'C4pixel-orbit family disjointness from all inspected historical single-object metadata and all new R4splits/reference/seed families. No selection by model or evaluator score.',
        'small_arrow':'True renderer kind2 and radius<(.13+.11/3). Rotation variants remain one base scene. Generated scale is an evaluator estimate, never a true renderer parameter.',
        'evaluation_separation':'Development usesval+ood; test prepared but not used for development selection. Confirmation uses newseed test+ood after candidate locking. References supply descriptive diversity only, never training inputs.',
        'fixed_strict_threshold_sha256':sha(THRESHOLD),'fixed_diversity_reader_sha256':sha(READER),'caps':{'stage_seconds':43200,'stage_bytes':30000000000}})

def historical():
    paths=[]
    for p in [ROOT/'data',ROOT/'followup/data',ROOT/'generation_recovery/data_v1',ROOT/'generation_recovery/confirmation_v1/data',ROOT/'completion_v2/generation/data']:
        paths.extend(p.rglob('*metadata*.json'))
    seen=set();sources=[]
    for path in sorted(set(paths)):
        rows=json.loads(path.read_text())
        if isinstance(rows,list):
            values=[v['orbit_sha256'] for v in rows if isinstance(v,dict) and 'orbit_sha256' in v]
            if values:seen.update(values);sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'rows':len(values)})
    return seen,sources

def prepare(seed):
    freeze();dest=BASE/f'data/d{seed}';dest.mkdir(parents=True,exist_ok=True);mp=dest/'manifest.json'
    if mp.exists():
        m=json.loads(mp.read_text())
        for name,h in m['files'].items():assert sha(dest/name)==h
        return
    assert not list(dest.iterdir()),'Partial new data must be retained and investigated'
    seen,sources=historical();prior=len(seen)
    for path in sorted((BASE/'data').glob('d*/*metadata.json')):
        seen.update(v['orbit_sha256'] for v in json.loads(path.read_text()))
    earlier=len(seen);counts={}
    for split,n in SPLITS.items():
        images=[];labels=[];metadata=[];candidate=0
        while len(images)<n:
            image,label,info=o.render_base(candidate,seed,64,split);candidate+=1;key=r.orbit_hash(image)
            if key in seen:continue
            seen.add(key);images.append(image);labels.append(label);metadata.append({**info,'rotation':0,'orbit_sha256':key})
        np.savez_compressed(dest/f'{split}.npz',images=np.stack(images),labels=np.array(labels,np.int64));dump(dest/f'{split}_metadata.json',metadata)
        counts[split]={'scenes':n,'candidates':candidate,'small_arrows':sum(m['kind']==2 and m['radius']<.13+.11/3 for m in metadata)}
        print('R4data',seed,split,n,flush=True)
    buckets={(k,c):[] for k in range(4) for c in range(6)};candidate=0
    while min(map(len,buckets.values()))<384:
        split='ood' if candidate%6==0 else 'val';image,label,info=o.render_base(candidate,seed+10000,64,split);candidate+=1
        if len(buckets[label])>=384:continue
        key=r.orbit_hash(image)
        if key in seen:continue
        seen.add(key);buckets[label].append((image,{**info,'orbit_sha256':key}))
    images=[];labels=[];metadata=[]
    for i in range(384):
        for k in range(4):
            for color in range(6):
                image,info=buckets[k,color][i];images.append(np.rot90(image,i%4).copy());labels.append([k,color]);metadata.append({**info,'rotation':i%4})
    np.savez_compressed(dest/'reference.npz',images=np.stack(images),labels=np.array(labels,np.int64));dump(dest/'reference_metadata.json',metadata)
    labels=np.array([[k,c] for _ in range(128) for k in range(4) for c in range(6)],np.int64);noise=torch.randn(len(labels),4,16,generator=torch.Generator().manual_seed(seed+7200000)).numpy()
    np.savez_compressed(dest/'generation_noise.npz',noise=noise,labels=labels)
    assert len(seen)-earlier==sum(SPLITS.values())+9216
    files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    dump(mp,{'seed':seed,'protocol_sha256':sha(BASE/'data_protocol.json'),'files':files,'counts':counts,'reference_scenes':9216,'reference_candidates':candidate,'historical_family_count':prior,'prior_sources':sources,'new_scene_families':len(seen)-earlier,'confirmation_started':False})
    event('R4_data_prepared',seed=seed);status('R4','development_data_prepared_models_not_trained',data=f'r4/data/d{seed}',note='Preparation while final R2pixel replays finish; no R4training or confirmation yet.')

def verify(seed):
    freeze();folder=BASE/f'data/d{seed}';m=json.loads((folder/'manifest.json').read_text());old,sources=historical();assert sources==m['prior_sources'];seen=set(old);checks=[]
    for name,digest in m['files'].items():assert sha(folder/name)==digest
    for split in [*SPLITS,'reference']:
        a=np.load(folder/f'{split}.npz');images=a['images'];labels=a['labels'];metadata=json.loads((folder/f'{split}_metadata.json').read_text())
        for image,label,row in zip(images,labels,metadata):
            ss,original_split,index=row['base_id'].split(':');base,y,info=o.render_base(int(index),int(ss),64,original_split)
            np.testing.assert_array_equal(image,np.rot90(base,row['rotation']));np.testing.assert_array_equal(label,y)
            for k,v in info.items():assert row[k]==v
            key=r.orbit_hash(image);assert key==row['orbit_sha256'] and key not in seen;seen.add(key)
        checks.append({'split':split,'all_images_and_labels_replayed':len(images),'metadata_sha256':sha(folder/f'{split}_metadata.json')});print('R4verified',seed,split,len(images),flush=True)
    expected=torch.randn(24*128,4,16,generator=torch.Generator().manual_seed(seed+7200000)).numpy();noise=np.load(folder/'generation_noise.npz');np.testing.assert_array_equal(noise['noise'],expected)
    np.testing.assert_array_equal(noise['labels'],np.array([[k,c] for _ in range(128) for k in range(4) for c in range(6)]))
    assert len(seen)-len(old)==11008
    dump(folder/'verification.json',{'all_rendered_images':11008,'groups':checks,'new_families_disjoint_from_prior':True,'paired_noise_fully_replayed':True,'source_sha256':sha(__file__),'manifest_sha256':sha(folder/'manifest.json'),'model_results_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=DEVELOPMENT);p.add_argument('--verify',action='store_true');a=p.parse_args();verify(a.seed) if a.verify else prepare(a.seed)
