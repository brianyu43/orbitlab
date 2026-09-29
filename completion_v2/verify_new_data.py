"""Recompute image orbit hashes and check manifest/physical-family integrity."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parent;ROOT=B.parent
sys.path[:0]=[str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
from svib_prepare import sha
import p4_prepare as prep
import object_edit_confirmation as edit
import dynamics_world as world

def orbit(im):return min(hashlib.sha256(np.ascontiguousarray(np.rot90(im,k)).tobytes()).hexdigest() for k in range(4))
checks=0;prior=prep.prior_orbits()
for p in (ROOT/'generation_recovery/confirmation_v1/data').rglob('*metadata.json'):prior.update(v['orbit_sha256'] for v in json.loads(p.read_text()))
seen=set();g=json.loads((B/'generation/data/manifest.json').read_text())
for v in g['files']:assert sha(B/'generation/data'/v['path'])==v['sha256'];checks+=1
for p in sorted((B/'generation/data').glob('d*/*_metadata.json')):
    meta=json.loads(p.read_text());a=np.load(p.with_name(p.name.replace('_metadata.json','.npz')))['images'];assert len(a)==len(meta)
    for im,row in zip(a,meta):
        h=orbit(im);assert h==row['orbit_sha256'] and h not in seen and h not in prior;seen.add(h)
gcount=len(seen);assert gcount==44032
prior=edit.previous_orbits()
for folder in (ROOT/'followup/data/object_image_edit_v1').iterdir():
    if folder.is_dir():
        prior.update(v['orbit_sha256'] for v in json.loads((folder/'scenes.json').read_text()))
        prior.update(v['target_orbit_sha256'] for v in json.loads((folder/'tasks.json').read_text()))
owners={};scenes=0;tasks=0
for branch in ['perception','perception_detail']:
    for p in sorted((B/branch/'data').glob('d*/manifest.json')):
        m=json.loads(p.read_text())
        for v in m['files']:assert sha(p.parent/v['path'])==v['sha256'];checks+=1
        for split in m['splits']:
            data=p.parent/split;records=json.loads((data/'scenes.json').read_text());commands=json.loads((data/'tasks.json').read_text());images=np.load(data/'rgb.npz')['images'];targets=np.load(data/'targets.npz')['images']
            for im,row in zip(images,records):
                h=orbit(im);family=(branch,p.parent.name,split,row['source_index']);assert h==row['orbit_sha256'] and h not in prior
                assert h not in owners or owners[h]==family;owners[h]=family;scenes+=1
            for im,row in zip(targets,commands):
                h=orbit(im);family=(branch,p.parent.name,split,row['source_index']);assert h==row['target_orbit_sha256'] and h not in prior
                assert h not in owners or owners[h]==family;owners[h]=family;tasks+=1
oldphysical=set();oldimages=set()
for root in [ROOT/'followup/data/dynamics_world_v1',ROOT/'followup/data/dynamics_repeats_v1']:
    for p in root.rglob('records.json'):
        for row in json.loads(p.read_text()):oldphysical.add(row['initial_orbit_sha256']);oldimages.add(row['initial_image_orbit_sha256'])
physical=set();images_seen=set();n=0
for row in json.loads((B/'dynamics/data/manifest.json').read_text())['groups']:
    p=B/'dynamics'/row['path'];assert sha(p)==row['archive_sha256'];checks+=1
    a=np.load(p);records=json.loads(p.with_name('records.json').read_text())
    for i,record in enumerate(records):
        count=int(a['n_objects'][i]);h=world.initial_orbit_hash(a['states'][i,0,:count],a['contexts'][i],a['actions'][i,3,:count]);im=orbit(a['observations'][i,0])
        assert h==record['physical'] and im==record['image'] and h not in oldphysical and im not in oldimages
        assert h not in physical and im not in images_seen;physical.add(h);images_seen.add(im);n+=1
result={'all_passed':True,'manifest_files_checked':checks,'generation_image_orbits_recomputed':gcount,'perception_source_orbits_recomputed':scenes,'perception_target_orbits_recomputed':tasks,'dynamics_initial_physical_and_image_orbits_recomputed':n,'cross_family_and_prior_overlap':0,'scope':'C4 image/physical orbits checked; repeated target variants within the same source family are permitted. This does not assert semantic novelty beyond the specified orbit relation.','source_sha256':sha(Path(__file__))}
(B/'new_data_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
