"""Local review bundle; inventories all artifacts but omits bulk replay/data arrays."""
from pathlib import Path
import json,zipfile,time
from svib_prepare import sha
B=Path(__file__).resolve().parent
assert json.loads((B/'scope_audit.json').read_text())['experiment_status']=='complete'
release=B/'release';release.mkdir(exist_ok=True)
files=[]
for p in sorted(B.rglob('*')):
    rel=p.relative_to(B)
    if not p.is_file() or rel.parts[0]=='release' or '__pycache__' in rel.parts or p.name=='artifact_manifest.json' or p.suffix=='.tmp':continue
    files.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p)})
manifest={'created_unix':time.time(),'root':str(B),'files':files,'file_count':len(files),'total_bytes':sum(v['bytes'] for v in files),'exclusions':['release/','__pycache__/','artifact_manifest.json','*.tmp'],'scope':'Inventory of the local completion directory; review ZIP is a subset, not a clean-install reproduction package.'}
(B/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
selected=[]
for item in files:
    p=Path(item['path'])
    if p.suffix in ['.py','.md','.json','.png','.pdf'] or p.name in ['model.pt','checkpoint.pt','ae.pt','decoder.pt','replay_samples.pt']:selected.append(item)
selected.append({'path':'artifact_manifest.json','bytes':(B/'artifact_manifest.json').stat().st_size,'sha256':sha(B/'artifact_manifest.json')})
archive=release/'orbitlab-completion-v2-review.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for item in selected:z.write(B/item['path'],'completion_v2/'+item['path'])
with zipfile.ZipFile(archive) as z:
    assert len(z.infolist())==len(selected)
    import hashlib
    for item in selected:assert hashlib.sha256(z.read('completion_v2/'+item['path'])).hexdigest()==item['sha256']
receipt={'archive':archive.name,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'verified_members':len(selected),
         'selection':'All Python/Markdown/JSON/PNG/PDF and final model/checkpoint/replay-sample PT files. Bulk data arrays, full prediction tensors, progress files, logs and downloaded archive remain local and are inventoried but omitted.',
         'not_standalone':True,'historical_dependencies_required':True,'manifest_sha256':sha(B/'artifact_manifest.json')}
(release/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
