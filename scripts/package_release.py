"""Create a local self-contained evidence archive, excluding runtime and caches."""
import hashlib, json, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'deliverables/OrbitLab_15sessions_2026-09-23.zip'
def sha(data): return hashlib.sha256(data).hexdigest()

def main():
    if TARGET.exists(): raise FileExistsError('Preserve the existing release; use a new filename for a revised release.')
    files=[]
    for folder in ['sources','data','configs','work','scripts','tests','reports','runs','deliverables']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or p.is_symlink():continue
            if any(x in ['__pycache__','.DS_Store'] for x in p.parts) or p.suffix=='.pyc':continue
            if p==TARGET or p.name=='release_archive_verification.json':continue
            files.append(p)
    files += list(ROOT.glob('*.md'))+[ROOT/'requirements.lock.txt']
    files += [ROOT/'.slides-build'/f for f in ['build.mjs','render_final.mjs','validation.json']]
    files=sorted(set(files))
    records=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in files]
    manifest={'schema_version':1,'purpose':'Originals, exact split, code, settings, checkpoint/results and presentation; no venv or external runtime.','file_count':len(records),'total_bytes':sum(r['bytes'] for r in records),'files':records,'exclusions':['.venv','.matplotlib','node_modules','temporary presentation candidates','release archive and its external verification receipt']}
    manifest_path=ROOT/'RELEASE_MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    with zipfile.ZipFile(TARGET,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[manifest_path]:z.write(p,Path('orbitlab')/p.relative_to(ROOT))
    with zipfile.ZipFile(TARGET) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(files)+1
        for rec in records:
            data=z.read('orbitlab/'+rec['path'])
            assert len(data)==rec['bytes'] and sha(data)==rec['sha256'],rec['path']
    receipt={'archive':str(TARGET.relative_to(ROOT)),'archive_sha256':sha(TARGET.read_bytes()),'archive_bytes':TARGET.stat().st_size,'checked_files':len(records),'manifest_sha256':sha(manifest_path.read_bytes()),'zip_crc_passed':True,'all_content_hashes_passed':True}
    (ROOT/'reports/release_archive_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
