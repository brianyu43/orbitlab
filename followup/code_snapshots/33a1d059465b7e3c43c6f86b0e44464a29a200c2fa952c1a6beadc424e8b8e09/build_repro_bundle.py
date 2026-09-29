"""Package an immutable local snapshot, then reread every ZIP member."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.venv','__pycache__','.DS_Store','releases'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    if not args.name.replace('-','').replace('_','').isalnum():raise ValueError('Simple bundle name required')
    out=ROOT/'followup/releases'/args.name;out.mkdir(parents=True,exist_ok=False)
    path=out/(args.name+'.zip');started=time.time()
    original=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    names={x['path'] for x in original['files']}|{'RELEASE_MANIFEST.json'};excluded=[]
    for tree in ['planning','followup']:
        for directory,dirs,files in os.walk(ROOT/tree,followlinks=False):
            parent=Path(directory)
            dirs[:]=[x for x in dirs if x not in SKIP]
            for name in files:
                p=parent/name;relative=p.relative_to(ROOT).as_posix()
                if name in SKIP or name.endswith(('.pyc','.tmp','.lock')):
                    excluded.append(relative);continue
                if p.is_symlink():raise ValueError('Symlink requires explicit review: '+relative)
                names.add(relative)
    entries=[]
    progress=out/'build_progress.json'
    def update(status,**extra):
        progress.write_text(json.dumps({'status':status,'started_unix':started,'completed_files':len(entries),'expected_files':len(names),**extra},indent=2)+'\n')
    update('packing')
    try:
        with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for i,name in enumerate(sorted(names)):
                source=ROOT/name;before=source.stat();digest=hashlib.sha256();size=0
                with source.open('rb') as src,z.open('orbitlab/'+name,'w',force_zip64=True) as dst:
                    for chunk in iter(lambda:src.read(8*1024*1024),b''):
                        digest.update(chunk);size+=len(chunk);dst.write(chunk)
                after=source.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or size!=before.st_size:raise RuntimeError('File changed during snapshot: '+name)
                entries.append({'path':name,'bytes':size,'sha256':digest.hexdigest()})
                if i%500==0:update('packing');print('packed',i+1,'/',len(names),flush=True)
            manifest={'schema_version':1,'name':args.name,'created_unix':time.time(),'origin_root':str(ROOT),
                'purpose':'Private local reproducibility snapshot. No external publication or new redistribution license.',
                'file_count':len(entries),'total_bytes':sum(x['bytes'] for x in entries),'files':entries,
                'excluded_directory_names':sorted(SKIP),'excluded_files':excluded,
                'completion_boundary':'A04 actual human responses remain pending unless the enclosed scope audit establishes otherwise. Packaging is not evidence of scientific success or full retraining.'}
            body=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
            z.writestr('orbitlab/BUNDLE_MANIFEST.json',body)
        (out/'BUNDLE_MANIFEST.json').write_bytes(body)
        update('verifying_zip')
        with zipfile.ZipFile(path) as z:
            assert len(z.infolist())==len(entries)+1
            for item in entries:
                h=hashlib.sha256();n=0
                with z.open('orbitlab/'+item['path']) as f:
                    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk);n+=len(chunk)
                assert n==item['bytes'] and h.hexdigest()==item['sha256'],item['path']
            assert z.read('orbitlab/BUNDLE_MANIFEST.json')==body
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        receipt={'all_passed':True,'zip_path':str(path.relative_to(ROOT)),'zip_bytes':path.stat().st_size,'zip_sha256':h.hexdigest(),
            'file_count':len(entries),'uncompressed_bytes':manifest['total_bytes'],'all_zip_members_reread':True,
            'manifest_sha256':hashlib.sha256(body).hexdigest(),'finished_unix':time.time(),'scope':'Package bytes fully verified; extracted-directory replay remains separate.'}
        (out/'archive_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');update('complete',receipt=receipt)
        print(json.dumps(receipt),flush=True)
    except Exception as e:
        update('failed',error=repr(e));raise


if __name__=='__main__':main()
