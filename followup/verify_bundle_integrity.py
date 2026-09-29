"""Dependency-free verification of an extracted OrbitLab research bundle."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def check(root):
    root=Path(root).resolve();manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text())
    errors=[];size=0;seen=set()
    for item in manifest['files']:
        name=item['path'];p=root/name
        if name in seen or not p.resolve().is_relative_to(root) or p.is_symlink():raise ValueError('Unsafe or duplicate member: '+name)
        seen.add(name)
        if not p.is_file():errors.append({'path':name,'reason':'missing'});continue
        n=p.stat().st_size;size+=n
        if n!=item['bytes'] or sha(p)!=item['sha256']:errors.append({'path':name,'reason':'size or SHA-256 mismatch'})
    if len(seen)!=manifest['file_count'] or size!=manifest['total_bytes']:errors.append({'reason':'count or total size mismatch'})
    result={'all_passed':not errors,'files_checked':len(seen),'bytes_checked':size,'errors':errors,
            'manifest_sha256':sha(root/'BUNDLE_MANIFEST.json'),'scope':'All listed package bytes; not training or scientific validity.'}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=check(a.root)
    if a.output:
        with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='errors'}))
    if not result['all_passed']:raise SystemExit(1)


if __name__=='__main__':main()
