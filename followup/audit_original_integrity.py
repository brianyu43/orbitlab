"""Read-only check that every artifact in the original release remains unchanged."""
import argparse
import json
import time
from pathlib import Path
from common import HERE,ROOT,dump,r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=(HERE/args.output).resolve()
    assert output.is_relative_to(HERE/'reports') and not output.exists()
    manifest_path=ROOT/'RELEASE_MANIFEST.json';manifest=json.loads(manifest_path.read_text())
    initial=json.loads((HERE/'reports/original_integrity_start.json').read_text())
    assert initial['all_match'] and initial['manifest_sha256']==r.sha(manifest_path)
    assert len(manifest['files'])==manifest['file_count']==1324
    results=[];bad=[]
    for item in manifest['files']:
        path=ROOT/item['path'];assert path.resolve().is_relative_to(ROOT)
        exists=path.is_file();actual_hash=r.sha(path) if exists else None;actual_bytes=path.stat().st_size if exists else None
        passed=exists and actual_hash==item['sha256'] and actual_bytes==item['bytes']
        results.append({'path':item['path'],'sha256':actual_hash,'bytes':actual_bytes,'matches':passed})
        if not passed:bad.append(item['path'])
    assert sum(x['bytes'] for x in manifest['files'])==manifest['total_bytes']
    dump(output,{'checked_unix':time.time(),'all_match':not bad,'manifest_sha256':r.sha(manifest_path),
        'initial_integrity_receipt_sha256':r.sha(HERE/'reports/original_integrity_start.json'),
        'checked_files':len(results),'expected_total_bytes':manifest['total_bytes'],'mismatches':bad,'results':results,
        'source_sha256':r.sha(__file__),'scope':'Original release files only, including all listed source/config/data/run/report artifacts. No original files modified; this does not certify the new follow-up release.'})
    print('original integrity',len(results),'files',manifest['total_bytes'],'bytes, mismatches',len(bad),flush=True)
    if bad:raise RuntimeError('Original artifacts differ: '+str(bad))


if __name__=='__main__':main()
