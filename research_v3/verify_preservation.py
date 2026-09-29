"""Read-only historical-file audit; all receipts stay in research_v3."""
import argparse,json,time
from pathlib import Path
from v3_common import ROOT,HERE,sha,dump

def main(output):
    plan=json.loads((HERE/'source_plan.json').read_text());assert sha(ROOT/plan['path'])==plan['sha256']
    assert sha(ROOT/'completion_v2/artifact_manifest.json')==plan['prior_manifest_sha256']
    assert sha(ROOT/'completion_v2/release/orbitlab-completion-v2-review.zip')==plan['prior_release_sha256']
    baseline=json.loads((ROOT/'completion_v2/original_integrity_start.json').read_text());known=json.loads((ROOT/'completion_v2/preexisting_manifest_differences.json').read_text())['differences'];cache={};results=[];began=time.time()
    specs=[(ROOT,ROOT/name,record['errors']) for name,record in baseline.items()]+[(ROOT/'completion_v2',ROOT/'completion_v2/artifact_manifest.json',[])]
    for base,mp,expected in specs:
        manifest=json.loads(mp.read_text());errors=[];total=0
        for i,row in enumerate(manifest['files']):
            p=(base/row['path']).resolve();assert p.is_relative_to(ROOT)
            if not p.is_file():errors.append({'path':row['path'],'reason':'missing'});continue
            if p not in cache:cache[p]=sha(p)
            if cache[p]!=row['sha256']:errors.append({'path':row['path'],'reason':'hash'})
            total+=p.stat().st_size
            if i and i%10000==0:print('Historical audit',mp.name,i,flush=True)
        assert errors==expected,(str(mp),errors)
        results.append({'manifest':str(mp.relative_to(ROOT)),'manifest_sha256':sha(mp),'checked':len(manifest['files']),'bytes':total,'preexisting_differences':errors,'no_new_drift':True})
    for name,record in known.items():assert sha(ROOT/name)==record['current_sha256']
    dest=HERE/output;assert dest.resolve().parent==HERE
    dump(dest,{'checked_unix':time.time(),'seconds':time.time()-began,'manifests':results,'unique_files_hashed':len(cache),'preexisting_four_hash_differences_preserved':True,'source_plan_manifest_release_zip_unchanged':True,'scope':'Files listed in the three original manifests; new v3 files and publication additions are outside those historical trees. Repeat before final goal closure.','verifier_sha256':sha(__file__)})
    print('Historical artifacts preserved',len(cache),'unique files',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='prior_integrity_latest.json');a=p.parse_args();main(a.output)
