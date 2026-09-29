"""Final finite-scope closure; fails on missing experiments or new historical drift."""
from pathlib import Path
import json,time,ast
from svib_prepare import sha
B=Path(__file__).resolve().parent;ROOT=B.parent
r=json.loads((B/'aggregate.json').read_text());assert r['all_experiments_and_primary_verification_complete']
assert (B/'verification_driver_completed.json').exists()
assert json.loads((B/'new_data_verification.json').read_text())['all_passed']
reconstruction=json.loads((B/'generation_reconstruction/summary.json').read_text())
assert len(reconstruction['units'])==18 and reconstruction['rows']=={'standard':18432,'area_edge':18432}
assert len(list((B/'generation_reconstruction').glob('*_verification.json')))==18
for path in B.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
assert len(list((B/'dynamics/evaluation').glob('d*/*/*/*/geometry.json')))==1422
assert len(list((B/'dynamics/evaluation').glob('d*/*/*/*/counterfactual_verification.json')))==432
counts={}
for branch,pattern,expected in [('generation','runs/d*/*/run.json',50),('perception','runs/*/run.json',6),('perception_detail','runs/*/run.json',6),('dynamics','runs/d*/*/run.json',30),('svib','runs/*/run.json',6)]:
    pp=list((B/branch).glob(pattern));assert len(pp)==expected,(branch,len(pp));counts[branch]=len(pp)
    for p in pp:
        a=json.loads(p.read_text());assert a['steps'] in [4000,6000,10000,16000]
        if branch=='generation':assert a.get('status')!='budget_limited'
for seed in range(3):
    a=json.loads((B/f'svib/runs/plain_s{seed}/run.json').read_text());b=json.loads((B/f'svib/runs/c4_s{seed}/run.json').read_text())
    assert a['order_sha256']==b['order_sha256'] and a['parameters']==b['parameters']
start=json.loads((B/'original_integrity_start.json').read_text());history={}
for name,baseline in start.items():
    manifest=json.loads((ROOT/name).read_text());errors=[]
    for item in manifest['files']:
        p=ROOT/item['path']
        if not p.is_file():errors.append({'path':item['path'],'reason':'missing'})
        elif sha(p)!=item['sha256']:errors.append({'path':item['path'],'reason':'hash'})
    assert errors==baseline['errors'],(name,errors)
    history[name]={'checked':len(manifest['files']),'errors':errors,'unchanged_from_start':True}
for name,a in json.loads((B/'preexisting_manifest_differences.json').read_text())['differences'].items():assert sha(ROOT/name)==a['current_sha256']
(B/'original_integrity_final.json').write_text(json.dumps(history,indent=2)+'\n')
result={'completed_unix':time.time(),'experiment_status':'complete','capability_status':'mixed; unresolved failures explicitly reported',
        'counts':r['counts'],'new_trained_components':counts,'total_trained_components':sum(counts.values()),
        'development_dynamics_units':18,'counterfactual_replay_units':432,'geometry_units':1422,
        'posthoc_decoder_reconstruction_units':18,'posthoc_decoder_reconstruction_rows':36864,
        'historical_files_unchanged_from_start':True,'new_independent_human_evaluation':False,
        'svib_scope':'One official full split, not all tasks; five epochs, not published baseline reproduction',
        'generation_selection':'Development diversity gate failed; candidate retained as diagnostic replication',
        'perception_correction':'Initial raster-symmetry assumption corrected in explicit amendment and fresh full-pose follow-up',
        'perception_initial_comparison_limit':'Global/crop jointly changes alignment, spatial sampling and effective pixel coordinate-loss weighting. Binary/alpha follow-up matches these factors.',
        'next_research_status':'Concrete plan documented; future stages not executed or counted as complete',
        'verification_scope':'Artifact checks and explicit replay samples; not independent full retraining or clean install',
        'source_sha256':sha(Path(__file__))}
(B/'scope_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
