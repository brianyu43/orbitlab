"""Cross-check final evidence against the original 15-session requirements."""
import csv,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
failures=[];checks={}
def check(name,value):
 checks[name]=bool(value)
 if not value:failures.append(name)
manifest=json.loads((ROOT/'reports/source_manifest.json').read_text())
for item in manifest['files']:
 p=ROOT/'sources/originals'/item['name'];check('source '+item['name'],hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'])
for name in ['original_doctor_mps','original_smoke_mps']:
 d=json.loads((ROOT/f'reports/{name}.json').read_text());check(name,d.get('passed',d.get('forward_parity_pass',False)) and d.get('gradient_parity_pass',True))
for specname in ['experiment_matrix','generation_ae_repair']:
 spec=json.loads((ROOT/f'configs/{specname}.json').read_text())
 for t in spec['tasks']:
  d=ROOT/t['out'];marker=json.loads((d/'COMPLETE.json').read_text());check('training '+t['id'],marker['task']==t and all(hashlib.sha256((d/f).read_bytes()).hexdigest()==h for f,h in marker['evidence'].items()))
  if specname=='experiment_matrix':
   check('analysis '+t['id'],(d/'analysis/COMPLETE.json').exists())
   for split in ['train','val','test','ood']:
    rows=list(csv.DictReader((d/f'analysis/{split}_samples.csv').open()));check(t['id']+' '+split+' samples',len(rows)==1024 and len({r['base_id'] for r in rows})==256)
   if t['family'] in ['primary','diagnostic']:check('probes '+t['id'],(d/'analysis/probes.json').exists())
gates=json.loads((ROOT/'reports/quality_gate_repair.json').read_text());check('numeric AE qualification',all(g['numeric_pass'] for g in gates.values()));check('visual AE qualification',json.loads((ROOT/'reports/repair_visual_review.json').read_text())['passed'])
gen=json.loads((ROOT/'reports/generation_status.json').read_text());check('generation eight runs',len(gen['completed'])==8 and gen['current'] is None)
noise={}
for name in gen['completed']:
 out=ROOT/'runs'/name;run=json.loads((out/'run.json').read_text());m=json.loads((out/'samples/metrics.json').read_text());check(name+' AE frozen',run['ae_frozen_and_unchanged']);noise[name]=m['noise_sha256']
 for steps in [0,16,32,64]:
  rows=list(csv.DictReader((out/f'samples/samples_{steps}.csv').open()));check(f'{name} {steps} sample count',len(rows)==576);check(f'{name} {steps} subgroup counts',m[str(steps)]['seen']['n']==480 and m[str(steps)]['ood']['n']==96)
  check(f'{name} {steps} nn finite',all(float(r['nn_mse'])>=0 for r in rows))
for seed in [0,1,2]:check('paired noise '+str(seed),noise[f'flow_aug_s{seed}']==noise[f'flow_equivariant_s{seed}'])
check('paired time control noise',noise['flow_time_aug_s0']==noise['flow_time_equivariant_s0'])
for f in ['RESULTS_KO.md','REPRODUCE.md','reports/C4_MATH_KO.md','reports/CODE_AUDIT_KO.md','reports/ae_results.csv','reports/generation_results.csv','reports/nn_reference.json','deliverables/OrbitLab_6min.pptx']:
 check('deliverable '+f,(ROOT/f).exists())
ppt=ROOT/'deliverables/OrbitLab_6min.pptx'
if ppt.exists():
 with zipfile.ZipFile(ppt) as z:check('six presentation slides',sum(p.startswith('ppt/slides/slide') and p.endswith('.xml') for p in z.namelist())==6)
report={'checks':checks,'failed':failures,'all_passed':not failures,'note':'This verifier checks artifact consistency. Source requirement audit, visual QA and scientific interpretation are documented separately; existence alone does not prove research quality.'}
(ROOT/'reports/completion_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps({'checks':len(checks),'failed':failures},indent=2));sys.exit(bool(failures))
