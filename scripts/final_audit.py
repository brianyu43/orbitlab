"""Independently recompute aggregate values, sample lineage and package evidence."""
import csv,hashlib,json,math,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'work'))
import research as r
checks={};stats={}
def check(name,value):
 checks[name]=bool(value)
 if not value: print('FAIL',name)
def read(p):return json.loads((ROOT/p).read_text())
# Re-hash every raster orbit, not just metadata strings.
all_orbits=[]
for split in ['train','val','test','ood']:
 a=np.load(ROOT/f'data/{split}.npz');meta=read(f'data/{split}_metadata.json')
 hashes=[r.orbit_hash(x) for x in a['images']];all_orbits.extend(hashes)
 check(split+' every orbit hash',hashes==[m['orbit_sha256'] for m in meta])
check('4864 unique actual orbits',len(all_orbits)==len(set(all_orbits))==4864)
# Recompute mean metrics from all saved per-sample rows.
for task in read('configs/experiment_matrix.json')['tasks']:
 root=ROOT/task['out'];summary=json.loads((root/'analysis/metrics.json').read_text())
 for split in ['train','val','test','ood']:
  rows=list(csv.DictReader((root/f'analysis/{split}_samples.csv').open()))
  vals=np.array([[float(x[k]) for k in ['mse','foreground_mae','mask_iou','centroid_error']] for x in rows])
  check(task['id']+' '+split+' finite/aggregate',np.isfinite(vals).all() and all(abs(vals[:,i].mean()-summary[split][k]['mean'])<1e-12 for i,k in enumerate(['mse','foreground_mae','mask_iou','centroid_error'])))
  check(task['id']+' '+split+' rotation labels',all(sum(x['base_id']==bid and int(x['rotation'])==k for x in rows)==1 for bid in {x['base_id'] for x in rows} for k in range(4)))
# Flow hashes, checkpoint identity, balanced independent noise sample design.
for name in read('reports/generation_status.json')['completed']:
 out=ROOT/'runs'/name;meta=json.loads((out/'samples/metrics.json').read_text());run=json.loads((out/'run.json').read_text())
 ck=torch.load(out/'flow.pt',map_location='cpu',weights_only=True);saved=torch.load(out/'samples/samples.pt',map_location='cpu',weights_only=True)
 check(name+' code and weights hashes',run['code_sha256']==r.sha(ROOT/'work/generation.py') and meta['flow_sha256']==r.sha(out/'flow.pt') and ck['ae_sha256']==r.sha(ck['ae_path']))
 check(name+' noise hash',meta['noise_sha256']==hashlib.sha256(saved['noise'].numpy().tobytes()).hexdigest())
 for step in [0,16,32,64]:
  rows=list(csv.DictReader((out/f'samples/samples_{step}.csv').open()))
  check(name+f' {step} balanced pairs',all(sum(int(x['shape'])==k and int(x['color'])==c for x in rows)==24 for k in range(4) for c in range(6)))
  check(name+f' {step} latent finite',torch.isfinite(saved['latents'][step]).all().item())
  check(name+f' {step} raw/summary',all(abs(np.mean([x[key]=='True' for x in rows if (x['ood']=='True')==(split=='ood')])-meta[str(step)][split][key]['mean'])<1e-12 for key in ['shape_correct','color_correct','joint_correct','valid'] for split in ['seen','ood']))
 for i in range(4):
  with Image.open(out/f'samples/latent_rotation_{i}.gif') as im:check(name+f' GIF{i}',im.n_frames==4 and im.size==(64,64))
# Moved-path replay preserves every tensor; tiny CPU replay is not a new training run.
original=torch.load(ROOT/'runs/flow_equivariant_s0/flow.pt',map_location='cpu',weights_only=True)
relocated=torch.load(ROOT/'reports/portability_replay/relocated_flow.pt',map_location='cpu',weights_only=True)
check('relocation preserves weights',all(torch.equal(v,relocated['state_dict'][k]) for k,v in original['state_dict'].items()))
check('relocation preserves normalization',all(torch.equal(original[k],relocated[k]) for k in ['mean','std']))
check('relocation CPU 24 samples',all(len(list(csv.DictReader((ROOT/f'reports/portability_replay/samples/samples_{n}.csv').open())))==24 for n in [0,16,32,64]))
# Exact chart values in delivered OOXML match rounded source means.
import xml.etree.ElementTree as ET
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
group=read('reports/ae_summary.json')['groups'];deck=read('reports/deck_data.json')
expected=[[round(group[k]['test_foreground_mae']['mean'],6) for k in ['primary/aug/n1024','primary/equivariant/n1024','parameter/equivariant/n1024','time/aug/n1024','time/equivariant/n1024']],[round(v,6) for s in deck['flowSeries'] for v in s['values']]]
with zipfile.ZipFile(ROOT/'deliverables/OrbitLab_6min.pptx') as z:
 for i,e in enumerate(expected,1):
  xml=ET.fromstring(z.read(f'ppt/slides/charts/chart{i}.xml'));values=[float(n.text) for n in xml.findall('.//c:ser/c:val//c:pt/c:v',ns)]
  check('slide chart source '+str(i),len(values)==len(e) and np.allclose(values,e,atol=1e-12,rtol=0))
 check('six speaker notes',sum(n.startswith('ppt/notesSlides/notesSlide') and n.endswith('.xml') for n in z.namelist())==6)
receipt=read('.slides-build/validation.json');check('validated PPTX hash',receipt['finalSha256']==r.sha(ROOT/'deliverables/OrbitLab_6min.pptx'))
for p in ['orbitlab.py','probe_latents.py','extract_dino.py']:
 check('original unchanged '+p,r.sha(ROOT/'work'/p)==r.sha(ROOT/'sources/extracted/66642ae290950e82/orbitlab_starter'/p))
failed=[k for k,v in checks.items() if not v]
r.dump(ROOT/'reports/final_audit.json',{'checks':checks,'failed':failed,'all_passed':not failed,'scope':'Independent artifact/source/aggregate consistency audit; not proof of generalization, evaluator validity on generated images, or native PowerPoint rendering.'})
print(json.dumps({'checks':len(checks),'failed':failed},indent=2));sys.exit(bool(failed))
