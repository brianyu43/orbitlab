"""Generation is conditional on recorded numeric and visual AE entry gates."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=json.loads((ROOT/'configs/generation_ae_repair.json').read_text())
gates=json.loads((ROOT/'reports/quality_gate_repair.json').read_text())
visual=json.loads((ROOT/'reports/repair_visual_review.json').read_text())
if not all(g['numeric_pass'] for g in gates.values()) or not visual['passed']:raise RuntimeError('AE entry gate not satisfied')
status={'completed':[],'total':6,'current':None}

def execute(name,ae,seed,steps=4000,seconds=0):
 out=ROOT/'runs'/name;sample=out/'samples';status['current']=name;(ROOT/'reports/generation_status.json').write_text(json.dumps(status,indent=2))
 with (ROOT/'reports'/f'{name}.log').open('a') as log:
  if not (out/'run.json').exists():
   cmd=[sys.executable,'work/generation.py','train','--ae',str(ae),'--seed',str(seed),'--steps',str(steps),'--seconds',str(seconds),'--out',str(out)]
   if subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode:raise RuntimeError('flow train failed '+name)
  if not (sample/'metrics.json').exists():
   cmd=[sys.executable,'work/generation.py','sample','--flow',str(out/'flow.pt'),'--seed',str(53000+seed),'--n','576','--out',str(sample)]
   if subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode:raise RuntimeError('sample failed '+name)
 status['completed'].append(name);print('generation',len(status['completed']),'/',status['total'],name,flush=True)

for task in spec['tasks']:
 execute(f'flow_{task["model"]}_s{task["seed"]}',ROOT/task['out']/'ae.pt',task['seed'])
# An explicitly exploratory equal-time flow control; paired with each other.
budget=json.loads((ROOT/'runs/flow_equivariant_s0/run.json').read_text())['train_wall_seconds']
control={'purpose':'exploratory equal-time flow control, seed 0 only','seconds':budget,'frozen_unix':time.time(),'base_noise_seed':53000};(ROOT/'configs/flow_time_control.json').write_text(json.dumps(control,indent=2));status['total']=8
for model in ['aug','equivariant']:
 task=next(t for t in spec['tasks'] if t['model']==model and t['seed']==0)
 execute('flow_time_'+model+'_s0',ROOT/task['out']/'ae.pt',0,100000,budget)
status['current']=None;status['finished_unix']=time.time();(ROOT/'reports/generation_status.json').write_text(json.dumps(status,indent=2))
