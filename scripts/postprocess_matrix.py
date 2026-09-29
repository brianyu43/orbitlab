import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];spec=json.loads((ROOT/'configs/experiment_matrix.json').read_text());status={'total':len(spec['tasks']),'completed':[]}
for task in spec['tasks']:
 source=ROOT/task['out'];out=source/'analysis';marker=out/'COMPLETE.json'
 if marker.exists():status['completed'].append(task['id']);continue
 if not (source/'COMPLETE.json').exists():raise RuntimeError('Training incomplete '+task['id'])
 status['current']=task['id'];(ROOT/'reports/analysis_status.json').write_text(json.dumps(status,indent=2))
 commands=[[sys.executable,'work/research.py','analyze','--device','mps','--ae',str(source/'ae.pt'),'--n','256','--out',str(out)]]
 if task['family'] in ['primary','diagnostic']:commands.append([sys.executable,'work/probes.py','--input',str(out),'--ae',str(source/'ae.pt')])
 with (ROOT/'reports'/f'{task["id"]}_analysis.log').open('w') as log:
  for cmd in commands:
   proc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   if proc.returncode:raise RuntimeError('Failed '+str(cmd))
 marker.write_text(json.dumps({'task':task,'probes':task['family'] in ['primary','diagnostic'],'completed_unix':time.time()},indent=2))
 status['completed'].append(task['id']);print('analysis',len(status['completed']),'/',len(spec['tasks']),task['id'],flush=True)
status['current']=None;(ROOT/'reports/analysis_status.json').write_text(json.dumps(status,indent=2))
