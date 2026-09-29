"""Execute a predeclared matrix sequentially; immutable completed run outputs."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',default='configs/experiment_matrix.json');args=p.parse_args();manifest_path=ROOT/args.manifest
 spec=json.loads(manifest_path.read_text());tasks=spec['tasks'];status={'manifest':str(manifest_path),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'completed':[],'total':len(tasks)}
 for task in tasks:
  out=ROOT/task['out'];done=out/'COMPLETE.json'
  if done.exists():
   old=json.loads(done.read_text())
   if old['task']!=task:raise RuntimeError('Conflicting existing task '+str(out))
   for f,h in old['evidence'].items():
    if hashlib.sha256((out/f).read_bytes()).hexdigest()!=h:raise RuntimeError('Evidence changed '+f)
   status['completed'].append(task['id']);continue
  status['current']=task['id'];status['started_unix']=time.time();(ROOT/'reports/matrix_status.json').write_text(json.dumps(status,indent=2))
  cmd=[sys.executable,str(ROOT/'work/research.py'),'train','--device','mps','--model',task['model'],'--n',str(task['n']),'--steps',str(task['steps']),'--width',str(task.get('width',16)),'--seed',str(task['seed']),'--out',task['out']]
  if task.get('seconds'):cmd+=['--seconds',str(task['seconds'])]
  with (ROOT/'reports'/f'{task["id"]}.log').open('w') as log:
   result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError(f'Failed {task["id"]}; inspect log')
  evidence={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in ['run.json','ae.pt','validation/metrics.json']}
  done.write_text(json.dumps({'task':task,'evidence':evidence},indent=2));status['completed'].append(task['id']);print('completed',len(status['completed']),'/',len(tasks),task['id'],flush=True)
 status['current']=None;status['finished_unix']=time.time();(ROOT/'reports/matrix_status.json').write_text(json.dumps(status,indent=2))

if __name__=='__main__':main()
