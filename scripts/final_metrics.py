"""Collect observed results only; never fill unmeasured values."""
import csv,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work'));import research as r


def aggregate():
 spec=json.loads((ROOT/'configs/experiment_matrix.json').read_text());rows=[]
 for task in spec['tasks']:
  root=ROOT/task['out'];run=json.loads((root/'run.json').read_text());metrics=json.loads((root/'analysis/metrics.json').read_text())
  row={k:task[k] for k in ['id','family','model','n','seed','width']};row.update(parameters=run['parameters'],steps=run['steps'],train_seconds=run['train_wall_seconds'],peak_process_rss_bytes=run['peak_process_rss_bytes'])
  for split in ['test','ood']:
   for key in ['mse','foreground_mae','mask_iou','centroid_error']:row[f'{split}_{key}']=metrics[split][key]['mean']
   row[f'{split}_encoder_max_abs']=metrics[split]['encoder_max_abs'];row[f'{split}_decoder_max_abs']=metrics[split]['decoder_max_abs']
  rows.append(row)
 r.write_csv(ROOT/'reports/ae_results.csv',rows)
 groups={}
 for row in rows:
  key=f'{row["family"]}/{row["model"]}/n{row["n"]}'
  groups.setdefault(key,[]).append(row)
 summary={}
 for key,group in groups.items():
  summary[key]={'runs':len(group),'seeds':[x['seed'] for x in group]}
  for metric in ['parameters','steps','train_seconds','test_mse','test_foreground_mae','test_mask_iou','ood_mse','ood_foreground_mae']:
   values=[x[metric] for x in group];summary[key][metric]={'mean':float(np.mean(values)),'seed_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,'values':values}
 paired={}
 for n in [256,1024,4096]:
  family='primary' if n==1024 else 'data'
  aug=[x for x in rows if x['family']==family and x['model']=='aug' and x['n']==n]
  eq=[x for x in rows if x['family']==family and x['model']=='equivariant' and x['n']==n]
  delta=[next(e['test_foreground_mae'] for e in eq if e['seed']==a['seed'])-a['test_foreground_mae'] for a in aug]
  paired[str(n)]={'eq_minus_aug_test_foreground_mae':delta,'mean':float(np.mean(delta)),'sd':float(np.std(delta,ddof=1)),'note':'3 paired initialization seeds; not a population significance claim'}
 r.dump(ROOT/'reports/ae_summary.json',{'groups':summary,'paired':paired,'bootstrap_unit':'base scene, four rotations averaged before resampling','seed_sd_note':'seed SD across three initializations; base-scene CI conditions on a trained model'})
 flow=[]
 for root in sorted((ROOT/'runs').glob('flow_*_s[012]')):
  if not (root/'samples/metrics.json').exists():continue
  d=json.loads((root/'samples/metrics.json').read_text());run=json.loads((root/'run.json').read_text())
  for steps in [0,16,32,64]:
   for split in ['seen','ood']:
    m=d[str(steps)][split]
    flow.append({'run':root.name,'steps':steps,'split':split,'n':m['n'],'shape_accuracy':m['shape_correct']['mean'],'color_accuracy':m['color_correct']['mean'],'joint_accuracy':m['joint_correct']['mean'],'valid_fraction':m['valid']['mean'],'template_iou':m['template_iou']['mean'],'nn_mse':m['nn_mse']['mean'],'centroid_coverage':m['valid_correct_centroid_grid_coverage'],'generation_seconds':d[str(steps)]['sample_and_decode_seconds'],'flow_train_seconds':run['train_wall_seconds']})
 if flow:r.write_csv(ROOT/'reports/generation_results.csv',flow)
 print('aggregated',len(rows),'AE runs;',len(flow),'generation evaluation rows')
if __name__=='__main__':aggregate()
