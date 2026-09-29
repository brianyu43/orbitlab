import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'work'))
import torch
import research as r,generation as g,orbitlab as o

torch.set_num_threads(4);ev=g.StateEvaluator();spec=json.loads((r.ROOT/(sys.argv[1] if len(sys.argv)>1 else 'configs/experiment_matrix.json')).read_text());gate=spec['entry_gate'];result={}
for task in [t for t in spec['tasks'] if t['family']=='primary']:
 root=r.ROOT/task['out'];run=json.loads((root/'run.json').read_text());ae,_=r.load_model(root/'ae.pt',torch.device('cpu'));x,y,_=r.load_data('val');preds=[]
 with torch.no_grad():
  for k in range(4):
   for i in range(0,len(x),32):preds.extend(ev.predict(ae(o.rotate(x[i:i+32],k))))
 labels=y.tolist()*4;shape=sum(p['predicted_shape']==y[0] for p,y in zip(preds,labels))/len(preds);color=sum(p['predicted_color']==y[1] for p,y in zip(preds,labels))/len(preds)
 m=run['validation']['val'];checks={'mse_vs_black':m['mse']['mean']<gate['validation_mse_lt_black_ratio']*m['black_mse']['mean'],'foreground_mae':m['foreground_mae']['mean']<=gate['foreground_mae_max'],'iou':m['mask_iou']['mean']>=gate['mask_iou_min'],'shape':shape>=gate['reconstructed_shape_accuracy_min'],'color':color>=gate['reconstructed_color_accuracy_min']}
 result[task['id']]={'checks':checks,'numeric_pass':all(checks.values()),'reconstructed_shape_accuracy':shape,'reconstructed_color_accuracy':color,'foreground_mae':m['foreground_mae']['mean'],'iou':m['mask_iou']['mean'],'visual_review_required':True}
r.dump(r.ROOT/('reports/quality_gate_repair.json' if len(sys.argv)>1 else 'reports/quality_gate.json'),result);print(json.dumps(result,indent=2))
if not all(v['numeric_pass'] for v in result.values()):raise SystemExit(1)
