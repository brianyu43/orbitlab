"""Measured training duration inventory; overlapping workloads, not speed benchmark."""
from pathlib import Path
import json
B=Path(__file__).resolve().parent;rows=[]
for branch in ['generation','perception','perception_detail','dynamics','svib']:
    for p in sorted((B/branch/'runs').rglob('run.json')):
        a=json.loads(p.read_text());seconds=a.get('seconds',a.get('train_seconds'))
        assert seconds is not None and a['steps']>0
        rows.append({'study':branch,'path':str(p.relative_to(B)),'steps':a['steps'],'training_seconds':seconds,'parameters':a.get('parameters')})
assert len(rows)==98
r={'rows':rows,'total_updates':sum(v['steps'] for v in rows),'summed_training_seconds':sum(v['training_seconds'] for v in rows),'scope':'Durations sum across overlapping workers. Not total wall-clock duration and not a controlled hardware/runtime benchmark. Multistep/C4 arms use different compute per update. Reported trainer timing boundaries include different checkpoint/validation overheads; consult individual logs.'}
(B/'cost_inventory.json').write_text(json.dumps(r,indent=2)+'\n');print(r['total_updates'],r['summed_training_seconds'])
