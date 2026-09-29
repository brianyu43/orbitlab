"""Validate geometry diagnostic against simulator truth, without model predictions."""
from pathlib import Path
import json
import numpy as np
from svib_prepare import sha
B=Path(__file__).resolve().parent;rows=[]
for path in sorted((B/'dynamics/data').glob('d*/*/*/trajectories.npz')):
    if path.parent.name in ['train','val']:continue
    a=np.load(path);s=a['states'][:,4:];rad=s[:,:, :,4];live=rad>0;xy=s[...,:2]
    wall=np.where(live[...,None],np.maximum(np.abs(xy)-(31.5-rad[...,None]),0),0).max((2,3))
    overlap=np.zeros(s.shape[:2])
    for i in range(4):
        for j in range(i+1,4):
            distance=np.linalg.norm(xy[:,:,i]-xy[:,:,j],axis=-1)
            overlap=np.maximum(overlap,np.where(live[:,:,i]&live[:,:,j],np.maximum(rad[:,:,i]+rad[:,:,j]-distance,0),0))
    stats={}
    for h in [1,16,61]:
        w=wall[:,:h].max(1);v=overlap[:,:h].max(1)
        stats[str(h)]={'wall_over_1px_rate':float((w>1).mean()),'overlap_over_1px_rate':float((v>1).mean()),'max_wall':float(w.max()),'max_overlap':float(v.max())}
    rows.append({'path':str(path.relative_to(B)),'data_sha256':sha(path),'summary':stats})
assert len(rows)==39
out={'rows':rows,'source_sha256':sha(Path(__file__)),'scope':'True radii and states; same horizon and 1px threshold. This checks simulator truth before interpreting model overlap rates.'}
(B/'dynamics_truth_geometry.json').write_text(json.dumps(out,indent=2)+'\n')
print('truth max wall',max(v['summary']['61']['max_wall'] for v in rows),'max overlap',max(v['summary']['61']['max_overlap'] for v in rows))
print('truth nonzero >1px rates',[(v['path'],v['summary']['61']) for v in rows if v['summary']['61']['overlap_over_1px_rate']>0 or v['summary']['61']['wall_over_1px_rate']>0])
