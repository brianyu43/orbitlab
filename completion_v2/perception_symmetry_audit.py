"""Supplementary observable-pose scoring; never replaces the original metric."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'completion_v2'),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import torch
import numpy as np
from object_state_readout import geometric_assignment
from object_oracle_study import factor_checks
from p0 import dump,sha
import perception as p

def audit(folder):
    out=folder/'symmetry_audit.json'
    if out.exists():return
    record=json.loads((folder/'summary.json').read_text());data=p.BASE/f"data/d{record['data_seed']}/{record['split']}"
    queries=np.load(data/'queries.npz');idx=queries['source_indices'];truth=torch.from_numpy(np.load(data/'scene_labels.npz')['states'].copy())[idx]
    target=torch.from_numpy(np.load(data/'targets.npz')['states'].copy());inference=torch.load(folder/'inference.pt',weights_only=True)
    source=inference['states'][idx];summary={}
    dirs=torch.tensor([[1.,0],[0,-1],[-1,0],[0,1]])
    for controller in ['analytic','learned']:
        pred=torch.load(folder/f'{controller}_states.pt',weights_only=True);rows=json.loads((folder/f'{controller}_rows.json').read_text())
        full=[];changed=[]
        for i,row in enumerate(rows):
            mapping,_,gt=geometric_assignment(source[i].numpy(),truth[i].numpy())
            mapped=torch.stack([pred[i,mapping[g]] if g in mapping else torch.zeros(16) for g in gt])
            checks=factor_checks(mapped[None],target[i,gt][None])[0]
            ps=(mapped[:,13:15]@dirs.T).argmax(-1);ts=(target[i,gt,13:15]@dirs.T).argmax(-1)
            zigzag=target[i,gt,:4].argmax(-1)==3
            checks[:,5]=torch.where(zigzag,ps%2==ts%2,checks[:,5])
            for j,g in enumerate(gt):
                if g not in mapping:checks[j]=False
            value=bool(checks.all() and int((pred[i,:,15]>=.5).sum())==len(gt) and row['target_selection_correct'])
            assert value or not row['end_to_end_success']
            full.append(value);changed.append(value and not row['end_to_end_success'])
        ids=sorted(set(idx.tolist()))
        summary[controller]={'strict_original':record['metrics'][controller]['end_to_end_success'],
                             'pose_equivalent_end_to_end':float(np.mean([np.mean(np.asarray(full)[idx==i]) for i in ids])),
                             'tasks_only_failing_unobservable_zigzag_pose':int(sum(changed)),
                             'n_tasks':len(rows)}
    dump(out,{'summary':summary,'source_sha256':sha(Path(__file__)),
              'scope':'Post-hoc supplementary metric identifies 180-degree symmetric zigzags. Original prespecified strict scores stay unchanged. Not a new claim of recovering hidden pose.'})

if __name__=='__main__':
    torch.set_num_threads(1)
    for path in sorted((p.BASE/'evaluation').glob('d*/*/*/summary.json')):audit(path.parent)
