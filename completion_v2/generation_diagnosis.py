"""Read-only decomposition of historical confirmation reconstruction failures."""
from pathlib import Path
import sys, json, time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'generation_recovery'), str(ROOT / 'work'), str(ROOT / 'followup')]
import numpy as np
import torch
import orbitlab as o
import research as r
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator
from p1_evaluate import extract_features
from p0 import sha, dump

OUT = ROOT / 'completion_v2/generation_diagnosis'
OLD = ROOT / 'generation_recovery/confirmation_v1'

def summarize(rows):
    groups = {}
    for split in ('test', 'ood'):
        for shape in range(4):
            for size in range(3):
                g = [v for v in rows if v['split'] == split and v['kind'] == shape and v['true_size_bin'] == size]
                groups[f'{split}/shape{shape}/size{size}'] = {
                    'n': len(g), 'strict': float(np.mean([v['strict_accepted_and_joint'] for v in g])) if g else None,
                    'shape': float(np.mean([v['shape_correct'] for v in g])) if g else None}
    return groups

def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    ev = DetailedEvaluator()
    all_rows, inputs = [], {}
    for seed in (770201,770202,770203):
        for init in range(3):
            dest = OUT / f'd{seed}_s{init}.json'
            run = OLD / f'runs/d{seed}_s{init}/p1'
            ae_path, dec_path = run/'ae/ae.pt', run/'decoder_logit_mean/decoder.pt'
            inputs[str(ae_path.relative_to(ROOT))] = sha(ae_path)
            inputs[str(dec_path.relative_to(ROOT))] = sha(dec_path)
            if dest.exists():
                saved=json.loads(dest.read_text())
                assert saved['source_sha256']==sha(Path(__file__))
                assert saved['ae_sha256']==sha(ae_path) and saved['decoder_sha256']==sha(dec_path)
                all_rows.extend(saved['rows']); continue
            ae,_ = r.load_model(ae_path,torch.device('cpu'))
            ck = torch.load(dec_path,map_location='cpu',weights_only=True)
            dec = ControlledDecoder('logit_mean'); dec.load_state_dict(ck['state_dict']);dec.eval()
            rows=[]
            for split in ('test','ood'):
                path=OLD/f'data/d{seed}/{split}.npz'
                meta_path=path.with_name(f'{split}_metadata.json')
                inputs[str(path.relative_to(ROOT))]=sha(path)
                inputs[str(meta_path.relative_to(ROOT))]=sha(meta_path)
                a=np.load(path);meta=json.loads(meta_path.read_text())
                x=torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255
                y=torch.from_numpy(a['labels'].copy()).long()
                for rotation in range(4):
                    for first in range(0,len(x),64):
                        with torch.no_grad():
                            xx=o.rotate(x[first:first+64],rotation)
                            pred=dec((ae.encode(xx)-ck['mean'])/ck['std'])
                        rr=extract_features(pred,y[first:first+64],ev)
                        for j,row in enumerate(rr):
                            radius=meta[first+j]['radius']
                            row.update(data_seed=seed,initialization=init,split=split,rotation=rotation,
                                       base_id=meta[first+j]['base_id'],true_radius=radius,
                                       true_size_bin=int(np.searchsorted([.13+.11/3,.13+2*.11/3],radius)))
                        rows.extend(rr)
            dump(dest,{'rows':rows,'summary':summarize(rows),'source_sha256':sha(Path(__file__)),
                       'ae_sha256':sha(ae_path),'decoder_sha256':sha(dec_path)})
            all_rows.extend(rows)
            print('diagnosis',seed,init,len(rows),flush=True)
    dest=OUT/'summary.json'
    if not dest.exists():
        dump(dest,{'rows':len(all_rows),'groups':summarize(all_rows),'inputs':inputs,
                   'source_sha256':sha(Path(__file__)),
                   'scope':'Post-hoc historical diagnosis. True renderer radius, not generated-size estimates. Rotations are not independent scenes.'})
    print(json.dumps(json.loads(dest.read_text())['groups']),flush=True)

if __name__=='__main__':main()
