"""Post-hoc paired decoder reconstruction by true renderer size; no model selection."""
from pathlib import Path
import json,sys
import generation_diagnosis as d
from driver import wait_for
import torch
import numpy as np
B=Path(__file__).resolve().parent;OUT=B/'generation_reconstruction';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);ev=d.DetailedEvaluator();all_rows={k:[] for k in ['standard','area_edge']};units=[]
for seed in [880201,880202,880203]:
    for init in range(3):
        run=B/f'generation/runs/d{seed}_s{init}';wait_for(run/'training_summary.json')
        ae_path=run/'ae/ae.pt';ae,_=d.r.load_model(ae_path,torch.device('cpu'))
        for kind in ['standard','area_edge']:
            dest=OUT/f'd{seed}_s{init}_{kind}.json';replay=dest.with_suffix('.pt');verify=dest.with_name(dest.stem+'_verification.json')
            cp=run/('decoder_logit_mean/decoder.pt' if kind=='standard' else 'area_edge_decoder/checkpoint.pt')
            ck=torch.load(cp,map_location='cpu',weights_only=True);model=d.ControlledDecoder('logit_mean');model.load_state_dict(ck['state_dict']);model.eval()
            hashes={'ae':d.sha(ae_path),'decoder':d.sha(cp),'data':{s:d.sha(B/f'generation/data/d{seed}/{s}.npz') for s in ['test','ood']},'metadata':{s:d.sha(B/f'generation/data/d{seed}/{s}_metadata.json') for s in ['test','ood']}}
            if not dest.exists():
                rows=[];first_pred=None
                for split in ['test','ood']:
                    path=B/f'generation/data/d{seed}/{split}.npz';a=np.load(path);meta=json.loads(path.with_name(split+'_metadata.json').read_text())
                    x=torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255;y=torch.from_numpy(a['labels'].copy()).long()
                    for rotation in range(4):
                        for first in range(0,len(x),64):
                            with torch.no_grad():pred=model((ae.encode(d.o.rotate(x[first:first+64],rotation))-ck['mean'])/ck['std'])
                            rr=d.extract_features(pred,y[first:first+64],ev)
                            for j,row in enumerate(rr):
                                radius=meta[first+j]['radius'];row.update(data_seed=seed,initialization=init,split=split,rotation=rotation,base_id=meta[first+j]['base_id'],true_radius=radius,true_size_bin=int(np.searchsorted([.13+.11/3,.13+2*.11/3],radius)))
                            rows.extend(rr)
                            if first_pred is None:first_pred=pred
                torch.save(first_pred,replay)
                d.dump(dest,{'rows':rows,'summary':d.summarize(rows),'hashes':hashes,'source_sha256':d.sha(Path(__file__)),'replay_sha256':d.sha(replay)})
            record=json.loads(dest.read_text());assert record['hashes']==hashes and record['source_sha256']==d.sha(Path(__file__))
            assert record['summary']==d.summarize(record['rows']) and d.sha(replay)==record['replay_sha256']
            a=np.load(B/f'generation/data/d{seed}/test.npz');x=torch.from_numpy(a['images'][:64].copy()).permute(0,3,1,2).float()/255
            with torch.no_grad():pred=model((ae.encode(x)-ck['mean'])/ck['std'])
            torch.testing.assert_close(pred,torch.load(replay,weights_only=True),rtol=0,atol=1e-6)
            if not verify.exists():d.dump(verify,{'all_passed':True,'replayed_images':64,'reaggregated_rows':len(record['rows']),'record_sha256':d.sha(dest)})
            all_rows[kind].extend(record['rows']);units.append(str(dest.relative_to(B)));print('reconstruction audited',seed,init,kind,flush=True)
summary={'units':units,'groups':{k:d.summarize(v) for k,v in all_rows.items()},'rows':{k:len(v) for k,v in all_rows.items()},'scope':'Post-hoc paired diagnostic on confirmation test/OOD. No candidate reselection; actual renderer size. Rotations and initializations on shared scenes are not independent datasets.','source_sha256':d.sha(Path(__file__))}
d.dump(OUT/'summary.json',summary)
