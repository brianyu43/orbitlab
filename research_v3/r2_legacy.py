"""Frozen palette-assisted historical reader, a domain-limited diagnostic.

Its four-slot capacity, color-direction prior and hard output are retained.
Artificial adapter logits below carry no probabilistic interpretation.
"""
import argparse,json,sys
import numpy as np
import torch
import r2_world as w
import r2_evaluate as e
from v3_common import ROOT,HERE,sha,dump,lock
sys.path.insert(0,str(ROOT/'completion_v2'))
import perception_detail as legacy
BASE=w.BASE/'legacy_palette'
CP=ROOT/'completion_v2/perception_detail/runs/alpha_s0/model.pt'
UNCERTAINTY=['predictive_set_coverage','shape_set_coverage','pose_set_coverage','radius_set_coverage','xy_interval_coverage','rgb_interval_coverage','mean_categorical_set_size']

def freeze():
    return lock(BASE/'protocol.json',{'source_sha256':sha(__file__),'legacy_source_sha256':sha(ROOT/'completion_v2/perception_detail.py'),'legacy_checkpoint_sha256':sha(CP),
        'kind':'alpha','initialization':0,'capacity':4,'input':'Current RGB only. Six known palette directions, brightness threshold13, color groups>=3pixels; largest four groups retained.','training':'Historical1024two-object scenes with distinct palette colors,6000updates. No retraining or target-domain adaptation.','scope':'Domain-limited diagnostic, not an upper bound on continuous/same-color/background inputs. Counts6 exceed capacity. Hard-state adapter values are not calibrated logits; uncertainty metrics are omitted.','eval':'All512development scenes for each count2/3/4/6; same state/edit/merge scoring; geometric-equivalent pose for shape3.'})

@torch.no_grad()
def inference(images,model):
    states=legacy.infer(images,model,'alpha').numpy();raw=np.full((len(images),6,18),-20.,np.float32);masks=np.zeros((len(images),6,64,64),np.float32)
    directions=np.array([[1,0],[0,-1],[-1,0],[0,1]])
    for i,(slots,image) in enumerate(zip(states,images)):
        color_masks=dict(legacy.masks_from_rgb(image))
        for j,v in enumerate(slots):
            if v[15]<=.5:continue
            shape=int(v[:4].argmax());color=int(v[4:10].argmax());pose=int((v[13:15]@directions.T).argmax());radius=int(round(v[12]*8))
            raw[i,j,shape]=20;raw[i,j,4+pose]=20;raw[i,j,8+radius-5]=20;raw[i,j,12:14]=v[10:12]*31.5+31.5
            rgb=np.array(legacy.o.PALETTE[color],np.float32)/255;raw[i,j,14:17]=np.log(rgb/(1-rgb));raw[i,j,17]=20
            masks[i,j]=color_masks[color]
    return states,raw,masks

def run(verify=False):
    freeze();torch.set_num_threads(2);model=legacy.Reader();model.load_state_dict(torch.load(CP,weights_only=True)['state_dict']);model.eval();checks=[]
    for count in [2,3,4,6]:
        folder=w.BASE/f'data/d{w.DEVELOPMENT}/val/n{count}';images=np.load(folder/'rgb.npz')['images'][:,0]
        states,raw,masks=inference(images,model);records=json.loads((folder/'labels.json').read_text());visible=np.load(folder/'masks.npz')['visible'].astype(np.float32)/255
        rows=e.score(raw,masks,records,visible,{'position_half_width':[64.,64.],'rgb_half_width':[255.,255.,255.]})
        for row in rows:
            for key in UNCERTAINTY:row.pop(key)
        metrics=e.summarize(rows);dest=BASE/f'n{count}';dest.mkdir(parents=True,exist_ok=True)
        if verify:
            np.testing.assert_array_equal(states,np.load(dest/'states.npz')['states']);assert rows==json.loads((dest/'rows.json').read_text())
            rec=json.loads((dest/'summary.json').read_text());assert rec['metrics']==metrics;assert rec['prediction_sha256']==sha(dest/'states.npz');assert rec['source_input_sha256']==sha(folder/'rgb.npz')
            assert rec['checkpoint_sha256']==sha(CP) and rec['protocol_sha256']==sha(BASE/'protocol.json')
        else:
            np.savez_compressed(dest/'states.npz',states=states);dump(dest/'rows.json',rows)
            dump(dest/'summary.json',{'count':count,'metrics':metrics,'checkpoint_sha256':sha(CP),'prediction_sha256':sha(dest/'states.npz'),'source_input_sha256':sha(folder/'rgb.npz'),'protocol_sha256':sha(BASE/'protocol.json')})
        checks.append({'count':count,'all_clips_replayed':len(images),'summary_sha256':sha(dest/'summary.json')});print('R2 legacy',count,'verified' if verify else 'evaluated',metrics['all']['joint_edit_success'],flush=True)
    if verify:dump(BASE/'verification.json',{'groups':checks,'clips':sum(x['all_clips_replayed'] for x in checks),'independent_retraining':False,'source_sha256':sha(__file__)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();run(a.verify)
