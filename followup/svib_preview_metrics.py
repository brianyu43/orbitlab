"""Transparent pixel diagnostics; target-derived regions are scoring only."""
import numpy as np
import torch
from common import HERE,r
from svib_preview_data import NAME


def load_images(address):
    root=HERE/f'data/{NAME}/{address}'
    with np.load(root/'inputs.npz') as a:
        assert a.files==['source_rgb'];source=torch.from_numpy(a['source_rgb'].copy()).permute(0,3,1,2).float()/255
    with np.load(root/'labels.npz') as a:
        target=torch.from_numpy(a['target_rgb'].copy()).permute(0,3,1,2).float()/255
        masks={'changed':a['changed_mask'].copy(),'foreground':a['foreground_mask'].copy()}
    return source,target,masks


def score(prediction,source,target,masks):
    pred=np.asarray(prediction,np.float64);source=np.asarray(source,np.float64);target=np.asarray(target,np.float64)
    assert pred.shape==source.shape==target.shape and pred.shape[1:]==(3,128,128)
    assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
    squared=(pred-target)**2;identity=(source-target)**2;rows=[]
    for i in range(len(pred)):
        changed=masks['changed'][i];foreground=masks['foreground'][i]
        row={'episode':i,'has_change':bool(changed.any()),'changed_pixels':int(changed.sum()),
            'foreground_pixels':int(foreground.sum()),'pixel_mse':float(squared[i].mean()),
            'image_squared_error_sum':float(squared[i].sum()),'identity_pixel_mse':float(identity[i].mean())}
        for name,mask in [('foreground',foreground),('changed',changed),('unchanged',~changed)]:
            row[name+'_pixel_mse']=float(squared[i,:,mask].mean()) if mask.any() else None
        row['pixel_mse_minus_identity']=row['pixel_mse']-row['identity_pixel_mse']
        row['lower_pixel_mse_than_identity']=bool(row['pixel_mse_minus_identity'] < -1e-12)
        rows.append(row)
    return rows


def summarize(rows):
    excluded={'episode','has_change','changed_pixels','foreground_pixels'};summary={}
    for name,selected in [('all',rows),('changed',[x for x in rows if x['has_change']]),('unchanged',[x for x in rows if not x['has_change']])]:
        entry={'episodes':len(selected),'metrics':{}}
        for metric in rows[0]:
            if metric in excluded:continue
            values=[x[metric] for x in selected if x[metric] is not None]
            entry['metrics'][metric]={'finite_episodes':len(values),'total_episodes':len(selected),
                **(r.bootstrap(values) if values else {'mean':None,'base_scene_ci95':None})}
        summary[name]=entry
    return summary
