"""Image-only connected-region baseline; validation diagnostic, no model fitting.

This deliberately simple baseline exposes the separable black-background world.
It is not a learned representation or a solution to touching/occluded objects.
"""
import json
import time
import numpy as np
from scipy import ndimage
from common import HERE, dump, r, torch, o
from object_perception import segmentation_metrics


def components(rgb):
    brightness=np.asarray(rgb,dtype=np.float64).max(-1)/255
    cc,total=ndimage.label(brightness>.05,structure=np.ones((3,3),int))
    regions=sorted([(int((cc==k).sum()),k) for k in range(1,total+1) if (cc==k).sum()>=4],reverse=True)[:4]
    out=np.zeros(cc.shape,dtype=np.int64)
    for j,(_,k) in enumerate(regions,1):
        mask=cc==k;level=np.quantile(brightness[mask],.95)
        out[mask & (brightness>=.5*level)]=j
    return out


def main():
    cp=HERE/'configs/object_component_diagnostic_v1.json'
    if not cp.exists():dump(cp,{'split':'val','initial_brightness_threshold':.05,'component_connectivity':8,
        'minimum_component_pixels':4,'maximum_objects':4,'edge_threshold':'.5 times each connected region 95th percentile brightness',
        'training':'None. Inference reads only RGB; no true mask, state, count or palette.',
        'limitations':'Assumes a black background and disconnected objects. Four-object capacity is a known task constraint.',
        'selection':'One prespecified diagnostic on validation. Thresholds not tuned against labels.',
        'data_manifest_sha256':r.sha(HERE/'data/object_world_v1/manifest.json'),'code_sha256':r.sha(__file__),'frozen_unix':time.time()})
    c=json.loads(cp.read_text());assert c['code_sha256']==r.sha(__file__)
    with np.load(HERE/'data/object_world_v1/val/scenes.npz') as a:rgb=a['images'].copy();gt=a['segmentation'].copy()
    rows=[];segments=[];examples=[];palette=np.array([[0.,0.,0.],[1.,.2,.2],[.2,1.,.2],[.2,.3,1.],[1.,1.,.2]])
    for i,image in enumerate(rgb):
        pred=components(image);row=segmentation_metrics(gt[i],pred);row['base_id']=i
        row['predicted_object_count']=int(len(np.unique(pred[pred>0])));rows.append(row);segments.append(pred)
        if i<12:
            examples.extend([torch.from_numpy(image.copy()).permute(2,0,1)/255,torch.from_numpy(palette[pred]).permute(2,0,1)])
    out=HERE/'reports/object_component_diagnostic_v1';out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'predictions.npz',segmentation=np.stack(segments));r.write_csv(out/'rows.csv',rows)
    summary={k:r.bootstrap([v[k] for v in rows]) for k in ['foreground_ari','all_pixel_ari','matched_visible_iou']}
    summary['object_count_accuracy']=float(np.mean([v['predicted_object_count']==v['visible_object_count'] for v in rows]))
    o.grid(torch.stack(examples),out/'examples.png',6)
    dump(out/'summary.json',{'metrics':summary,'scenes':len(rows),'config_sha256':r.sha(cp),'prediction_sha256':r.sha(out/'predictions.npz'),
        'claim_boundary':'Validation-only analytic segmentation baseline; no learned object representation, amodal recovery, or end-to-end editing accuracy.'})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
