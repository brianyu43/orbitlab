"""Recompute the analytic baseline's matching directly from confusion counts."""
import csv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from common import HERE, dump, r
from object_component_diagnostic import components


def main():
    out=HERE/'reports/object_component_diagnostic_v1';cp=HERE/'configs/object_component_diagnostic_v1.json'
    c=json.loads(cp.read_text());summary=json.loads((out/'summary.json').read_text())
    assert r.sha(cp)==summary['config_sha256']
    assert r.sha(HERE/'object_component_diagnostic.py')==c['code_sha256']
    assert r.sha(HERE/'data/object_world_v1/manifest.json')==c['data_manifest_sha256']
    assert r.sha(out/'predictions.npz')==summary['prediction_sha256']
    with np.load(out/'predictions.npz') as a:pred=a['segmentation'].copy()
    with np.load(HERE/'data/object_world_v1/val/scenes.npz') as a:images=a['images'].copy();truth=a['segmentation'].copy()
    rows=list(csv.DictReader((out/'rows.csv').open()));ious=[];rotation_checks=0
    for i,(rgb,gt,saved,row) in enumerate(zip(images,truth,pred,rows)):
        np.testing.assert_array_equal(components(rgb),saved)
        counts=np.bincount((gt.ravel().astype(int)*5+saved.ravel()).astype(int),minlength=25).reshape(5,5)
        objects=np.flatnonzero(counts.sum(1)[1:])+1
        intersections=counts[objects];unions=counts.sum(1)[objects,None]+counts.sum(0)[None]-intersections
        matrix=intersections/np.maximum(unions,1);rr,cc=linear_sum_assignment(-matrix)
        value=float(matrix[rr,cc].mean());assert abs(value-float(row['matched_visible_iou']))<1e-12;ious.append(value)
        assert len(np.unique(saved[saved>0]))==len(objects)==int(row['predicted_object_count'])
        for k in [1,2,3]:
            rot=components(np.rot90(rgb,k));assert adjusted_rand_score(rot.ravel(),np.rot90(saved,k).ravel())==1
            rotation_checks+=1
    assert len(rows)==len(pred)==len(images)==128
    assert abs(np.mean(ious)-summary['metrics']['matched_visible_iou']['mean'])<1e-12
    # Expose the documented failure: connected touching shapes merge.
    touching=np.zeros((64,64,3),np.uint8);touching[20:30,20:30]=[255,0,0];touching[20:30,30:40]=[0,255,0]
    separated=touching.copy();separated[:,29:31]=0
    assert len(np.unique(components(touching)))-1==1
    assert len(np.unique(components(separated)))-1==2
    report={'all_passed':True,'validation_scenes_replayed':128,'rotation_partition_checks':rotation_checks,
        'matched_visible_iou_recomputed_from_confusion_counts':float(np.mean(ious)),
        'touching_object_merge_failure_demonstrated':True,'verifier_sha256':r.sha(__file__),
        'claim_boundary':'Disconnected validation shapes only; this diagnostic has not evaluated occlusion or learned editing.'}
    dump(out/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
