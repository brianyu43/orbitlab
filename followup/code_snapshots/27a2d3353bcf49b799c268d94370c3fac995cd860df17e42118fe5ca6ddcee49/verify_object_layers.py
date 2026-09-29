"""Replay fixed RGB-only outputs and independently check object-mask matching."""
import argparse
import csv
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from common import HERE,dump,r
from object_perception import images_only
from object_perception_v2 import LinearImageSlots
from object_component_diagnostic import components
from object_layer_diagnostic import output_segments


@torch.no_grad()
def main(scope):
    torch.set_num_threads(2);cp=HERE/f'configs/object_layers_{scope}_v1.json';c=json.loads(cp.read_text())
    root=HERE/f'reports/object_layers_{scope}_v1';summary=json.loads((root/'summary.json').read_text())
    assert summary['config_sha256']==r.sha(cp) and c['runner_sha256']==r.sha(HERE/'object_layer_diagnostic.py')
    assert c['data_manifest_sha256']==r.sha(HERE/'data/object_world_v1/manifest.json')
    for source,sha in c['source_hashes'].items():assert r.sha(HERE/source)==sha
    prediction_count=matching_count=aggregate_count=0
    for split in c['splits']:
        folder=root/split;manifest=json.loads((folder/'manifest.json').read_text())
        for file,key in [('outputs.pt','outputs_sha256'),('rows.csv','rows_sha256')]:assert r.sha(folder/file)==manifest[key]
        assert r.sha(HERE/f'data/object_world_v1/{split}/scenes.npz')==manifest['source_data_sha256']
        saved=torch.load(folder/'outputs.pt',map_location='cpu',weights_only=True);x=images_only(split)
        eps=torch.randn(len(x),5,32,generator=torch.Generator().manual_seed(c['epsilon_seed']))
        assert torch.equal(eps,saved['epsilon'])
        with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as a:gt=a['segmentation'].copy();rgb=a['images'].copy()
        for kind,entry in c['models'].items():
            assert r.sha(HERE/entry['path'])==entry['sha256']
            ck=torch.load(HERE/entry['path'],map_location='cpu',weights_only=True);model=LinearImageSlots(kind,ck['dim'],ck['slots'])
            model.load_state_dict(ck['state_dict']);model.eval()
            assert ck['slots']==5 # Enough learned capacity for four objects and background.
            for start in range(0,len(x),c['batch']):
                end=start+c['batch'];out=model(x[start:end],eps[start:end])
                torch.testing.assert_close(out['image'],saved['outputs'][kind]['image'][start:end],atol=0,rtol=0)
                segments=output_segments(out,x[start:end])
                for variant,pred in segments.items():assert torch.equal(pred,saved['outputs'][kind]['segmentations'][variant][start:end])
            prediction_count+=len(x)
        np.testing.assert_array_equal(np.stack([components(im) for im in rgb]),saved['outputs']['component']['segmentations']['native'].numpy())
        rows=list(csv.DictReader((folder/'rows.csv').open()))
        for row in rows:
            kind=row['kind'];variant=row['variant'];i=int(row['base_id']);truth=gt[i]
            pred=saved['outputs'][kind]['segmentations'][variant][i].numpy()
            native=variant.endswith('_native');candidates=list(range(5)) if native else list(range(1,5 if kind=='component' else 6))
            objects=np.unique(truth[truth>0]);iou=np.zeros((len(objects),len(candidates)))
            for a,obj in enumerate(objects):
                for b,label in enumerate(candidates):
                    intersection=np.count_nonzero((truth==obj)&(pred==label));union=np.count_nonzero((truth==obj)|(pred==label))
                    iou[a,b]=intersection/max(1,union)
            rr,cc=linear_sum_assignment(-iou);matched=np.isin(pred,np.array(candidates)[cc])
            fg=truth>0;expected={'matched_visible_iou':float(iou[rr,cc].mean()),
                'foreground_ari':adjusted_rand_score(truth[fg],pred[fg]),
                'all_pixel_ari':adjusted_rand_score(truth.ravel(),pred.ravel()),
                'matched_region_background_fraction':np.count_nonzero(matched&~fg)/max(1,np.count_nonzero(matched))}
            for key,value in expected.items():assert abs(value-float(row[key]))<1e-10
            matching_count+=1
        for result in [v for v in summary['results'] if v['split']==split]:
            records=[v for v in rows if v['kind']==result['kind'] and v['variant']==result['variant']]
            assert len(records)==result['scenes']==128
            for key,value in result['metrics'].items():
                assert abs(np.mean([float(v[key]) for v in records])-value['mean'])<1e-10;aggregate_count+=1
        print('verified',scope,split,flush=True)
    report={'all_passed':True,'learned_image_predictions_replayed':prediction_count,'region_matchings_recomputed':matching_count,
        'aggregate_checks':aggregate_count,'five_learned_slots_per_model_verified':True,
        'source_and_checkpoint_hashes_verified':True,'verifier_sha256':r.sha(__file__),
        'claim_boundary':c['claim_boundary']}
    dump(root/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['validation','heldout'],required=True);args=parser.parse_args();main(args.scope)
