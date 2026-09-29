"""Recompute saved validation metrics and replay image predictions on CPU."""
import argparse
import csv
import json
from common import HERE, dump, digest_tensor, torch, np, r
from object_perception import ImageSlots, images_only, segmentation_metrics
from object_perception_v2 import LinearImageSlots


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,choices=[1,2],required=True);args=parser.parse_args()
    torch.set_num_threads(2);version=args.version;cp=HERE/f'configs/object_perception_pilot_v{version}.json';c=json.loads(cp.read_text())
    assert r.sha(HERE/'data/object_world_v1/manifest.json')==c['data_manifest_sha256']
    result=[];x=images_only('val');train_sha=digest_tensor(images_only('train'))
    with np.load(HERE/'data/object_world_v1/val/scenes.npz') as archive:gt=archive['segmentation'].copy()
    for kind in c['kinds']:
        folder=HERE/f'runs/object_perception_pilot_v{version}/{kind}_s0'
        run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
        assert run['steps']==c['steps'] and run['training_image_sha256']==train_sha and run['config_sha256']==r.sha(cp)
        assert run['checkpoint_sha256']==r.sha(folder/'model.pt')==m['checkpoint_sha256']
        assert r.sha(folder/'evaluation.pt')==m['evaluation_sha256']
        source='object_perception.py' if version==1 else 'object_perception_v2.py'
        assert run['code_sha256']==r.sha(HERE/source)
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        model=(ImageSlots if version==1 else LinearImageSlots)(kind,ck['dim'],ck['slots']);model.load_state_dict(ck['state_dict']);model.eval()
        assert sum(v.numel() for v in model.parameters())==run['parameters']
        saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True)['val'];records=list(csv.DictReader((folder/'val_rows.csv').open()))
        assert len(records)==len(x)==128
        values=[];pixel=r.pixel_metrics(saved['images'],x);max_error=0.;seg_diff=0;pixels=0
        for i in range(0,len(x),16):
            replay=model(x[i:i+16],saved['epsilon'][i:i+16]);expected=saved['images'][i:i+16]
            torch.testing.assert_close(replay['image'],expected,atol=2e-4,rtol=2e-3)
            max_error=max(max_error,float((replay['image']-expected).abs().max()))
            labels=replay['masks'][:,:,0].argmax(1);seg_diff+=int((labels!=saved['segmentation'][i:i+16]).sum());pixels+=labels.numel()
        for i in range(len(x)):
            mm=segmentation_metrics(gt[i],saved['segmentation'][i].numpy());mm.update({k:float(v[i]) for k,v in pixel.items()})
            for key in ['mse','foreground_mae','mask_iou','foreground_ari','all_pixel_ari','matched_visible_iou']:
                assert abs(mm[key]-float(records[i][key]))<1e-6
            values.append(mm)
        for key,metric in m['splits']['val'].items():assert abs(float(np.mean([v[key] for v in values]))-metric['mean'])<1e-6
        result.append({'kind':kind,'parameters':run['parameters'],'validation_scenes':len(x),'max_cpu_mps_image_replay_error':max_error,
            'cpu_mps_hard_mask_disagreement_fraction':seg_diff/pixels,'sampling_rng_sha256':run['sampling_rng_sha256']})
    assert result[0]['sampling_rng_sha256']==result[1]['sampling_rng_sha256']
    report={'all_passed':True,'version':version,'results':result,'image_replay_tolerance':{'atol':.0002,'rtol':.002},
        'mask_note':'Metrics recomputed from stored MPS masks. Cross-backend argmax changes are recorded because nearly tied logits can flip.',
        'claim_boundary':'Pilot artifact/metric consistency, not evidence of successful object discovery. Validation-only one-seed experiment.',
        'verifier_sha256':r.sha(__file__)}
    dump(HERE/f'reports/object_perception_pilot_v{version}/verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
