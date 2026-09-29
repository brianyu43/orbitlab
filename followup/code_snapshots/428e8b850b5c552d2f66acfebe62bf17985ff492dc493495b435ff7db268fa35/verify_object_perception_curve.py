"""Check exact continuation provenance and replay every saved validation output."""
import csv
import json
from common import HERE, dump, digest_tensor, state_digest, torch, np, r
from object_perception import images_only, segmentation_metrics
from object_perception_v2 import LinearImageSlots


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/object_perception_curve_v1.json';c=json.loads(cp.read_text())
    assert r.sha(HERE/'data/object_world_v1/manifest.json')==c['data_manifest_sha256']
    x=images_only('val');train=images_only('train');train_sha=digest_tensor(train);rows=[]
    with np.load(HERE/'data/object_world_v1/val/scenes.npz') as a:gt=a['segmentation'].copy()
    for end in c['checkpoints']:
        pairing=[]
        for kind in c['kinds']:
            folder=HERE/f'runs/object_perception_curve_v1/{kind}_s0_step{end}'
            run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
            assert run['steps']==end and run['training_image_sha256']==train_sha and run['config_sha256']==r.sha(cp)
            assert run['checkpoint_sha256']==r.sha(folder/'model.pt')==m['checkpoint_sha256']
            assert r.sha(folder/'evaluation.pt')==m['evaluation_sha256']
            assert run['code_sha256']==r.sha(HERE/'object_perception_curve.py')
            parent=HERE/run['parent'];assert r.sha(parent/'model.pt')==run['parent_model_sha256']
            assert r.sha(parent/'optimizer.pt')==run['parent_optimizer_sha256']
            if run['continued_from_step']==3000:assert run['parent_model_sha256']==c['parent_checkpoints'][kind]
            parent_ck=torch.load(parent/'model.pt',map_location='cpu',weights_only=True)
            model=LinearImageSlots(kind,c['dim'],c['slots']);model.load_state_dict(parent_ck['state_dict'])
            assert state_digest(model)==run['initial_weights_sha256']
            ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
            assert sum(p.numel() for p in model.parameters())==run['parameters']
            saved_opt=torch.load(folder/'optimizer.pt',map_location='cpu',weights_only=True)
            assert all(int(v['step'])==end for v in saved_opt['optimizer']['state'].values())
            # Replay both RNGs from the original seeds, not just pairwise hashes.
            rg=torch.Generator().manual_seed(986000);ng=torch.Generator().manual_seed(987000)
            for _ in range(end):
                torch.randint(len(train),(c['batch'],),generator=rg)
                torch.randn(c['batch'],c['slots'],c['dim'],generator=ng)
            assert torch.equal(rg.get_state(),saved_opt['sampling_rng'])
            assert torch.equal(ng.get_state(),saved_opt['noise_rng'])
            assert digest_tensor(rg.get_state())==run['sampling_rng_sha256']
            assert digest_tensor(ng.get_state())==run['noise_rng_sha256']
            saved=torch.load(folder/'evaluation.pt',map_location='cpu',weights_only=True)['val']
            records=list(csv.DictReader((folder/'val_rows.csv').open()));assert len(records)==len(x)==128
            pixel=r.pixel_metrics(saved['images'],x);max_error=0.;mask_changes=0
            for i in range(0,len(x),16):
                out=model(x[i:i+16],saved['epsilon'][i:i+16])
                torch.testing.assert_close(out['image'],saved['images'][i:i+16],atol=2e-4,rtol=2e-3)
                max_error=max(max_error,float((out['image']-saved['images'][i:i+16]).abs().max()))
                mask_changes+=int((out['masks'][:,:,0].argmax(1)!=saved['segmentation'][i:i+16]).sum())
            values=[]
            for i in range(len(x)):
                mm=segmentation_metrics(gt[i],saved['segmentation'][i].numpy());mm.update({k:float(v[i]) for k,v in pixel.items()})
                for key in m['splits']['val']:assert abs(mm[key]-float(records[i][key]))<1e-6
                values.append(mm)
            for key,stat in m['splits']['val'].items():assert abs(np.mean([v[key] for v in values])-stat['mean'])<1e-6
            rows.append({'kind':kind,'steps':end,'max_cpu_mps_image_error':max_error,'hard_mask_disagreement_pixels':mask_changes,'observations':128})
            pairing.append(run)
        for key in ['sampling_rng_sha256','noise_rng_sha256','steps']:assert pairing[0][key]==pairing[1][key]
    result={'all_passed':True,'checkpoints':4,'validation_predictions_replayed':512,'results':rows,
        'optimizer_and_rng_continuation_verified':True,'verifier_sha256':r.sha(__file__),
        'claim_boundary':'One-seed validation diagnostics, no test-selected accuracy or successful object discovery inferred.'}
    dump(HERE/'reports/object_perception_curve_v1/verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
