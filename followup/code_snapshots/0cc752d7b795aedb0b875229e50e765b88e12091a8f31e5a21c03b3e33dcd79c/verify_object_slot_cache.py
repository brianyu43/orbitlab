"""Replay each cache entry through the full frozen image model."""
import json
import torch
from common import HERE,dump,digest_tensor,r
from object_perception import images_only
from object_perception_v2 import LinearImageSlots


@torch.no_grad()
def main():
    torch.set_num_threads(2);cp=HERE/'configs/object_slot_cache_v1.json';c=json.loads(cp.read_text())
    root=HERE/'data/object_slot_cache_v1';manifest=json.loads((root/'manifest.json').read_text());total=0
    assert manifest['config_sha256']==r.sha(cp) and c['code_sha256']==r.sha(HERE/'cache_object_slots.py')
    assert c['data_manifest_sha256']==r.sha(HERE/'data/object_world_v1/manifest.json')
    for source,sha in c['source_hashes'].items():assert r.sha(HERE/source)==sha
    for kind in c['kinds']:
        source=HERE/f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt';assert r.sha(source)==c['checkpoints'][kind]
        ck=torch.load(source,map_location='cpu',weights_only=True);model=LinearImageSlots(kind,ck['dim'],ck['slots']);model.load_state_dict(ck['state_dict']);model.eval()
        for row in [v for v in manifest['files'] if v['kind']==kind]:
            split=row['split'];path=root/f'{kind}_{split}.pt';assert r.sha(path)==row['file_sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True);x=images_only(split);assert digest_tensor(x)==row['rgb_sha256']
            seed=c['train_epsilon_seed'] if split=='train' else c['evaluation_epsilon_seed']
            eps=torch.randn(len(x),5,32,generator=torch.Generator().manual_seed(seed));assert torch.equal(eps,saved['epsilon'])
            assert digest_tensor(saved['slots'])==row['slot_sha256']
            for i in range(0,len(x),c['batch']):
                full=model(x[i:i+c['batch']],eps[i:i+c['batch']])['slots'];torch.testing.assert_close(full,saved['slots'][i:i+c['batch']],atol=0,rtol=0)
            total+=len(x);print('verified cache',kind,split,flush=True)
    dump(HERE/'reports/object_slot_cache_v1/verification.json',{'all_passed':True,'image_encodings_replayed':total,'cache_files':len(manifest['files']),
        'verifier_sha256':r.sha(__file__),'claim_boundary':'RGB-only extraction verified; readout training and intervention results are separate.'})


if __name__=='__main__':main()
