"""Extract frozen slots from RGB only; no labels or masks are materialized."""
import json
import time
import torch
import torch.nn.functional as F
from common import HERE,dump,fresh_dir,digest_tensor,r
from object_perception import images_only
from object_perception_v2 import LinearImageSlots


@torch.no_grad()
def encode(model,x,eps):
    feature=model.encoder(x).permute(0,2,3,1)+model.encoder_position(model.encoder_grid)
    feature=model.feature_mlp(model.feature_norm(feature))
    if model.kind=='slot':return model.router(feature.reshape(len(x),-1,model.dim),eps)[0]
    pooled=F.adaptive_avg_pool2d(feature.permute(0,3,1,2),(4,4))
    return model.router(pooled.flatten(1)).reshape(len(x),model.slots,model.dim)


def freeze():
    cp=HERE/'configs/object_slot_cache_v1.json'
    if not cp.exists():dump(cp,{'splits':['train','val','test','ood','count3','count4','occlusion'],'kinds':['flat','slot'],
        'train_epsilon_seed':993000,'evaluation_epsilon_seed':988000,'batch':16,
        'checkpoints':{k:r.sha(HERE/f'runs/object_perception_curve_v1/{k}_s0_step12000/model.pt') for k in ['flat','slot']},
        'source_hashes':{n:r.sha(HERE/n) for n in ['object_perception.py','object_perception_v2.py']},
        'data_manifest_sha256':r.sha(HERE/'data/object_world_v1/manifest.json'),'code_sha256':r.sha(__file__),
        'information':'Extraction reads RGB only. Labels are accessed separately by supervised readout training or evaluation.', 'frozen_unix':time.time()})
    return json.loads(cp.read_text())


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(2);root=fresh_dir(HERE/'data/object_slot_cache_v1');rows=[]
    for kind in c['kinds']:
        path=HERE/f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt';assert r.sha(path)==c['checkpoints'][kind]
        ck=torch.load(path,map_location='cpu',weights_only=True);model=LinearImageSlots(kind,ck['dim'],ck['slots']);model.load_state_dict(ck['state_dict']);model.eval()
        for split in c['splits']:
            x=images_only(split);seed=c['train_epsilon_seed'] if split=='train' else c['evaluation_epsilon_seed']
            eps=torch.randn(len(x),5,32,generator=torch.Generator().manual_seed(seed));pieces=[]
            for i in range(0,len(x),c['batch']):pieces.append(encode(model,x[i:i+c['batch']],eps[i:i+c['batch']]))
            slots=torch.cat(pieces)
            # The lightweight path must be exactly the original model's slots.
            full=model(x[:16],eps[:16])['slots'];torch.testing.assert_close(slots[:16],full,atol=0,rtol=0)
            path=root/f'{kind}_{split}.pt';torch.save({'slots':slots,'epsilon':eps},path)
            rows.append({'kind':kind,'split':split,'images':len(x),'rgb_sha256':digest_tensor(x),'slot_sha256':digest_tensor(slots),
                'checkpoint_sha256':c['checkpoints'][kind],'file_sha256':r.sha(path)})
            print('cached',kind,split,len(x),flush=True)
    dump(root/'manifest.json',{'files':rows,'config_sha256':r.sha(HERE/'configs/object_slot_cache_v1.json'),
        'claim_boundary':'RGB feature extraction only, no state readout or intervention result.'})


if __name__=='__main__':main()
