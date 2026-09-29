"""Frozen-checkpoint encoder/decoder and RGB-background diagnostics.

No fitting or label-dependent postprocessing. Validation and held-out runs use
separate frozen configs; the latter evaluates every prespecified variant.
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from common import HERE,dump,fresh_dir,r,o
from object_perception import images_only,segmentation_metrics
from object_perception_v2 import LinearImageSlots
from object_component_diagnostic import components


def freeze(scope):
    cp=HERE/f'configs/object_layers_{scope}_v1.json'
    if not cp.exists():
        models={kind:{'path':f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt',
            'sha256':r.sha(HERE/f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt')} for kind in ['flat','slot']}
        dump(cp,{'scope':scope,'splits':['val'] if scope=='validation' else ['test','ood','count3','count4','occlusion'],
            'models':models,'batch':16,'epsilon_seed':988000,'brightness_threshold':.05,
            'variants':{'flat':['decoder_native','decoder_rgb_foreground','contribution_rgb_foreground'],
                        'slot':['decoder_native','decoder_rgb_foreground','contribution_rgb_foreground','attention_native','attention_rgb_foreground']},
            'foreground':'Input RGB max-channel > .05, same as the existing training loss; no true segmentation.',
            'attention':'Bilinear upsample final encoder-to-slot attention from 32px to 64px, align_corners=False.',
            'contribution':'Hard assignment by max positive RGB channel times decoder alpha, with the same RGB foreground mask.',
            'matching':'Native outputs match all five learned slots; foreground variants exclude the explicit background label from matching.',
            'selection':'Fixed 12k checkpoint; report all variants and component baseline. No label-driven thresholds, checkpoint or variant selection.',
            'data_manifest_sha256':r.sha(HERE/'data/object_world_v1/manifest.json'),
            'source_hashes':{name:r.sha(HERE/name) for name in ['object_perception.py','object_perception_v2.py','object_component_diagnostic.py']},
            'runner_sha256':r.sha(__file__),'frozen_unix':time.time(),
            'claim_boundary':'Single initialization. Diagnoses image segmentation; no causal factor separation or end-to-end edit accuracy inferred.'})
    return json.loads(cp.read_text())


def output_segments(out,x):
    fg=x.amax(1)>.05;decoder=out['masks'][:,:,0].argmax(1)
    contribution=(out['rgb'].clamp_min(0).amax(2)*out['masks'][:,:,0]).argmax(1)
    result={'decoder_native':decoder,'decoder_rgb_foreground':torch.where(fg,decoder+1,0),
            'contribution_rgb_foreground':torch.where(fg,contribution+1,0)}
    if out['attention'] is not None:
        attention=out['attention'].transpose(1,2).reshape(len(x),5,32,32)
        native=F.interpolate(attention,(64,64),mode='bilinear',align_corners=False).argmax(1)
        result.update(attention_native=native,attention_rgb_foreground=torch.where(fg,native+1,0))
    return result


def score(gt,pred,native=False,slots=5):
    shifted=pred if native else pred-1
    result=segmentation_metrics(gt,shifted,slots)
    assigned=list(result['assignment'].values());chosen=np.isin(shifted,assigned)
    result['matched_region_background_fraction']=float(np.sum(chosen&(gt==0))/max(1,chosen.sum()))
    result['predicted_regions_on_foreground']=int(len(np.unique(pred[gt>0])))
    return result


@torch.no_grad()
def run(scope):
    c=freeze(scope);torch.set_num_threads(2);root=fresh_dir(HERE/f'reports/object_layers_{scope}_v1');summary=[]
    palettes=torch.tensor([[0.,0.,0.],[1.,.2,.2],[.2,1.,.2],[.2,.3,1.],[1.,1.,.2],[1.,.2,1.]])
    for split in c['splits']:
        folder=fresh_dir(root/split);x=images_only(split)
        with np.load(HERE/f'data/object_world_v1/{split}/scenes.npz') as a:gt=a['segmentation'].copy();rgb=a['images'].copy()
        eps=torch.randn(len(x),5,32,generator=torch.Generator().manual_seed(c['epsilon_seed']))
        outputs={};rows=[]
        for kind,entry in c['models'].items():
            assert r.sha(HERE/entry['path'])==entry['sha256']
            ck=torch.load(HERE/entry['path'],map_location='cpu',weights_only=True)
            model=LinearImageSlots(kind,ck['dim'],ck['slots']);model.load_state_dict(ck['state_dict']);model.eval()
            pieces={key:[] for key in c['variants'][kind]};images=[]
            for i in range(0,len(x),c['batch']):
                out=model(x[i:i+c['batch']],eps[i:i+c['batch']]);segments=output_segments(out,x[i:i+c['batch']]);images.append(out['image'])
                for key in pieces:pieces[key].append(segments[key])
            outputs[kind]={'image':torch.cat(images),'segmentations':{k:torch.cat(v) for k,v in pieces.items()}}
            for variant,preds in outputs[kind]['segmentations'].items():
                native=variant.endswith('_native')
                for i,pred in enumerate(preds.numpy()):
                    rows.append({'split':split,'kind':kind,'variant':variant,'base_id':i,**score(gt[i],pred,native)})
            examples=[]
            for i in range(12):
                examples.extend([x[i],outputs[kind]['image'][i]])
                for variant in c['variants'][kind]:
                    labels=outputs[kind]['segmentations'][variant][i]+int(variant.endswith('_native'))
                    examples.append(palettes[labels].permute(2,0,1))
            o.grid(torch.stack(examples),folder/f'{kind}_examples.png',2+len(c['variants'][kind]))
        baseline=np.stack([components(im) for im in rgb]);outputs['component']={'segmentations':{'native':torch.from_numpy(baseline)}}
        for i,pred in enumerate(baseline):rows.append({'split':split,'kind':'component','variant':'native','base_id':i,**score(gt[i],pred,False,4)})
        for kind,variant in sorted({(v['kind'],v['variant']) for v in rows}):
            selected=[v for v in rows if v['kind']==kind and v['variant']==variant]
            metrics={key:r.bootstrap([v[key] for v in selected]) for key in ['matched_visible_iou','foreground_ari','all_pixel_ari','matched_region_background_fraction']}
            summary.append({'split':split,'kind':kind,'variant':variant,'scenes':len(selected),'metrics':metrics})
            print(split,kind,variant,{k:round(v['mean'],4) for k,v in metrics.items()},flush=True)
        torch.save({'outputs':outputs,'epsilon':eps},folder/'outputs.pt');r.write_csv(folder/'rows.csv',rows)
        dump(folder/'manifest.json',{'outputs_sha256':r.sha(folder/'outputs.pt'),'rows_sha256':r.sha(folder/'rows.csv'),
            'source_data_sha256':r.sha(HERE/f'data/object_world_v1/{split}/scenes.npz')})
    dump(root/'summary.json',{'results':summary,'config_sha256':r.sha(HERE/f'configs/object_layers_{scope}_v1.json'),
        'claim_boundary':c['claim_boundary']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=['validation','heldout'],required=True);args=parser.parse_args();run(args.scope)
