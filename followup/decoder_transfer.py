"""Decode identical saved flow latents with original and freshly trained decoders.

No generator retraining or new noise selection. Exploratory reuse of diagnostic
samples; original decoder is a descriptive reference with different training
history. The two new decoders are the controlled comparison.
"""
import json
import time
from common import HERE, ROOT, dump, fresh_dir, digest_tensor, torch, np, o, r
from decoder_study import ControlledDecoder
from evaluator_v2 import DetailedEvaluator, accepted


@torch.no_grad()
def main():
    cp=HERE/'configs/decoder_transfer_v1.json'
    if not cp.exists():dump(cp,{'seeds':[0,1,2],'decoders':['original','pixel_mean','logit_mean'],
        'latent_source':'Completed fixed-step C4 flows on original C4 AE; 64-step saved samples used without retraining.',
        'selection':'All saved samples, identical latent and condition per decoder, no cherry-picking.',
        'boundary':'Exploratory decoder intervention on previously evaluated noise. Fresh decoder comparison has matched training; original training history differs.',
        'frozen_unix':time.time()})
    cfg=json.loads(cp.read_text());root=fresh_dir(HERE/'reports/decoder_transfer_v1')
    device=o.choose_device('mps');torch.set_num_threads(4);ev=DetailedEvaluator()
    threshold=json.loads((HERE/'reports/evaluator_calibration_v1/locked_threshold.json').read_text());runs=[]
    for seed in cfg['seeds']:
        flow_folder=HERE/f'runs/flow_fixed_s{seed}_equivariant'
        flow_ck=torch.load(flow_folder/'flow.pt',map_location='cpu',weights_only=True)
        data=torch.load(flow_folder/'samples/samples.pt',map_location='cpu',weights_only=True)
        z=data['latents'][64];labels=data['labels'];ae_path=ROOT/f'runs/repair_equivariant_n1024_s{seed}/ae.pt'
        ae,_=r.load_model(ae_path,device);assert flow_ck['ae_sha256']==r.sha(ae_path)
        for kind in cfg['decoders']:
            folder=root/f'{kind}_s{seed}';folder.mkdir();decoder_path=ae_path
            if kind!='original':
                decoder_path=HERE/f'runs/decoder_study_v1/learned_{kind}_s{seed}/decoder.pt'
                ck=torch.load(decoder_path,map_location='cpu',weights_only=True)
                assert ck['ae_sha256']==r.sha(ae_path)
                model=ControlledDecoder(kind).to(device);model.load_state_dict(ck['state_dict']);model.eval()
            images=[]
            for i in range(0,len(z),64):
                zz=z[i:i+64].to(device)
                if kind=='original':images.append(ae.decode(zz).cpu())
                else:images.append(model((zz-ck['mean'].to(device))/ck['std'].to(device)).cpu())
            images=torch.cat(images);rows=[]
            for i,(p,label) in enumerate(zip(ev.predict(images),labels.tolist())):
                joint=p['predicted_shape']==label[0] and p['predicted_color']==label[1]
                rows.append({'sample':i,'shape':label[0],'color':label[1],'ood':label[0]==label[1],**p,
                    'joint_correct':joint,'strict_accepted_and_joint':bool(joint and accepted(p,threshold))})
            r.write_csv(folder/'rows.csv',rows);o.grid(images[:48],folder/'samples.png',8)
            m={split:{'n':len(rr),**{key:float(np.mean([v[key] for v in rr])) for key in ['joint_correct','strict_accepted_and_joint','template_iou']}}
               for split,rr in [('seen',[v for v in rows if not v['ood']]),('ood',[v for v in rows if v['ood']])]}
            run={'kind':kind,'seed':seed,'metrics':m,'raw_latent_sha256':digest_tensor(z),'noise_sha256':digest_tensor(data['noise']),
                 'decoder_sha256':r.sha(decoder_path),'ae_sha256':r.sha(ae_path),'flow_sha256':r.sha(flow_folder/'flow.pt')}
            dump(folder/'metrics.json',run);runs.append(run);print(kind,seed,m,flush=True)
    summary={kind:{f'{split}_{key}':{'mean':float(np.mean([v['metrics'][split][key] for v in runs if v['kind']==kind])),
                 'per_seed':[v['metrics'][split][key] for v in runs if v['kind']==kind]}
                 for split in ['seen','ood'] for key in ['joint_correct','strict_accepted_and_joint']} for kind in cfg['decoders']}
    for seed in cfg['seeds']:
        for key in ['raw_latent_sha256','noise_sha256','ae_sha256','flow_sha256']:assert len({v[key] for v in runs if v['seed']==seed})==1
    dump(root/'summary.json',{'summary':summary,'runs':runs,'paired_latents_verified':True,'new_training_runs':0,
        'config_sha256':r.sha(cp),'code_sha256':r.sha(__file__),'claim_boundary':cfg['boundary']})


if __name__=='__main__':main()
