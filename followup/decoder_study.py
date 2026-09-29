"""Controlled decoder experiment: average probabilities vs average logits.

Both decoders have exactly the same parameters and C4 action. The learned code
encoder is frozen. A renderer-state oracle is a separately labelled privileged
information diagnostic, not a competing unsupervised representation.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch.nn.functional as F
from torch import nn
from common import HERE, ROOT, dump, fresh_dir, state_digest, digest_tensor, update_status, torch, np, o, r
from diagnostics import probe_data
from evaluator_v2 import DetailedEvaluator, accepted


class ControlledDecoder(nn.Module):
    def __init__(self,kind):
        super().__init__()
        if kind not in ['pixel_mean','logit_mean']:raise ValueError(kind)
        self.kind=kind;self.branch=o.Decoder(64,16)

    def forward(self,z):
        logits=self.branch.fc(z.reshape(-1,16)).reshape(-1,32,8,8)
        for layer in list(self.branch.net)[:-1]:logits=layer(logits)
        logits=logits.reshape(len(z),4,3,64,64)
        if self.kind=='pixel_mean':
            return torch.stack([o.rotate(logits[:,k].sigmoid(),k) for k in range(4)]).mean(0)
        return torch.stack([o.rotate(logits[:,k],k) for k in range(4)]).mean(0).sigmoid()


def rot_vector(x,k):
    k=k%4
    if k==0:return x
    if k==1:return torch.stack([x[:,1],-x[:,0]],1)
    if k==2:return -x
    return torch.stack([-x[:,1],x[:,0]],1)


def oracle_codes(labels,metadata,rotation):
    """Renderer factors in a regular representation with exactly 4x16 entries."""
    fixed=torch.cat([F.one_hot(labels[:,0],4),F.one_hot(labels[:,1],6)],1).float()
    center=torch.tensor([[2*m['cx']-1,2*m['cy']-1] for m in metadata])
    radius=torch.tensor([[m['radius']/.24] for m in metadata])
    direction=torch.tensor([[1.,0.]]).expand(len(labels),2)
    return torch.stack([torch.cat([fixed,rot_vector(center,rotation-k),radius,
                                  rot_vector(direction,rotation-k),torch.ones(len(labels),1)],1) for k in range(4)],1)


def freeze():
    p=HERE/'configs/decoder_study_v1.json'
    if not p.exists():
        dump(p,{'ae_seeds':[0,1,2],'roles':['learned','oracle'],'decoders':['pixel_mean','logit_mean'],
            'steps':6000,'batch':32,'lr':.001,'init_seed_offset':86000,'sampling_seed_offset':87000,
            'train_n':1024,'eval_n':256,'data_seed':42,'selection':'Fixed final step; no validation/test model selection.',
            'purpose':'Exploratory mechanism test. The original C4 encoder is frozen and all decoders are freshly initialized.',
            'matched':'Exact decoder parameter count and initial weights, train scenes, sampled indices, optimizer steps.',
            'oracle_privilege':'True shape/color/position/radius/global orientation; not an unsupervised model.',
            'normalization':'Each role normalized by its own train-only group-shared channel mean/std; equivariance retained.',
            'strict_threshold_sha256':r.sha(HERE/'reports/evaluator_calibration_v1/locked_threshold.json'),
            'frozen_unix':time.time()})
    return json.loads(p.read_text())


@torch.no_grad()
def prepare_codes(seed,role,device):
    ae_path=ROOT/f'runs/repair_equivariant_n1024_s{seed}/ae.pt'
    ae,_=r.load_model(ae_path,device);ae.requires_grad_(False)
    result={}
    for split in ['train','val','test','ood']:
        if split=='train':x,y,meta=r.load_data('train',1024)
        else:
            x,y=probe_data(split)
            meta=json.loads((HERE/f'data/probe_v1/{split}_metadata.json').read_text())
        zs=[]
        for k in range(4):
            if role=='oracle':zs.append(oracle_codes(y,meta,k))
            else:zs.append(torch.cat([ae.encode(o.rotate(x[i:i+64].to(device),k)).cpu() for i in range(0,len(x),64)]))
        result[split]={'z':torch.cat(zs),'x':torch.cat([o.rotate(x,k) for k in range(4)]),'labels':y.repeat(4,1),'base_n':len(x)}
    z=result['train']['z'];mean=z.mean((0,1),keepdim=True);std=z.std((0,1),unbiased=False,keepdim=True).clamp_min(.02)
    for d in result.values():d['z']=(d['z']-mean)/std
    return result,mean,std,ae_path


def train(c,seed,role,kind,data,mean,std,ae_path,device):
    out=HERE/f'runs/decoder_study_v1/{role}_{kind}_s{seed}'
    if (out/'run.json').exists():
        run=json.loads((out/'run.json').read_text());assert r.sha(out/'decoder.pt')==run['checkpoint_sha256'];return out
    fresh_dir(out);o.seed_all(c['init_seed_offset']+seed);model=ControlledDecoder(kind).to(device)
    initial=state_digest(model);opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=1e-4)
    rng=torch.Generator().manual_seed(c['sampling_seed_offset']+seed)
    x,z=data['train']['x'],data['train']['z'];logs=[];o.sync(device);start=time.perf_counter()
    dump(out/'started.json',{'seed':seed,'role':role,'kind':kind,'config_sha256':r.sha(HERE/'configs/decoder_study_v1.json'),'code_sha256':r.sha(__file__)})
    for step in range(1,c['steps']+1):
        idx=torch.randint(len(x),(c['batch'],),generator=rng);target=x[idx].to(device)
        pred=model(z[idx].to(device));loss=o.reconstruction_loss(pred,target)
        if not torch.isfinite(loss):raise FloatingPointError('Decoder loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Decoder gradient')
        opt.step()
        if step%1000==0:
            o.sync(device);logs.append({'step':step,'loss':loss.detach().item(),'seconds':time.perf_counter()-start})
            print(out.name,logs[-1],flush=True)
    o.sync(device);elapsed=time.perf_counter()-start
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'kind':kind,'role':role,
        'seed':seed,'mean':mean,'std':std,'ae_path':str(ae_path),'ae_sha256':r.sha(ae_path)},out/'decoder.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},out/'optimizer.pt')
    dump(out/'run.json',{'seed':seed,'role':role,'kind':kind,'steps':c['steps'],'train_wall_seconds':elapsed,
        'parameters':sum(p.numel() for p in model.parameters()),'initial_weights_sha256':initial,
        'train_image_sha256':digest_tensor(x),'train_latent_sha256':digest_tensor(z),'ae_sha256':r.sha(ae_path),
        'encoder_training':'frozen original AE for learned role; no learned encoder for privileged oracle',
        'loss_log':logs,'checkpoint_sha256':r.sha(out/'decoder.pt'),'code_sha256':r.sha(__file__),
        'config_sha256':r.sha(HERE/'configs/decoder_study_v1.json')})
    return out


@torch.no_grad()
def evaluate(out,data,device):
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    ck=torch.load(out/'decoder.pt',map_location='cpu',weights_only=True)
    model=ControlledDecoder(ck['kind']).to(device);model.load_state_dict(ck['state_dict']);model.eval()
    ev=DetailedEvaluator();threshold=json.loads((HERE/'reports/evaluator_calibration_v1/locked_threshold.json').read_text())
    rows=[];summary={};max_eq=0.
    for split in ['val','test','ood']:
        d=data[split];x,z,labels=d['x'],d['z'],d['labels'];sr=[]
        for i in range(0,len(x),64):
            b=z[i:i+64].to(device);target=x[i:i+64].to(device);pred=model(b);pm=r.pixel_metrics(pred,target)
            if i==0:
                for k in range(4):max_eq=max(max_eq,float((model(o.rho(b,k))-o.rotate(pred,k)).abs().max()))
                o.grid(torch.cat([target[:8],pred[:8]]),out/f'{split}_reconstruction.png',8)
            for j,(p,y) in enumerate(zip(ev.predict(pred),labels[i:i+64].tolist())):
                joint=p['predicted_shape']==y[0] and p['predicted_color']==y[1]
                row={'split':split,'base_id':(i+j)%d['base_n'],'rotation':(i+j)//d['base_n'],
                    'shape_correct':int(p['predicted_shape']==y[0]),'color_correct':int(p['predicted_color']==y[1]),
                    'joint_correct':int(joint),'strict_accepted_and_joint':int(joint and accepted(p,threshold)),
                    **{key:float(v[j]) for key,v in pm.items()}}
                rows.append(row);sr.append(row)
        summary[split]={key:r.bootstrap(np.array([v[key] for v in sr]).reshape(4,-1).mean(0))
                       for key in ['mse','foreground_mae','mask_iou','shape_correct','color_correct','strict_accepted_and_joint']}
    assert max_eq<1e-5
    r.write_csv(out/'rows.csv',rows)
    dump(out/'metrics.json',{'splits':summary,'max_c4_equivariance_error':max_eq,'checkpoint_sha256':r.sha(out/'decoder.pt')})
    return summary


def summarize(c):
    rows=[]
    for seed in c['ae_seeds']:
        for role in c['roles']:
            pair=[]
            for kind in c['decoders']:
                folder=HERE/f'runs/decoder_study_v1/{role}_{kind}_s{seed}'
                run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
                row={'seed':seed,'role':role,'kind':kind,'parameters':run['parameters'],'train_seconds':run['train_wall_seconds'],
                     'initial_sha':run['initial_weights_sha256'],'train_image_sha':run['train_image_sha256'],'train_latent_sha':run['train_latent_sha256'],
                     **{f'{s}_{key}':m['splits'][s][key]['mean'] for s in ['test','ood'] for key in ['foreground_mae','mask_iou','shape_correct','strict_accepted_and_joint']}}
                rows.append(row);pair.append(row)
            for key in ['parameters','initial_sha','train_image_sha','train_latent_sha']:assert len({v[key] for v in pair})==1
    out=HERE/'reports/decoder_study_v1';out.mkdir(exist_ok=True)
    r.write_csv(out/'runs.csv',rows)
    summary={f'{role}/{kind}':{key:{'mean':float(np.mean([v[key] for v in rows if v['role']==role and v['kind']==kind])),
                    'seed_sd':float(np.std([v[key] for v in rows if v['role']==role and v['kind']==kind],ddof=1))}
                    for key in ['test_foreground_mae','test_mask_iou','test_shape_correct','test_strict_accepted_and_joint','ood_strict_accepted_and_joint','train_seconds']}
                    for role in c['roles'] for kind in c['decoders']}
    dump(out/'summary.json',{'summary':summary,'runs':rows,'decoder_trainings':12,'matching_verified':True,
        'claim_boundary':'Mechanism test of sigmoid placement and a renderer-state oracle; not a new generation benchmark or proof that probability averaging causes blur.',
        'config_sha256':r.sha(HERE/'configs/decoder_study_v1.json')})
    update_status('A10','complete',['reports/decoder_study_v1/summary.json'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args();c=freeze()
    if a.prepare_only:print('decoder protocol frozen');return
    torch.set_num_threads(4);device=o.choose_device('mps');update_status('A10','running',['configs/decoder_study_v1.json'])
    for seed in c['ae_seeds']:
        for role in c['roles']:
            d,mean,std,ae_path=prepare_codes(seed,role,device)
            for kind in c['decoders']:
                folder=train(c,seed,role,kind,d,mean,std,ae_path,device);evaluate(folder,d,device)
    summarize(c)


if __name__=='__main__':main()
