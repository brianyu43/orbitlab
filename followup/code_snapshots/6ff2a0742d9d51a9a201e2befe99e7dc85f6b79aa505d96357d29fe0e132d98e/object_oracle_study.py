"""Equal-parameter flat vs shared-object residual predictors on true state inputs.

This diagnoses the inductive bias of object factorization, not object discovery.
Both predictors learn the update and preservation; neither gets a hard-coded
target-only update gate. The analytic rule is a separately named upper bound.
"""
from __future__ import annotations
import argparse
import json
import time
import torch.nn.functional as F
from torch import nn
from common import HERE, dump, fresh_dir, digest_tensor, state_digest, update_status, np, torch, o, r
from object_world import ObjectState, render
from object_tasks import load, analytic_step


class Predictor(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='flat':sizes=[81,124,131,64]
        elif kind=='object':sizes=[30,166,163,16]
        else:raise ValueError(kind)
        self.net=nn.Sequential(nn.Linear(sizes[0],sizes[1]),nn.SiLU(),nn.Linear(sizes[1],sizes[2]),nn.SiLU(),nn.Linear(sizes[2],sizes[3]))

    def forward(self,x,mask,c):
        if self.kind=='flat':delta=self.net(torch.cat([x.flatten(1),mask,c],1)).reshape_as(x)
        else:delta=self.net(torch.cat([x,mask[:,:,None],c[:,None,:].expand(-1,4,-1)],-1))
        return x+delta


def permute_slots(x,y,mask,rng):
    permutation=torch.rand((len(x),4),generator=rng).argsort(1)
    idx=permutation[:,:,None].expand(-1,-1,16)
    return x.gather(1,idx),y.gather(1,idx),mask.gather(1,permutation)


def factor_checks(pred,target):
    """Categorical equality; xy <= 1 pixel; radius <= .5 pixel; presence threshold .5."""
    directions=pred.new_tensor([[1,0],[0,-1],[-1,0],[0,1]])
    return torch.stack([pred[:,:,:4].argmax(-1)==target[:,:,:4].argmax(-1),
        pred[:,:,4:10].argmax(-1)==target[:,:,4:10].argmax(-1),
        (pred[:,:,10]-target[:,:,10]).abs()*31.5<=1,
        (pred[:,:,11]-target[:,:,11]).abs()*31.5<=1,
        (pred[:,:,12]-target[:,:,12]).abs()*8<=.5,
        (pred[:,:,13:15]@directions.T).argmax(-1)==(target[:,:,13:15]@directions.T).argmax(-1),
        (pred[:,:,15]>=.5)==(target[:,:,15]>=.5)],-1)


def measure(pred,data,records):
    y=data['target_states'];x=data['source_states'];check=factor_checks(pred,y);present=y[:,:,15]>.5
    rows=[]
    for i,e in enumerate(records):
        tid=int(data['target_ids'][i]);target_check=check[i,tid]
        edited={'color':[1],'novel_color_pair':[1],'rotate':[5],'translate':[2,3],'color_then_rotate':[1,5]}[e['operation']]
        untouched=[k for k in range(7) if k not in edited]
        other=[k for k in range(4) if k!=tid and present[i,k]]
        empty=~present[i]
        target_ok=bool(target_check[edited].all());same_target=bool(target_check[untouched].all())
        preserve=bool(check[i,other].all());empty_ok=bool((pred[i,empty,15]<.5).all())
        rows.append({'task_index':i,'source_index':e['source_index'],'operation':e['operation'],'target_id':tid,
            'source_target_pair_is_unseen':e['source_target_pair_is_unseen'],'target_pair_is_unseen':e['target_pair_is_unseen'],
            'target_success':target_ok,'target_other_preserved':same_target,'non_target_preserved':preserve,
            'empty_slots_preserved':empty_ok,'full_scene_success':target_ok and same_target and preserve and empty_ok,
            'state_mse':float((pred[i]-y[i]).square().mean()),
            'target_position_error_pixels':float((pred[i,tid,10:12]-y[i,tid,10:12]).abs().mean()*31.5),
            'non_target_position_drift_pixels':float((pred[i,other,10:12]-x[i,other,10:12]).abs().mean()*31.5)})
    metrics=['target_success','target_other_preserved','non_target_preserved','empty_slots_preserved','full_scene_success',
             'state_mse','target_position_error_pixels','non_target_position_drift_pixels']
    groups={'all':rows}
    for op in sorted({v['operation'] for v in rows}):groups[op]=[v for v in rows if v['operation']==op]
    for label,key in [('unseen_source_target','source_target_pair_is_unseen'),('unseen_output_target','target_pair_is_unseen')]:
        selected=[v for v in rows if v[key]]
        if selected:groups[label]=selected
    summary={}
    for group,rr in groups.items():
        ids=sorted({v['source_index'] for v in rr})
        summary[group]={'tasks':len(rr),'source_scenes':len(ids),**{key:r.bootstrap([np.mean([v[key] for v in rr if v['source_index']==j]) for j in ids]) for key in metrics}}
    return rows,summary


def image_from_slots(slots):
    values=slots.detach().cpu().numpy();objects=[];dirs=np.array([[1,0],[0,-1],[-1,0],[0,1]])
    # Only used for fixed example grids. Metrics retain raw predictions and penalize presence/position errors.
    for v in values:
        if v[15]<.5:continue
        objects.append(ObjectState(len(objects),int(v[:4].argmax()),int(v[4:10].argmax()),
            int(np.clip(np.rint(v[10]*31.5+31.5),-32,95)),int(np.clip(np.rint(v[11]*31.5+31.5),-32,95)),
            int(np.clip(np.rint(v[12]*8),1,12)),int((dirs@v[13:15]).argmax())))
    return render(tuple(objects))['image'] if objects else np.zeros((64,64,3),np.uint8)


def freeze():
    cp=HERE/'configs/object_oracle_study_v1.json'
    if not cp.exists():dump(cp,{'seeds':[0,1,2],'kinds':['flat','object'],'steps':6000,'batch':128,'lr':.001,'device':'cpu',
        'data_manifest_sha256':r.sha(HERE/'data/object_edit_tasks_v1/manifest.json'),'parameters_each':34991,
        'information':'True shape/color/center/radius/local pose/presence per slot, target mask, same command. No image perception.',
        'matching':'Exact parameter counts; same sampled task indices and slot permutations, optimizer, loss and steps. Architecture-specific initial weights.',
        'permutation':'Shuffle all four slots including padding on every training batch; both target positions and occupied positions get training exposure.',
        'output':'Both predict unrestricted residuals for every slot. No analytic preservation gate in either learned model.',
        'composition':'Apply learned primitive transition twice without teacher forcing. No composed training tasks.',
        'metrics':'Target operation success; target untouched factors; non-target factors; empty slot presence; joint scene success.',
        'tolerances':{'position_pixels':1,'radius_pixels':.5,'presence_threshold':.5},
        'selection':'Fixed final step, no test selection. Three initializations on one source dataset; developmental experiment.',
        'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def train(c,kind,seed):
    folder=HERE/f'runs/object_oracle_study_v1/{kind}_s{seed}'
    if (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text());assert r.sha(folder/'model.pt')==run['checkpoint_sha256'];return folder
    fresh_dir(folder);data,records=load('train');x,y,commands=data['source_states'],data['target_states'],data['commands'][:,0]
    mask=F.one_hot(data['target_ids'],4).float();assert not data['commands'][:,1].any()
    assert {v['operation'] for v in records}=={'color','rotate','translate'}
    o.seed_all(974000+seed);model=Predictor(kind);assert sum(v.numel() for v in model.parameters())==c['parameters_each']
    initial=state_digest(model);opt=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=1e-4)
    rng=torch.Generator().manual_seed(975000+seed);coverage=torch.zeros(4,dtype=torch.int64);occupied=coverage.clone()
    dump(folder/'started.json',{'kind':kind,'seed':seed,'config_sha256':r.sha(HERE/'configs/object_oracle_study_v1.json'),'code_sha256':r.sha(__file__)})
    logs=[];start=time.perf_counter()
    for step in range(1,c['steps']+1):
        idx=torch.randint(len(x),(c['batch'],),generator=rng)
        bx,by,bmask=permute_slots(x[idx],y[idx],mask[idx],rng);coverage+=bmask.sum(0).long();occupied+=bx[:,:,15].sum(0).long()
        pred=model(bx,bmask,commands[idx]);loss=F.mse_loss(pred,by)
        if not torch.isfinite(loss):raise FloatingPointError('State loss')
        opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('State gradient')
        opt.step()
        if step%1000==0:
            logs.append({'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-start});print(folder.name,logs[-1],flush=True)
    seconds=time.perf_counter()-start
    torch.save({'state_dict':model.state_dict(),'kind':kind,'seed':seed},folder/'model.pt')
    torch.save({'optimizer':opt.state_dict(),'sampling_rng':rng.get_state()},folder/'optimizer.pt')
    dump(folder/'run.json',{'kind':kind,'seed':seed,'parameters':c['parameters_each'],'steps':c['steps'],'train_seconds':seconds,
        'initial_weights_sha256':initial,'source_state_sha256':digest_tensor(x),'target_state_sha256':digest_tensor(y),
        'target_slot_exposure':coverage.tolist(),'occupied_slot_exposure':occupied.tolist(),'sampling_rng_final_sha256':digest_tensor(rng.get_state()),
        'checkpoint_sha256':r.sha(folder/'model.pt'),'config_sha256':r.sha(HERE/'configs/object_oracle_study_v1.json'),
        'code_sha256':r.sha(__file__),'loss_log':logs})
    return folder


@torch.no_grad()
def predict(model,data):
    outputs=[]
    for i in range(0,len(data['source_states']),256):
        x=data['source_states'][i:i+256].clone();mask=F.one_hot(data['target_ids'][i:i+256],4).float();commands=data['commands'][i:i+256]
        for j in range(2):
            c=commands[:,j];active=c[:,:3].sum(1)>0
            if model=='identity':p=x
            elif model=='analytic':p=analytic_step(x,mask,c)
            else:p=model(x,mask,c)
            x=torch.where(active[:,None,None],p,x)
        outputs.append(x)
    return torch.cat(outputs)


@torch.no_grad()
def evaluate(folder,baseline=None):
    if (folder/'metrics.json').exists():return
    if baseline:model=baseline;folder.mkdir(parents=True,exist_ok=True)
    else:
        ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);model=Predictor(ck['kind']);model.load_state_dict(ck['state_dict']);model.eval()
    results={};outputs={}
    for split in ['val','test','ood','count3','count4','occlusion']:
        data,records=load(split);pred=predict(model,data);rows,m=measure(pred,data,records)
        r.write_csv(folder/f'{split}_rows.csv',rows);results[split]=m;outputs[split]=pred
        images=[]
        for i in range(0,min(12,len(pred))):
            images.extend([image_from_slots(data['source_states'][i]),image_from_slots(data['target_states'][i]),image_from_slots(pred[i])])
        o.grid(torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255,folder/f'{split}_examples.png',6)
        print('eval',folder.name,split,'target',round(m['all']['target_success']['mean'],4),'full',round(m['all']['full_scene_success']['mean'],4),flush=True)
    torch.save(outputs,folder/'predictions.pt')
    dump(folder/'metrics.json',{'splits':results,'baseline':baseline,'prediction_sha256':r.sha(folder/'predictions.pt'),
        'checkpoint_sha256':None if baseline else r.sha(folder/'model.pt'),'data_manifest_sha256':r.sha(HERE/'data/object_edit_tasks_v1/manifest.json')})


def summarize(c):
    rows=[]
    for kind in c['kinds']:
        for seed in c['seeds']:
            folder=HERE/f'runs/object_oracle_study_v1/{kind}_s{seed}';run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
            rows.append({'kind':kind,'seed':seed,'parameters':run['parameters'],'steps':run['steps'],'sampling_sha':run['sampling_rng_final_sha256'],
                **{f'{s}_{key}':m['splits'][s]['all'][key]['mean'] for s in ['test','ood','count3','count4','occlusion']
                   for key in ['target_success','non_target_preserved','full_scene_success']}})
    for seed in c['seeds']:
        pair=[v for v in rows if v['seed']==seed]
        for key in ['parameters','steps','sampling_sha']:assert len({v[key] for v in pair})==1
    root=HERE/'reports/object_oracle_study_v1';root.mkdir(exist_ok=True)
    r.write_csv(root/'runs.csv',rows)
    summary={kind:{key:{'mean':float(np.mean([v[key] for v in rows if v['kind']==kind])),
                       'seed_sd':float(np.std([v[key] for v in rows if v['kind']==kind],ddof=1))}
                     for key in rows[0] if key not in ['kind','seed','parameters','steps','sampling_sha']} for kind in c['kinds']}
    dump(root/'summary.json',{'summary':summary,'trainings':len(rows),'matching_verified':True,
        'claim_boundary':'Privileged true-state intervention diagnostic, not image understanding or unsupervised object discovery. Composition is sequential primitive execution.',
        'config_sha256':r.sha(HERE/'configs/object_oracle_study_v1.json')})
    update_status('B03','complete',['reports/object_oracle_study_v1/summary.json'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args();c=freeze()
    if a.prepare_only:return
    torch.set_num_threads(2);update_status('B03','running',['configs/object_oracle_study_v1.json'])
    for baseline in ['identity','analytic']:evaluate(HERE/f'runs/object_oracle_study_v1/{baseline}',baseline)
    for seed in c['seeds']:
        for kind in c['kinds']:evaluate(train(c,kind,seed))
    summarize(c)


if __name__=='__main__':main()
