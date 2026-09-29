"""Autonomous trajectory scoring without teacher forcing after initialization."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'completion_v2'),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
import dynamics as study
from dynamics_models import Transition,encode_states,encode_context
from dynamics_rgb_baseline import measure_frame,infer
from dynamics_physical_baselines import predict as physical_predict
from p0 import dump,sha

def pack(raw):
    slots=np.zeros((len(raw),4,7),np.float32);colors=np.full((len(raw),4),-1,int)
    for i,scene in enumerate(raw):
        present=scene[scene[:,4]>0];present=present[np.argsort(present[:,5],kind='stable')][:4]
        slots[i,:len(present)]=present;colors[i,:len(present)]=present[:,5].astype(int)
    return slots,colors

def measured(path,out):
    cache=out/'rgb_initial.npz'
    if cache.exists():return dict(np.load(cache))
    images=np.load(path)['observations'];raw=[];contexts=[]
    for sequence in images:
        f=np.stack([measure_frame(x) for x in sequence]);s,c,_=infer(f);raw.append(s);contexts.append([*c[:2],0.,0.,c[2]])
    states,colors=pack(np.stack(raw));np.savez_compressed(cache,states=states,colors=colors,contexts=np.asarray(contexts))
    return dict(np.load(cache))

@torch.no_grad()
def rollout(initial,context,actions,model,limit,physical=False):
    s=encode_states(initial);ctx=encode_context(context);out=[]
    for t in range(actions.shape[1]):
        if physical:
            raw=initial.copy();raw[...,:2]=s[...,:2].numpy()*31.5;raw[...,2:4]=s[...,2:4].numpy()*3
            motion=physical_predict(raw,context,actions[:,t])
            normalized=torch.tensor(motion,dtype=torch.float32)/torch.tensor([31.5,31.5,3,3])
            s=torch.cat([normalized,s[...,4:]],-1)
        else:s=study.advance(model,s,torch.tensor(actions[:,t],dtype=torch.float32)/3,ctx,limit)
        out.append(s[...,:4].double().numpy()*np.array([31.5,31.5,3,3]))
    return np.stack(out,1)

def score(pred,truth,colors,targets=None,counter=None,cftruth=None):
    rows=[]
    for i in range(len(pred)):
        original=truth[i,0];ids=np.flatnonzero(original[:,4]>0)
        mapping={j:next((k for k,c in enumerate(colors[i]) if c==int(original[j,5])),None) for j in ids}
        missing=sum(v is None for v in mapping.values())
        for h in [1,16,61]:
            if h>pred.shape[1]:continue
            err=[];blow=False;finite=True;te=[];ne=[];zero_t=[];zero_n=[]
            for j,k in mapping.items():
                if k is None:continue
                trajectory=pred[i,:h,k];finite=finite and bool(np.isfinite(trajectory).all())
                blow=blow or bool(np.any(np.abs(trajectory[:,:2])>64))
                err.append(abs(pred[i,h-1,k]-truth[i,h,j,:4]))
                if counter is not None:
                    delta=counter[i,h-1,k,:2]-pred[i,h-1,k,:2]
                    actual=cftruth[i,h,j,:2]-truth[i,h,j,:2]
                    target=int(targets[i])==j
                    (te if target else ne).append(abs(delta-actual).mean())
                    (zero_t if target else zero_n).append(abs(actual).mean())
            e=np.mean(err,axis=0) if err else np.full(4,np.nan)
            def number(x):return float(x) if np.isfinite(x) else None
            rows.append({'episode':i,'horizon':h,'missing_objects':missing,'nonfinite':not finite,
                         'finite_blowup':blow and finite,'position_mae':number(e[:2].mean()),'velocity_mae':number(e[2:].mean()),
                         'target_response_mae':number(np.mean(te)) if te else None,'nontarget_response_mae':number(np.mean(ne)) if ne else None,
                         'zero_target_response_mae':number(np.mean(zero_t)) if zero_t else None,'zero_nontarget_response_mae':number(np.mean(zero_n)) if zero_n else None})
    summary={}
    for h in sorted({v['horizon'] for v in rows}):
        rr=[v for v in rows if v['horizon']==h];summary[str(h)]={}
        for key in rows[0]:
            if key in ['episode','horizon']:continue
            values=[v[key] for v in rr if v[key] is not None]
            summary[str(h)][key]={'mean':float(np.mean(values)) if values else None,'n':len(values)}
    return rows,summary

def evaluate(seed,init,split='val'):
    study.freeze();root=study.data_root(seed);run=study.BASE/f'runs/d{seed}_s{init}'
    variants={'one':('one',False),'one_bounded':('one',True),'multi':('multi',False),
              'multi_projected':('multi',True),'multi_bounded':('multi_bounded',True),'force_wall':(None,False)}
    for mode in study.MODES:
        if split=='force_ood' and mode!='variable_force':continue
        path=root/mode/split/'trajectories.npz';a=dict(np.load(path))
        out=study.BASE/f'evaluation/d{seed}_s{init}/{mode}/{split}';out.mkdir(parents=True,exist_ok=True)
        for input_kind in (['oracle'] if split=='val' else ['oracle','rgb_measurement']):
            if input_kind=='oracle':initial,colors=pack(a['states'][:,3]);context=a['contexts'].copy()
            else:
                inputs=measured(path,out);initial=inputs['states'];colors=inputs['colors'];context=inputs['contexts']
            commands=np.zeros((len(initial),61,4,2),np.float32)
            for i in range(len(initial)):
                tid=a['target_ids'][i];color=int(a['states'][i,3,tid,5]);slot=np.flatnonzero(colors[i]==color)
                if len(slot):commands[i,0,slot[0]]=a['actions'][i,3,tid]
            for variant,(kind,bounded) in variants.items():
                folder=out/f'{input_kind}_{variant}';folder.mkdir(exist_ok=True)
                if (folder/'summary.json').exists():continue
                model=None;limit=None;checkpoint=None
                if kind is not None:
                    checkpoint=run/kind/'model.pt'
                    if not checkpoint.exists():continue
                    ck=torch.load(checkpoint,weights_only=True);model=Transition('joint_exact');model.load_state_dict(ck['state_dict']);model.eval()
                    limit=ck['velocity_limit'] if bounded else None
                pred=rollout(initial,context,commands,model,limit,variant=='force_wall')
                counter=rollout(initial,context,-commands,model,limit,variant=='force_wall') if 'counterfactual_states' in a else None
                truth=a['states'][:,3:];cf=a.get('counterfactual_states');cf=cf[:,3:] if cf is not None else None
                rows,summary=score(pred,truth,colors,a['target_ids'],counter,cf)
                saved={'predictions':pred,'initial':initial,'context':context,'actions':commands,'colors':colors}
                if counter is not None:saved['counterfactual']=counter
                np.savez_compressed(folder/'predictions.npz',**saved);dump(folder/'rows.json',rows)
                dump(folder/'summary.json',{'summary':summary,'variant':variant,'input':input_kind,'mode':mode,'split':split,
                                           'data_seed':seed,'init':init,'data_sha256':sha(path),'source_sha256':sha(Path(__file__)),
                                           'checkpoint_sha256':sha(checkpoint) if checkpoint else None,'predictions_sha256':sha(folder/'predictions.npz'),
                                           'claim':'Errors are means over matched finite objects; missing objects and nonfinite/blow-up paths separately reported. RGB measurement uses known palette/circle/physics priors.'})
                print('dynamics score',seed,init,mode,split,input_kind,variant,summary['61']['position_mae'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=640031);p.add_argument('--init',type=int,default=0)
    p.add_argument('--split',default='val');a=p.parse_args();torch.set_num_threads(2);evaluate(a.seed,a.init,a.split)
