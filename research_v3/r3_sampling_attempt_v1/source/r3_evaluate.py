"""All-path autonomous scoring with force and impulse counterfactual controls."""
import argparse,json
import numpy as np
import torch
import r3_core as c
from dynamics_physical_baselines import predict as force_wall
from v3_common import dump,sha,event

@torch.no_grad()
def rollout(initial,context,actions,model,arm,limit):
    s=c.encode_states(initial);ctx=c.encode_context(context);pred=[]
    for t in range(actions.shape[1]):
        if arm=='force_wall':
            raw=initial.copy();raw[...,:2]=s[...,:2].numpy()*31.5;raw[...,2:4]=s[...,2:4].numpy()*3
            motion=torch.tensor(force_wall(raw,context,actions[:,t]),dtype=torch.float32)/torch.tensor([31.5,31.5,3,3])
            s=torch.cat([motion,s[...,4:]],-1)
        else:s=c.advance(model,s,torch.tensor(actions[:,t],dtype=torch.float32)/3,ctx,arm,limit)
        pred.append(s[...,:4].double().numpy()*np.array([31.5,31.5,3,3]))
    return np.stack(pred,1)

def geometry(pred,radii):
    xy=pred[...,:2]
    wall=np.maximum(np.abs(xy)-(31.5-radii[:,None,:,None]),0).max((2,3))
    overlap=np.zeros(pred.shape[:2]);near=np.zeros(pred.shape[:2],bool)
    for i in range(pred.shape[2]):
        for j in range(i+1,pred.shape[2]):
            gap=np.linalg.norm(xy[:,:,i]-xy[:,:,j],axis=-1)-radii[:,None,i]-radii[:,None,j]
            overlap=np.maximum(overlap,-gap);near|=gap<=.5
    return wall,overlap,near

def rows_for(pred,reverse,force,truth,reverse_truth,force_truth,radii,targets,strata):
    wall,overlap,near=geometry(pred,radii);_,_,gt_near=geometry(truth,radii);rows=[]
    for i in range(len(pred)):
        target=int(targets[i]);others=[j for j in range(pred.shape[2]) if j!=target]
        for h in [1,16,61,128]:
            finite=bool(np.isfinite(pred[i,:h]).all());blow=bool(np.any(np.abs(pred[i,:h,:,:2])>64));failed=not finite or blow
            e=np.abs(pred[i,h-1]-truth[i,h-1])
            cf_fail=not np.isfinite(reverse[i,:h]).all() or np.any(np.abs(reverse[i,:h,:,:2])>64)
            f_fail=not np.isfinite(force[i,:h]).all() or np.any(np.abs(force[i,:h,:,:2])>64)
            actual=reverse_truth[i,h-1,:,:2]-truth[i,h-1,:,:2]
            response=(reverse[i,h-1,:,:2]-pred[i,h-1,:,:2])-actual
            force_response=(force[i,h-1,:,:2]-pred[i,h-1,:,:2])-(force_truth[i,h-1,:,:2]-truth[i,h-1,:,:2])
            first=lambda flags: int(np.flatnonzero(flags)[0]+1) if flags.any() else h+1
            rows.append({'episode':i,'horizon':h,'stratum':int(strata[i]),'nonfinite':float(not finite),'finite_blowup':float(finite and blow),'failure':float(failed),
                'position_mae_all':128. if failed else float(e[:,:2].mean()),'velocity_mae_all':16. if failed else float(e[:,2:].mean()),
                'position_mae_finite':float(e[:,:2].mean()) if finite else None,
                'wall_over_1px':float(finite and np.any(wall[i,:h]>1)), 'overlap_over_1px':float(finite and np.any(overlap[i,:h]>1)),
                'near_contact_trace_error':float(np.mean(near[i,:h]!=gt_near[i,:h])) if not failed else 1.,
                'first_near_contact_timing_error':float(abs(first(near[i,:h])-first(gt_near[i,:h]))) if not failed else float(h+1),
                'target_response_mae':128. if failed or cf_fail else float(np.abs(response[target]).mean()),
                'nontarget_response_mae':128. if failed or cf_fail else float(np.abs(response[others]).mean()),
                'zero_target_response_mae':float(np.abs(actual[target]).mean()),'zero_nontarget_response_mae':float(np.abs(actual[others]).mean()),
                'force_response_mae':128. if failed or f_fail else float(np.abs(force_response).mean()),
                'zero_force_response_mae':float(np.abs(force_truth[i,h-1,:,:2]-truth[i,h-1,:,:2]).mean()),
                'reversed_failure':float(cf_fail),'force_failure':float(f_fail)})
    return rows

def summarize(rows):
    result={}
    for h in [1,16,61,128]:
        result[str(h)]={}
        for stratum in ['all',0,1,2]:
            rr=[v for v in rows if v['horizon']==h and (stratum=='all' or v['stratum']==stratum)]
            keys=[k for k in rr[0] if k not in ['episode','horizon','stratum']]
            result[str(h)][str(stratum)]={k:float(np.mean([v[k] for v in rr if v[k] is not None])) if any(v[k] is not None for v in rr) else None for k in keys}
            result[str(h)][str(stratum)]['episodes']=len(rr)
    return result

def evaluate(seed,init,arm):
    c.freeze();torch.set_num_threads(2);model=None;limit=None
    if arm in c.ARMS:
        cp=c.BASE/f'runs/d{seed}_s{init}/{arm}/model.pt';ck=torch.load(cp,weights_only=True);model=c.model_for(arm);model.load_state_dict(ck['state_dict']);model.eval();limit=ck['velocity_limit'];model_sha=sha(cp)
    else:model_sha='nonlearned_'+arm
    split='val' if seed==c.DEVELOPMENT else 'test'
    for count in [2,3,4,6]:
        root=c.BASE/f'data/d{seed}/{split}/n{count}';a=np.load(root/'trajectories.npz');initial=a['states'][:,3];actions=a['actions'][:,3:]
        for input_kind in ['known_force','rgb_estimated_force']:
            out=c.BASE/f'evaluation/d{seed}_s{init}/{arm}/n{count}/{input_kind}';out.mkdir(parents=True,exist_ok=True)
            if (out/'summary.json').exists():
                old=json.loads((out/'summary.json').read_text());assert old['model_sha256']==model_sha;continue
            context=a['contexts'].copy() if input_kind=='known_force' else np.load(root/'rgb_context.npz')['contexts']
            changed_context=context.copy();changed_context[:,2]+=.04
            pred=rollout(initial,context,actions,model,arm,limit)
            reverse=rollout(initial,context,-actions,model,arm,limit)
            force=rollout(initial,changed_context,actions,model,arm,limit)
            np.savez_compressed(out/'predictions.npz',prediction=pred,reversed=reverse,force=force,initial=initial,context=context,actions=actions)
            rows=rows_for(pred,reverse,force,a['states'][:,4:,:,:4],a['reversed_states'][:,1:,:,:4],a['force_states'][:,1:,:,:4],initial[:,:,4],a['target_ids'],a['stratum'])
            dump(out/'rows.json',rows);summary=summarize(rows)
            dump(out/'summary.json',{'seed':seed,'init':init,'arm':arm,'count':count,'input':input_kind,'metrics':summary,'model_sha256':model_sha,'data_sha256':sha(root/'trajectories.npz'),'predictions_sha256':sha(out/'predictions.npz'),'protocol_sha256':sha(c.BASE/'protocol.json')})
            print('R3 evaluate',seed,init,arm,count,input_kind,summary['61']['all'],flush=True)
    event('R3_evaluation_complete',seed=seed,init=init,arm=arm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=c.DEVELOPMENT);p.add_argument('--init',type=int,default=0);p.add_argument('--arm',choices=c.ARMS+['coarse_physics','force_wall'],required=True);a=p.parse_args();evaluate(a.seed,a.init,a.arm)
