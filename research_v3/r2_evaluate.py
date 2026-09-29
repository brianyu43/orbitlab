"""Current-frame state/edit diagnostics and calibrated predictive-set coverage.

All model outputs are saved before current-state labels are loaded. The known
analytic controller is held fixed; its relative preservation is not learned.
"""
import argparse,json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import r2_core as c
import r2_world as w
from v3_common import dump,sha,event,lock

def discrete(raw):
    return [{'slot':i,'shape':int(v[:4].argmax()),'pose':int(v[4:8].argmax()),'radius':int(v[8:12].argmax())+5,
             'x':float(v[12]),'y':float(v[13]),'rgb':(1/(1+np.exp(-v[14:17].clip(-80,80)))*255).tolist()} for i,v in enumerate(raw) if v[17]>=0]

def match(pred,truth):
    if not pred:return {}
    cost=np.array([[(p['x']-g['x'])**2+(p['y']-g['y'])**2 for p in pred] for g in truth])
    gi,pi=linear_sum_assignment(cost)
    return {int(g):int(p) for g,p in zip(gi,pi)}

def checks(p,g):
    if p is None:return [False]*6
    return [p['shape']==g['shape'],(p['pose']-g['pose'])%(2 if g['shape']==3 else 4)==0,p['radius']==g['radius'],
            abs(p['x']-g['x'])<=1 and abs(p['y']-g['y'])<=1,bool(np.max(np.abs(np.array(p['rgb'])-g['rgb']))<=20),True]

def probability_set(logits,mass=.9):
    p=np.exp(logits-logits.max());p/=p.sum();order=np.argsort(-p,kind='stable');n=int(np.searchsorted(np.cumsum(p[order]),mass))+1
    return order[:min(n,len(order))].tolist()

@torch.no_grad()
def predict(model,images,repeat_current=False):
    raw=[];masks=[]
    for i in range(0,len(images),32):
        x=torch.from_numpy(images[i:i+32]).permute(0,1,4,2,3).float()/255
        if repeat_current:x=x[:,:1].expand(-1,4,-1,-1,-1)
        out=model(x);raw.append(out['raw'].numpy());masks.append(out['masks'].numpy())
    return np.concatenate(raw),np.concatenate(masks)

def calibration(raw,records):
    xy=[];rgb=[]
    for values,row in zip(raw,records):
        pred=discrete(values);m=match(pred,row['objects'])
        for gi,pi in m.items():
            p,g=pred[pi],row['objects'][gi]
            xy.append([abs(p['x']-g['x']),abs(p['y']-g['y'])]);rgb.append(np.abs(np.array(p['rgb'])-g['rgb']).tolist())
    return {'matched_objects':len(xy),'position_half_width':np.quantile(xy,.95,axis=0,method='higher').tolist() if xy else [64.,64.],
            'rgb_half_width':np.quantile(rgb,.95,axis=0,method='higher').tolist() if rgb else [255.,255.,255.],
            'fallback_full_range':not bool(xy),'note':'Matched calibration-object residual quantiles; not formal simultaneous or OOD coverage. Missing objects remain uncovered in metrics.'}

def score(raw,masks,records,visible,cal):
    rows=[]
    for i,(values,record) in enumerate(zip(raw,records)):
        truth=record['objects'];pred=discrete(values);assignment=match(pred,truth);n=len(truth)
        object_checks=[checks(pred[assignment[j]] if j in assignment else None,g) for j,g in enumerate(truth)]
        close=[j in assignment and np.hypot(pred[assignment[j]]['x']-g['x'],pred[assignment[j]]['y']-g['y'])<=4 for j,g in enumerate(truth)]
        full=bool(len(pred)==n and np.array(object_checks).all())
        pos=[];covered=[];shape_cover=[];pose_cover=[];radius_cover=[];xy_cover=[];color_cover=[];set_sizes=[]
        for j,g in enumerate(truth):
            if j not in assignment:
                pos.append(64.);covered.append(False);shape_cover.append(False);pose_cover.append(False);radius_cover.append(False);xy_cover.append(False);color_cover.append(False);continue
            p=pred[assignment[j]];v=values[p['slot']]
            sets=[probability_set(v[start:start+4]) for start in [0,4,8]]
            sx=g['shape'] in sets[0];sp=any((pose-g['pose'])%(2 if g['shape']==3 else 4)==0 for pose in sets[1]);sr=g['radius']-5 in sets[2]
            spxy=abs(p['x']-g['x'])<=cal['position_half_width'][0] and abs(p['y']-g['y'])<=cal['position_half_width'][1]
            scrgb=bool((np.abs(np.array(p['rgb'])-g['rgb'])<=np.array(cal['rgb_half_width'])).all())
            shape_cover.append(sx);pose_cover.append(sp);radius_cover.append(sr);xy_cover.append(spxy);color_cover.append(scrgb);covered.append(sx and sp and sr and spxy and scrgb);set_sizes.append([len(a) for a in sets])
            pos.append((abs(p['x']-g['x'])+abs(p['y']-g['y']))/2)
        # Merge diagnostic: one present predicted slot covers >=25% visible alpha
        # mass of at least two true objects. Not a semantic identity oracle.
        visible_mass=visible[i].sum((1,2)).clip(1e-6)
        present_slots=[p['slot'] for p in pred]
        overlaps=(masks[i,present_slots,None]*visible[i,None]).sum((2,3))/visible_mass[None] if present_slots else np.empty((0,n))
        merged=int(((overlaps>=.25).sum(1)>=2).sum())
        tid=record['target_id'];click=record['click']
        selected=int(np.argmin([(p['x']-click[0])**2+(p['y']-click[1])**2 for p in pred])) if pred else -1
        selected_ok=tid in assignment and assignment[tid]==selected and close[tid]
        # Fixed input commands derive from scene index and contain no hidden ID.
        new_rgb=[45+((i*37+71)%201),45+((i*53+127)%201),45+((i*19+29)%201)]
        successes=[];preserved=[]
        for operation in ['color','rotate','translate']:
            output=[dict(p) for p in pred];target=[dict(g) for g in truth]
            if operation=='color':
                target[tid]['rgb']=new_rgb
                if selected>=0:output[selected]['rgb']=new_rgb
            elif operation=='rotate':
                target[tid]['pose']=(target[tid]['pose']+1)%4
                if selected>=0:output[selected]['pose']=(output[selected]['pose']+1)%4
            else:
                target[tid]['x']+=5
                if selected>=0:output[selected]['x']+=5
            state_checks=[all(checks(output[assignment[j]] if j in assignment else None,g)) for j,g in enumerate(target)]
            successes.append(bool(selected_ok and len(pred)==n and all(state_checks)))
            preserved.append(all(state_checks[j] for j in range(n) if j!=tid))
        rows.append({'source_index':i,'count':n,'same_color':record['same_color'],'continuous_color':record['continuous_color'],'textured_background':record['textured_background'],
            'requested_occlusion':record['requested_occlusion'],'actual_target_occlusion':record['actual_target_occlusion'],
            'count_correct':float(len(pred)==n),'predicted_count':len(pred),'missing_fraction':float(1-np.mean(close)),'merged_predicted_slots':merged,
            'full_state_success':float(full),'shape_accuracy':float(np.mean(np.array(object_checks)[:,0])),'pose_accuracy':float(np.mean(np.array(object_checks)[:,1])),
            'radius_accuracy':float(np.mean(np.array(object_checks)[:,2])),'position_accuracy':float(np.mean(np.array(object_checks)[:,3])),'color_accuracy':float(np.mean(np.array(object_checks)[:,4])),
            'position_mae_all':float(np.mean(pos)),'target_selection':float(selected_ok),'joint_edit_success':float(np.mean(successes)),
            'non_target_correct':float(np.mean(preserved)),'relative_analytic_preservation':1.,
            'predictive_set_coverage':float(np.mean(covered)),'shape_set_coverage':float(np.mean(shape_cover)),'pose_set_coverage':float(np.mean(pose_cover)),
            'radius_set_coverage':float(np.mean(radius_cover)),'xy_interval_coverage':float(np.mean(xy_cover)),'rgb_interval_coverage':float(np.mean(color_cover)),
            'mean_categorical_set_size':float(np.mean(set_sizes)) if set_sizes else None})
    return rows

def summarize(rows):
    grouping=['source_index','count','same_color','continuous_color','textured_background','requested_occlusion']
    def means(rr):
        return {'scenes':len(rr),**{k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None for k in rr[0] if k not in grouping}}
    groups={'all':means(rows)}
    for f in [0.,.25,.5,.75]:groups[f'occlusion_{f}']=means([r for r in rows if r['requested_occlusion']==f])
    for key in ['same_color','continuous_color','textured_background']:
        for val in [0,1]:groups[f'{key}_{val}']=means([r for r in rows if r[key]==val])
    return groups

def evaluate(seed,init,arm,repeat_current=False):
    c.freeze();torch.set_num_threads(2);name=arm+('_repeat_current' if repeat_current else '')
    cp=w.BASE/f'runs/d{seed}_s{init}/{arm}/model.pt';ck=torch.load(cp,weights_only=True);model=c.model_for(arm);model.load_state_dict(ck['state_dict']);model.eval()
    output=w.BASE/f'evaluation/d{seed}_s{init}/{name}';output.mkdir(parents=True,exist_ok=True)
    cal_path=output/'calibration.json'
    if cal_path.exists():cal=json.loads(cal_path.read_text())
    else:
        folder=w.BASE/f'data/d{seed}/calibration/n2';images=np.load(folder/'rgb.npz')['images'];raw,_=predict(model,images,repeat_current)
        np.savez_compressed(output/'calibration_inference.npz',raw=raw)
        labels=json.loads((folder/'labels.json').read_text());cal=calibration(raw,labels);dump(cal_path,cal)
    split='val' if seed==w.DEVELOPMENT else 'test'
    for count in [2,3,4,6]:
        folder=w.BASE/f'data/d{seed}/{split}/n{count}';dest=output/f'n{count}';dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():
            rec=json.loads((dest/'summary.json').read_text());assert rec['model_sha256']==sha(cp);continue
        images=np.load(folder/'rgb.npz')['images'];raw,masks=predict(model,images,repeat_current)
        np.savez_compressed(dest/'inference.npz',raw=raw,masks=masks.astype(np.float32))
        labels=json.loads((folder/'labels.json').read_text());visible=np.load(folder/'masks.npz')['visible'].astype(np.float32)/255
        rows=score(raw,masks,labels,visible,cal);dump(dest/'rows.json',rows);summary=summarize(rows)
        dump(dest/'summary.json',{'seed':seed,'init':init,'arm':arm,'repeat_current':repeat_current,'count':count,'metrics':summary,'model_sha256':sha(cp),
            'input_sha256':sha(folder/'rgb.npz'),'inference_sha256':sha(dest/'inference.npz'),'calibration_sha256':sha(cal_path),'evaluator_sha256':sha(__file__),
            'scope':'State/analytic-edit diagnostics. No claim of learned background-preserving pixel editing. All categorical metrics supervised; uncertainty sets are empirical and include misses.'})
        print('R2 evaluate',name,count,{k:summary['all'][k] for k in ['count_correct','missing_fraction','joint_edit_success','non_target_correct','predictive_set_coverage']},flush=True)
    event('R2_evaluation_complete',seed=seed,init=init,arm=name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=w.DEVELOPMENT);p.add_argument('--init',type=int,default=0);p.add_argument('--arm',choices=c.ARMS,required=True);p.add_argument('--repeat-current',action='store_true');a=p.parse_args();evaluate(a.seed,a.init,a.arm,a.repeat_current)
