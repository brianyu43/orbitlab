"""Frozen RGB -> state -> click selection -> intervention -> raster pipeline."""
import json
import numpy as np
import torch
from common import HERE, dump, fresh_dir, digest_tensor, r, o
from object_edit_confirmation import NAME, freeze
from cache_object_slots import encode
from object_perception_v2 import LinearImageSlots
from object_state_readout import StateReadout, discrete_states, geometric_assignment
from object_oracle_study import Predictor, factor_checks, image_from_slots
from object_tasks import analytic_step

EDITED={'color':[1],'novel_color_pair':[1],'rotate':[5],'translate':[2,3],'color_then_rotate':[1,5]}


def pack_slots(raw):
    """No labels/count input; capacity is the protocol's maximum of four."""
    order=raw[...,15].argsort(dim=-1,descending=True,stable=True)[:,:4].sort(-1).values
    state=discrete_states(raw).gather(1,order[:,:,None].expand(-1,-1,16))
    overflow=(raw[...,15]>=0).sum(-1)-(state[...,15]>=.5).sum(-1)
    return state,order,overflow


def select_target(state,click_xy):
    xy=state[...,10:12]*31.5+31.5
    distance=(xy-click_xy[:,None,:]).square().sum(-1)
    distance=distance.masked_fill(state[...,15]<.5,float('inf'))
    chosen=distance.argmin(-1);valid=(state[...,15]>=.5).any(-1)
    return torch.where(valid,chosen,torch.full_like(chosen,-1))


@torch.no_grad()
def apply_commands(state,selected,commands,controller):
    out=state.clone();valid=selected>=0
    mask=torch.zeros_like(state[...,15]);mask[valid,selected[valid]]=1
    for j in range(commands.shape[1]):
        cmd=commands[:,j];active=(cmd[:,:3].sum(-1)>0)&valid
        if controller=='identity':updated=out
        elif controller=='analytic':updated=analytic_step(out,mask,cmd)
        else:updated=controller(out,mask,cmd)
        out=torch.where(active[:,None,None],updated,out)
    if not torch.isfinite(out).all():raise FloatingPointError('Nonfinite predicted state')
    return out


def load_controller(seed):
    ck=torch.load(HERE/f'runs/object_oracle_study_v1/object_s{seed}/model.pt',map_location='cpu',weights_only=True)
    model=Predictor('object');model.load_state_dict(ck['state_dict']);model.eval();return model


def load_queries(split):
    # This loader has no state/mask/ID/target-image labels.
    with np.load(HERE/f'data/{NAME}/{split}/queries.npz') as a:
        return {k:torch.from_numpy(a[k].copy()) for k in ['source_indices','commands','click_xy']}


@torch.no_grad()
def cache_rgb(c):
    root=HERE/f'runs/{NAME}/rgb_cache'
    if (root/'manifest.json').exists():
        manifest=json.loads((root/'manifest.json').read_text())
        for f in manifest['files']:assert r.sha(root/f['path'])==f['sha256']
        return
    fresh_dir(root);files=[]
    for kind in c['kinds']:
        ck=torch.load(HERE/f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt',map_location='cpu',weights_only=True)
        encoder=LinearImageSlots(kind,ck['dim'],ck['slots']);encoder.load_state_dict(ck['state_dict']);encoder.eval()
        for si,split in enumerate(c['splits']):
            with np.load(HERE/f'data/{NAME}/{split}/rgb.npz') as a:x=torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255
            eps=torch.randn(len(x),5,32,generator=torch.Generator().manual_seed(c['epsilon_seed']+si))
            slots=torch.cat([encode(encoder,x[i:i+c['batch']],eps[i:i+c['batch']]) for i in range(0,len(x),c['batch'])])
            heads={}
            for seed in c['head_seeds']:
                ck=torch.load(HERE/f'runs/object_readout_v1/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
                head=StateReadout(ck['mean'],ck['std']);head.load_state_dict(ck['state_dict']);head.eval()
                raw=head(slots);state,order,overflow=pack_slots(raw)
                heads[str(seed)]={'raw':raw,'state':state,'retained_indices':order,'overflow':overflow}
            path=root/f'{kind}_{split}.pt';torch.save({'epsilon':eps,'slots':slots,'heads':heads},path)
            files.append({'path':path.name,'sha256':r.sha(path),'rgb_sha256':digest_tensor(x),'images':len(x)})
            print('image edit RGB cache',kind,split,len(x),flush=True)
    dump(root/'manifest.json',{'files':files,'config_sha256':r.sha(HERE/f'configs/{NAME}.json')})


def summarize_rows(rows):
    excluded={'task_index','source_index','operation','target_id','selected_slot','source_target_pair_is_unseen','target_pair_is_unseen'}
    keys=[k for k in rows[0] if k not in excluded];groups={'all':rows}
    for op in sorted({v['operation'] for v in rows}):groups[op]=[v for v in rows if v['operation']==op]
    for name,key in [('unseen_source_target','source_target_pair_is_unseen'),('unseen_output_target','target_pair_is_unseen')]:
        subset=[v for v in rows if v[key]]
        if subset:groups[name]=subset
    result={}
    for group,rr in groups.items():
        ids=sorted({v['source_index'] for v in rr})
        metrics={}
        for key in keys:
            values=[np.mean([v[key] for v in rr if v['source_index']==i and v[key] is not None]) for i in ids if any(v['source_index']==i and v[key] is not None for v in rr)]
            metrics[key]=r.bootstrap(values) if values else None
        result[group]={'tasks':len(rr),'source_scenes':len(ids),'metrics':metrics}
    return result


def image_errors(pred,target,source):
    p=pred.astype(np.float32)/255;t=target.astype(np.float32)/255;s=source.astype(np.float32)/255
    fg=(t.max(-1)>.05)|(s.max(-1)>.05);changed=np.any(target!=source,axis=-1)
    pm=pred.max(-1)>13;tm=target.max(-1)>13
    return {'image_mae':float(np.abs(p-t).mean()),'image_foreground_mae':float(np.abs(p-t)[fg].mean()),
        'image_mask_iou':float((pm&tm).sum()/max((pm|tm).sum(),1)),
        'changed_pixels_mae':float(np.abs(p-t)[changed].mean()) if changed.any() else None,
        'input_passthrough_foreground_mae':float(np.abs(s-t)[fg].mean())}


def score(source,pred,selected,commands,truth_source,truth_target,records,predicted_images,target_images,input_images,overflow):
    """Match only the source, keep that correspondence for every later command."""
    expected=apply_commands(source,selected,commands,'analytic');relative=factor_checks(pred,expected).numpy()
    rows=[];matches=[]
    for i,record in enumerate(records):
        match,_,gt=geometric_assignment(source[i].numpy(),truth_source[i].numpy());matches.append(match)
        tid=record['target_object_id'];sel=int(selected[i]);idx=match.get(tid,-1);edited=EDITED[record['operation']]
        untouched=[j for j in range(7) if j not in edited];chosen_correct=idx>=0 and sel==idx
        mapped=torch.stack([pred[i,match[g]] if g in match else torch.zeros(16) for g in gt])
        checks=factor_checks(mapped[None],truth_target[i,gt][None])[0].numpy()
        # Missing assignments are failures even if an empty slot's argmax agrees.
        for j,g in enumerate(gt):
            if g not in match:checks[j]=False
        target_checks=checks[list(gt).index(tid)];other=[j for j,g in enumerate(gt) if g!=tid]
        final_count=int((pred[i,:,15]>=.5).sum());initial_count=int((source[i,:,15]>=.5).sum())
        target_success=bool(idx>=0 and pred[i,idx,15]>=.5 and target_checks[edited].all())
        target_other=bool(idx>=0 and target_checks[untouched].all());non_target=bool(checks[other].all())
        full=bool(final_count==len(gt) and len(match)==len(gt) and checks.all())
        initial_present=(source[i,:,15]>=.5).numpy();other_slots=[j for j in range(4) if j!=sel and initial_present[j]]
        empty_slots=~initial_present;empty_ok=bool((pred[i,empty_slots,15]<.5).all())
        relative_target=bool(sel>=0 and pred[i,sel,15]>=.5 and relative[i,sel,edited].all())
        relative_other=bool(sel>=0 and relative[i,sel,untouched].all())
        relative_nontarget=bool(relative[i,other_slots].all())
        position=float(np.abs((pred[i,idx,10:12]-truth_target[i,tid,10:12]).numpy()).mean()*31.5) if idx>=0 else None
        row={'task_index':i,'source_index':record['source_index'],'operation':record['operation'],'target_id':tid,'selected_slot':sel,
            'source_target_pair_is_unseen':record['source_target_pair_is_unseen'],'target_pair_is_unseen':record['target_pair_is_unseen'],
            'target_selection_correct':chosen_correct,'target_success_gt':target_success,'target_other_correct_gt':target_other,
            'non_target_correct_gt':non_target,'count_correct_gt':final_count==len(gt),'full_scene_correct_gt':full,
            'end_to_end_success':chosen_correct and full,'source_objects_matched_fraction':len(match)/len(gt),
            'source_count_correct':initial_count==len(gt),'discarded_positive_slots':int(overflow[i]),
            'command_followed_relative_to_prediction':relative_target,'target_other_preserved_relative_to_prediction':relative_other,
            'non_target_preserved_relative_to_prediction':relative_nontarget,'empty_slots_preserved':empty_ok,
            'full_command_success_relative_to_prediction':relative_target and relative_other and relative_nontarget and empty_ok,
            'target_position_mae_pixels_if_matched':position,
            **image_errors(predicted_images[i],target_images[i],input_images[i])}
        rows.append(row)
    return rows,matches


@torch.no_grad()
def evaluate_condition(c,split,name,source,selected,overflow,controller,queries,truth,targets,records,images):
    folder=HERE/f'runs/{NAME}/{name}/{split}'
    if (folder/'metrics.json').exists():return json.loads((folder/'metrics.json').read_text())
    fresh_dir(folder);pred=apply_commands(source,selected,queries['commands'],controller)
    output_images=np.stack([image_from_slots(p) for p in pred])
    rows,matches=score(source,pred,selected,queries['commands'],truth,targets['states'],records,
        output_images,targets['images'],images,overflow)
    r.write_csv(folder/'rows.csv',rows);dump(folder/'initial_matches.json',matches)
    torch.save({'source_states':source,'selected_slots':selected,'predicted_states':pred,'overflow':overflow},folder/'predictions.pt')
    np.savez_compressed(folder/'images.npz',images=output_images)
    examples=[]
    for i in range(min(12,len(records))):examples.extend([images[i],targets['images'][i],output_images[i]])
    o.grid(torch.from_numpy(np.stack(examples)).permute(0,3,1,2).float()/255,folder/'examples.png',6)
    metrics={'name':name,'split':split,'groups':summarize_rows(rows),'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'files':{f:r.sha(folder/f) for f in ['rows.csv','initial_matches.json','predictions.pt','images.npz','examples.png']}}
    dump(folder/'metrics.json',metrics)
    a=metrics['groups']['all']['metrics'];print('image edit result',name,split,'select',round(a['target_selection_correct']['mean'],4),
        'full',round(a['end_to_end_success']['mean'],4),'command',round(a['full_command_success_relative_to_prediction']['mean'],4),flush=True)
    return metrics


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(2);cache_rgb(c);results=[]
    for split in c['splits']:
        data=HERE/f'data/{NAME}/{split}';queries=load_queries(split);indices=queries['source_indices']
        # Labels are materialized after RGB-only cached inference, for evaluation
        # and explicitly named privileged controls only.
        with np.load(data/'scene_labels.npz') as a:truth=torch.from_numpy(a['states'].copy())[indices]
        with np.load(data/'targets.npz') as a:targets={'states':torch.from_numpy(a['states'].copy()),'images':a['images'].copy()}
        with np.load(data/'rgb.npz') as a:images=a['images'][indices.numpy()].copy()
        records=json.loads((data/'tasks.json').read_text());zero=torch.zeros(len(records),dtype=torch.long)
        tid=torch.tensor([v['target_object_id'] for v in records])
        for name,sel,controller in [('oracle_id_analytic',tid,'analytic'),('oracle_click_analytic',select_target(truth,queries['click_xy']),'analytic')]+[
            (f'oracle_id_object_s{seed}',tid,load_controller(seed)) for seed in c['head_seeds']]:
            results.append(evaluate_condition(c,split,name,truth,sel,zero,controller,queries,truth,targets,records,images))
        for kind in c['kinds']:
            cache=torch.load(HERE/f'runs/{NAME}/rgb_cache/{kind}_{split}.pt',map_location='cpu',weights_only=True)
            for seed in c['head_seeds']:
                source=cache['heads'][str(seed)]['state'][indices];overflow=cache['heads'][str(seed)]['overflow'][indices]
                selected=select_target(source,queries['click_xy'])
                for label,controller in [('identity','identity'),('analytic','analytic'),('object',load_controller(seed))]:
                    name=f'{kind}_s{seed}_{label}'
                    results.append(evaluate_condition(c,split,name,source,selected,overflow,controller,queries,truth,targets,records,images))
        dump(HERE/f'reports/{NAME}/progress.json',{'completed_conditions':len(results),'expected_conditions':len(c['splits'])*23})
    dump(HERE/f'reports/{NAME}/summary.json',{'results':results,'config_sha256':r.sha(HERE/f'configs/{NAME}.json'),
        'data_manifest_sha256':r.sha(HERE/f'data/{NAME}/manifest.json'),
        'scope':'Fresh single-seed synthetic RGB/click edit confirmation of fixed supervised state readouts and controllers. Not natural language understanding or amodal depth recovery.'})


if __name__=='__main__':main()
