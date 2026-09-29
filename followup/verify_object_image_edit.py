"""Replay RGB inference and independently score the frozen image edit study."""
import csv
import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from common import HERE,dump,digest_tensor,r
from object_edit_confirmation import NAME,freeze
from object_perception_v2 import LinearImageSlots
from object_state_readout import StateReadout
from object_oracle_study import Predictor
from object_world import ObjectState,render


def check(v,t):
    pose=lambda a:int(np.rint(-np.arctan2(a[14],a[13])/(np.pi/2)))%4
    return np.array([v[:4].argmax()==t[:4].argmax(),v[4:10].argmax()==t[4:10].argmax(),
        abs(v[10]-t[10])*31.5<=1,abs(v[11]-t[11])*31.5<=1,abs(v[12]-t[12])*8<=.5,
        pose(v)==pose(t),(v[15]>=.5)==(t[15]>=.5)])


def analytic_reference(states,selected,commands):
    out=states.copy()
    for i,sel in enumerate(selected):
        if sel<0:continue
        for c in commands[i]:
            if c[0]:out[i,sel,4:10]=c[3:9]
            if c[1]:
                co,si=c[9:11];x,y=out[i,sel,13:15];out[i,sel,13:15]=[x*co-y*si,x*si+y*co]
            if c[2]:out[i,sel,10:12]+=c[11:13]
    return out


def decode_raw(raw):
    raw=raw.numpy();states=[];orders=[];overflows=[]
    directions=np.array([[1,0],[0,-1],[-1,0],[0,1]],dtype=np.float32)
    for x in raw:
        order=sorted(sorted(range(5),key=lambda j:(-float(x[j,15]),j))[:4]);out=np.zeros((4,16),np.float32)
        for j,k in enumerate(order):
            v=x[k]
            if v[15]<0:continue
            out[j,v[:4].argmax()]=1;out[j,4+v[4:10].argmax()]=1;out[j,10:13]=v[10:13]
            angle=int(np.rint(-np.arctan2(v[14],v[13])/(np.pi/2)))%4;out[j,13:15]=directions[angle];out[j,15]=1
        states.append(out);orders.append(order);overflows.append(int((x[:,15]>=0).sum()-(out[:,15]>=.5).sum()))
    return np.stack(states),np.array(orders),np.array(overflows)


def select_reference(states,clicks):
    selected=[]
    for ss,click in zip(states,clicks):
        present=np.where(ss[:,15]>=.5)[0]
        if len(present):
            xy=ss[present,10:12]*31.5+31.5;distance=((xy-click)**2).sum(1);selected.append(int(present[distance.argmin()]))
        else:selected.append(-1)
    return np.array(selected)


def rendered_reference(state):
    objects=[]
    for v in state:
        if v[15]<.5:continue
        pose=int(np.rint(-np.arctan2(v[14],v[13])/(np.pi/2)))%4
        objects.append(ObjectState(len(objects),int(v[:4].argmax()),int(v[4:10].argmax()),
            int(np.clip(np.rint(v[10]*31.5+31.5),-32,95)),int(np.clip(np.rint(v[11]*31.5+31.5),-32,95)),
            int(np.clip(np.rint(v[12]*8),1,12)),pose))
    return render(tuple(objects))['image'] if objects else np.zeros((64,64,3),np.uint8)


def independent_rows(source,pred,selected,commands,truth,targets,records,output_images,source_images,overflow):
    expected=analytic_reference(source,selected,commands);result=[];matches=[]
    for i,rec in enumerate(records):
        s=source[i];p=pred[i];t=truth[i];y=targets['states'][i];tid=rec['target_object_id'];sel=int(selected[i])
        present=np.where(s[:,15]>=.5)[0];gt=np.where(t[:,15]>=.5)[0];mapping={}
        if len(present):
            cost=((t[gt,None,10:12]-s[None,present,10:12])**2).sum(-1);a,b=linear_sum_assignment(cost)
            mapping={int(gt[j]):int(present[k]) for j,k in zip(a,b)}
        matches.append(mapping);checks={g:check(p[mapping[g]],y[g]) if g in mapping else np.zeros(7,bool) for g in gt}
        dims={'color':[1],'novel_color_pair':[1],'rotate':[5],'translate':[2,3],'color_then_rotate':[1,5]}[rec['operation']]
        untouched=[j for j in range(7) if j not in dims];idx=mapping.get(tid,-1)
        target_ok=idx>=0 and p[idx,15]>=.5 and checks[tid][dims].all();full_count=int((p[:,15]>=.5).sum())==len(gt)
        full=full_count and len(mapping)==len(gt) and all(v.all() for v in checks.values());correct=idx>=0 and sel==idx
        rel=[check(p[k],expected[i,k]) for k in range(4)];empty=np.where(s[:,15]<.5)[0]
        rt=sel>=0 and p[sel,15]>=.5 and rel[sel][dims].all();ro=sel>=0 and rel[sel][untouched].all()
        rn=all(rel[k].all() for k in present if k!=sel);re=all(p[k,15]<.5 for k in empty)
        pi=output_images[i].astype(np.float32)/255;ti=targets['images'][i].astype(np.float32)/255;si=source_images[i].astype(np.float32)/255
        fg=(ti.max(-1)>.05)|(si.max(-1)>.05);changed=np.any(targets['images'][i]!=source_images[i],axis=-1)
        pm=output_images[i].max(-1)>13;tm=targets['images'][i].max(-1)>13
        result.append({'task_index':i,'source_index':rec['source_index'],'operation':rec['operation'],'target_id':tid,'selected_slot':sel,
            'source_target_pair_is_unseen':rec['source_target_pair_is_unseen'],'target_pair_is_unseen':rec['target_pair_is_unseen'],
            'target_selection_correct':bool(correct),'target_success_gt':bool(target_ok),'target_other_correct_gt':bool(idx>=0 and checks[tid][untouched].all()),
            'non_target_correct_gt':bool(all(v.all() for k,v in checks.items() if k!=tid)),'count_correct_gt':full_count,
            'full_scene_correct_gt':bool(full),'end_to_end_success':bool(correct and full),'source_objects_matched_fraction':len(mapping)/len(gt),
            'source_count_correct':len(present)==len(gt),'discarded_positive_slots':int(overflow[i]),
            'command_followed_relative_to_prediction':bool(rt),'target_other_preserved_relative_to_prediction':bool(ro),
            'non_target_preserved_relative_to_prediction':bool(rn),'empty_slots_preserved':bool(re),
            'full_command_success_relative_to_prediction':bool(rt and ro and rn and re),
            'target_position_mae_pixels_if_matched':float(np.abs(p[idx,10:12]-y[tid,10:12]).mean()*31.5) if idx>=0 else None,
            'image_mae':float(np.abs(pi-ti).mean()),'image_foreground_mae':float(np.abs(pi-ti)[fg].mean()),
            'image_mask_iou':float(np.logical_and(pm,tm).sum()/max(np.logical_or(pm,tm).sum(),1)),
            'changed_pixels_mae':float(np.abs(pi-ti)[changed].mean()) if changed.any() else None,
            'input_passthrough_foreground_mae':float(np.abs(si-ti)[fg].mean())})
    return result,matches


def audit_aggregates(rows,groups):
    count=0
    for group,info in groups.items():
        if group=='all':rr=rows
        elif group=='unseen_source_target':rr=[v for v in rows if v['source_target_pair_is_unseen']]
        elif group=='unseen_output_target':rr=[v for v in rows if v['target_pair_is_unseen']]
        else:rr=[v for v in rows if v['operation']==group]
        assert info['tasks']==len(rr);ids=sorted({v['source_index'] for v in rr});assert info['source_scenes']==len(ids)
        for key,stat in info['metrics'].items():
            values=[]
            for i in ids:
                vv=[v[key] for v in rr if v['source_index']==i and v[key] is not None]
                if vv:values.append(np.mean(vv))
            if not values:assert stat is None;continue
            a=np.asarray(values);rng=np.random.default_rng(9123);boot=a[rng.integers(len(a),size=(1000,len(a)))].mean(1)
            assert abs(float(a.mean())-stat['mean'])<1e-8,(group,key)
            np.testing.assert_allclose(np.quantile(boot,[.025,.975]),stat['base_scene_ci95'],atol=1e-8,rtol=0);count+=1
    return count


@torch.no_grad()
def main():
    c=freeze();torch.set_num_threads(2);root=HERE/f'reports/{NAME}';study=json.loads((root/'summary.json').read_text())
    assert json.loads((root/'data_verification.json').read_text())['all_passed'];assert study['data_manifest_sha256']==r.sha(HERE/f'data/{NAME}/manifest.json')
    assert study['config_sha256']==r.sha(HERE/f'configs/{NAME}.json');assert len(study['results'])==115
    sources={};encoded=0;head_count=0
    cache_manifest=json.loads((HERE/f'runs/{NAME}/rgb_cache/manifest.json').read_text())
    for kind in c['kinds']:
        ck=torch.load(HERE/f'runs/object_perception_curve_v1/{kind}_s0_step12000/model.pt',map_location='cpu',weights_only=True)
        model=LinearImageSlots(kind,ck['dim'],ck['slots']);model.load_state_dict(ck['state_dict']);model.eval()
        for si,split in enumerate(c['splits']):
            path=HERE/f'runs/{NAME}/rgb_cache/{kind}_{split}.pt';cm=next(v for v in cache_manifest['files'] if v['path']==path.name)
            assert cm['sha256']==r.sha(path);saved=torch.load(path,map_location='cpu',weights_only=True)
            with np.load(HERE/f'data/{NAME}/{split}/rgb.npz') as a:rgb=torch.from_numpy(a['images'].copy()).permute(0,3,1,2).float()/255
            assert cm['rgb_sha256']==digest_tensor(rgb)
            eps=torch.randn(len(rgb),5,32,generator=torch.Generator().manual_seed(c['epsilon_seed']+si));assert torch.equal(eps,saved['epsilon'])
            # Use the original full forward, not the lightweight cache helper.
            slots=torch.cat([model(rgb[i:i+c['batch']],eps[i:i+c['batch']])['slots'] for i in range(0,len(rgb),c['batch'])])
            torch.testing.assert_close(slots,saved['slots'],atol=0,rtol=0);encoded+=len(rgb)
            for seed in c['head_seeds']:
                ck=torch.load(HERE/f'runs/object_readout_v1/{kind}_s{seed}/model.pt',map_location='cpu',weights_only=True)
                head=StateReadout(ck['mean'],ck['std']);head.load_state_dict(ck['state_dict']);head.eval();raw=head(slots)
                torch.testing.assert_close(raw,saved['heads'][str(seed)]['raw'],atol=0,rtol=0)
                state,order,overflow=decode_raw(raw);stored=saved['heads'][str(seed)]
                for actual,key in [(state,'state'),(order,'retained_indices'),(overflow,'overflow')]:np.testing.assert_array_equal(actual,stored[key].numpy())
                sources[(kind,seed,split)]=(state,overflow);head_count+=len(raw)
            print('verified image encoder/readout',kind,split,flush=True)
    rows_count=images_count=aggregate_count=0
    for split in c['splits']:
        data=HERE/f'data/{NAME}/{split}'
        with np.load(data/'queries.npz') as a:queries={k:a[k].copy() for k in a.files}
        ix=queries['source_indices'];cmds=queries['commands'];clicks=queries['click_xy']
        with np.load(data/'scene_labels.npz') as a:truth=a['states'][ix].copy()
        with np.load(data/'targets.npz') as a:targets={k:a[k].copy() for k in a.files}
        with np.load(data/'rgb.npz') as a:rgb=a['images'][ix].copy()
        records=json.loads((data/'tasks.json').read_text())
        for entry in [e for e in study['results'] if e['split']==split]:
            name=entry['name'];folder=HERE/f'runs/{NAME}/{name}/{split}';metrics=json.loads((folder/'metrics.json').read_text());assert metrics==entry
            for file,sha in metrics['files'].items():assert sha==r.sha(folder/file)
            ck=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=True)
            if name.startswith('oracle_'):
                source=truth.copy();overflow=np.zeros(len(ix),np.int64)
                selected=select_reference(source,clicks) if name=='oracle_click_analytic' else np.array([v['target_object_id'] for v in records])
                seed=int(name[-1]) if '_object_s' in name else None;control='object' if seed is not None else 'analytic'
            else:
                kind,ss,control=name.split('_');seed=int(ss[1:]);base,over=sources[(kind,seed,split)];source=base[ix];overflow=over[ix]
                selected=select_reference(source,clicks)
            np.testing.assert_array_equal(source,ck['source_states'].numpy());np.testing.assert_array_equal(selected,ck['selected_slots'].numpy())
            np.testing.assert_array_equal(overflow,ck['overflow'].numpy())
            if control=='identity':pred=source.copy()
            elif control=='analytic':pred=analytic_reference(source,selected,cmds)
            else:
                mc=torch.load(HERE/f'runs/object_oracle_study_v1/object_s{seed}/model.pt',map_location='cpu',weights_only=True)
                controller=Predictor('object');controller.load_state_dict(mc['state_dict']);controller.eval();out=torch.from_numpy(source.copy())
                valid=torch.from_numpy(selected>=0);mask=torch.zeros(len(source),4);mask[valid,torch.from_numpy(selected)[valid]]=1
                for j in range(2):
                    cmd=torch.from_numpy(cmds[:,j]);active=(cmd[:,:3].sum(-1)>0)&valid;new=controller(out,mask,cmd)
                    out[active]=new[active]
                pred=out.numpy()
            np.testing.assert_array_equal(pred,ck['predicted_states'].numpy())
            with np.load(folder/'images.npz') as a:output_images=a['images'].copy()
            for i,p in enumerate(pred):np.testing.assert_array_equal(rendered_reference(p),output_images[i]);images_count+=1
            rows,matches=independent_rows(source,pred,selected,cmds,truth,targets,records,output_images,rgb,overflow)
            saved_matches=json.loads((folder/'initial_matches.json').read_text());assert saved_matches==[{str(k):v for k,v in m.items()} for m in matches]
            actual=list(csv.DictReader((folder/'rows.csv').open()));assert len(rows)==len(actual)
            for a,b in zip(rows,actual):
                assert set(a)==set(b)
                for key,v in a.items():
                    if v is None:assert b[key]==''
                    elif isinstance(v,(bool,np.bool_)):assert b[key]==str(bool(v)),(name,split,key)
                    elif isinstance(v,str):assert b[key]==v
                    else:assert abs(float(b[key])-float(v))<1e-8,(name,split,key,b[key],v)
            aggregate_count+=audit_aggregates(rows,metrics['groups']);rows_count+=len(rows)
            print('verified image edit',name,split,len(rows),flush=True)
    dump(root/'verification.json',{'all_passed':True,'rgb_encodings_replayed':encoded,'head_predictions_replayed':head_count,
        'conditions_verified':len(study['results']),'edit_prediction_rows_independently_scored':rows_count,
        'rendered_outputs_replayed':images_count,'scene_aggregate_and_interval_checks':aggregate_count,
        'initial_matching_preserved':True,'source_relative_and_ground_truth_scores_separated':True,
        'checkpoints_and_frozen_sources_unchanged':True,'verifier_sha256':r.sha(__file__),'scope':study['scope']})


if __name__=='__main__':main()
