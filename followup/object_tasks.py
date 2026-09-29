"""Locked intervention tensors shared by oracle and future image-input experiments."""
import argparse
import dataclasses
import json
import time
from common import HERE, dump, fresh_dir, np, torch, r
from object_world import ObjectState, edit, render, state_slots


def command(operation,value):
    a=np.zeros(13,np.float32)
    if operation=='color':a[0]=1;a[3+int(value)]=1
    elif operation=='rotate':
        a[1]=1;angle=-int(value)*np.pi/2;a[9:11]=[np.cos(angle),np.sin(angle)]
    elif operation=='translate':a[2]=1;a[11:13]=np.asarray(value)/31.5
    else:raise ValueError(operation)
    return a


def operations(record):
    op,v=record['operation'],record['value']
    if op=='novel_color_pair':return [('color',v)]
    if op=='color_then_rotate':return [('color',v['color']),('rotate',v['rotation'])]
    return [(op,v)]


def build():
    source=HERE/'data/object_world_v1';root=fresh_dir(HERE/'data/object_edit_tasks_v1')
    config={'source_manifest_sha256':r.sha(source/'manifest.json'),'command_dim':13,'max_commands':2,
        'train_operations':['color','rotate','translate'],'held_out_operation':'Sequential color then rotate; no composed training pairs.',
        'count_occlusion_commands':'For each source, target source_index modulo object_count; seen recolor, +90 local pose, world +5 x, recolor then +90.',
        'visibility':'Object factors are preserved for non-targets. Extended count/occlusion edits may change their visibility legitimately.',
        'frozen_unix':time.time()}
    cp=HERE/'configs/object_edit_tasks_v1.json';assert not cp.exists();dump(cp,config)
    all_orbits=set();summary={};files=[]
    for split in ['train','val','test','ood','count3','count4','occlusion']:
        folder=source/split;meta=json.loads((folder/'scenes.json').read_text());records=[]
        if (folder/'edits.json').exists():records=json.loads((folder/'edits.json').read_text())
        else:
            for i,m in enumerate(meta):
                ss=tuple(ObjectState(**v) for v in m['objects']);tid=i%len(ss);target=ss[tid]
                choices=[c for c in range(6) if c!=target.color and c!=target.shape and c not in [v.color for v in ss if v.id!=tid]]
                color=choices[i%len(choices)]
                for op,value in [('color',color),('rotate',1),('translate',[5,0]),('color_then_rotate',{'color':color,'rotation':1})]:
                    e={'source_index':i,'target_object_id':tid,'operation':op,'value':value,'creates_unseen_pair':False}
                    changed=ss
                    for action,v in operations(e):changed=edit(changed,tid,action,v)
                    e['target_objects']=[dataclasses.asdict(v) for v in changed];records.append(e)
        xs=[];ys=[];cs=[];ts=[];images=[];local={v['orbit_sha256'] for v in meta}
        for idx,e in enumerate(records):
            ss=tuple(ObjectState(**v) for v in meta[e['source_index']]['objects']);tid=e['target_object_id']
            target=tuple(ObjectState(**v) for v in e['target_objects']);changed=ss
            cmds=np.zeros((2,13),np.float32)
            for j,(op,v) in enumerate(operations(e)):
                cmds[j]=command(op,v);changed=edit(changed,tid,op,v)
            assert target==changed
            a=render(target)['image'];h=r.orbit_hash(a);local.add(h)
            e={**e,'task_index':idx,'target_orbit_sha256':h,'source_target_pair_is_unseen':ss[tid].shape==ss[tid].color,
                'target_pair_is_unseen':target[tid].shape==target[tid].color}
            records[idx]=e;xs.append(state_slots(ss));ys.append(state_slots(target));cs.append(cmds);ts.append(tid);images.append(a)
        assert not local.intersection(all_orbits);all_orbits.update(local)
        dest=root/split;dest.mkdir()
        np.savez_compressed(dest/'tasks.npz',source_states=np.stack(xs),target_states=np.stack(ys),commands=np.stack(cs),target_ids=np.array(ts),target_images=np.stack(images))
        dump(dest/'records.json',records)
        summary[split]={'tasks':len(records),'source_scenes':len(meta),'unique_source_and_target_orbits':len(local)}
        for p in sorted(dest.iterdir()):files.append({'path':str(p.relative_to(root)),'sha256':r.sha(p)})
        print(split,summary[split],flush=True)
    dump(root/'manifest.json',{'splits':summary,'files':files,'cross_split_orbit_overlap':0,'code_sha256':r.sha(__file__),'config_sha256':r.sha(cp)})


def load(split):
    root=HERE/f'data/object_edit_tasks_v1/{split}'
    with np.load(root/'tasks.npz') as a:
        data={k:torch.from_numpy(a[k].copy()) for k in ['source_states','target_states','commands','target_ids']}
    return data,json.loads((root/'records.json').read_text())


def analytic_step(x,mask,c):
    out=x.clone();m=mask[...,None]
    color=c[:,0,None,None]*m
    out[:,:,4:10]=out[:,:,4:10]*(1-color)+c[:,None,3:9]*color
    pose=x[:,:,13:15];co=c[:,None,9];si=c[:,None,10]
    rotated=torch.stack([pose[:,:,0]*co-pose[:,:,1]*si,pose[:,:,0]*si+pose[:,:,1]*co],-1)
    rot=c[:,1,None,None]*m;out[:,:,13:15]=pose*(1-rot)+rotated*rot
    out[:,:,10:12]+=c[:,None,11:13]*(c[:,2,None,None]*m)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');args=parser.parse_args()
    if args.build:build()
