"""Frozen trajectory splits and paired action counterfactuals for stages C01/C02."""
import dataclasses
import json
import time
from PIL import Image, ImageDraw
from common import HERE, dump, fresh_dir, update_status, r
import dynamics_world as d
import numpy as np


def freeze():
    cp=HERE/'configs/dynamics_world_v1.json'
    if not cp.exists():
        dump(cp,{'seed':640031,'modes':['isotropic','fixed_gravity','variable_force'],
            'sizes':{'train':256,'val':64,'test':128,'attribute_ood':128,'count3':64,'count4':64,'force_ood':128},
            'force_ood_modes':['variable_force'],'observation_frames':4,'train_future_frames':16,'eval_future_frames':61,
            'max_objects':4,'physics':dataclasses.asdict(d.Physics()),'code_sha256':r.sha(HERE/'dynamics_world.py'),
            'builder_sha256':r.sha(__file__),'actions':'One velocity impulse at t=3. Counterfactual negates that impulse, keeps the initial state and context.',
            'counterfactual_splits':['test','force_ood'],
            'splitting':'Each seeded initial scene and its complete trajectory, rotations, windows and action counterfactuals form one split group.',
            'training_limit':'Train archives contain 4 past frames plus 16 future frames only. Evaluation archives contain 4 past plus 61 future frames.',
            'representation':'Continuous equal-mass disks; different visual primitives from B-stage polygons. Color has no physical effect.',
            'condition':'Gravity and wind are constant accelerations; only their sum and drag are identifiable from state trajectories.',
            'claim_boundary':'A custom discrete simulator dataset, not real physics validation or a public benchmark.',
            'frozen_unix':time.time()})
    return json.loads(cp.read_text())


def pad(array,n):
    out=np.zeros((*array.shape[:-2],4,array.shape[-1]),dtype=array.dtype);out[...,:n,:]=array;return out


def save_examples(folder,states,context,actions,cf=None):
    frames=[]
    for t,s in enumerate(states):
        images=[Image.fromarray(d.render(s)['image']).resize((192,192),Image.Resampling.NEAREST)]
        if cf is not None:images.append(Image.fromarray(d.render(cf[t])['image']).resize((192,192),Image.Resampling.NEAREST))
        canvas=Image.new('RGB',(len(images)*192,216),'white');draw=ImageDraw.Draw(canvas)
        for i,im in enumerate(images):canvas.paste(im,(i*192,24))
        draw.text((5,5),f't={t:02d}  original'+(' | reversed impulse' if cf is not None else ''),fill='black');frames.append(canvas)
    frames[0].save(folder/'example.gif',save_all=True,append_images=frames[1:],duration=90,loop=0)
    ts=[0,3,min(19,len(states)-1),len(states)-1]
    sheet=Image.new('RGB',(4*192,216),'white');draw=ImageDraw.Draw(sheet)
    for column,t in enumerate(ts):
        sheet.paste(Image.fromarray(d.render(states[t])['image']).resize((192,192),Image.Resampling.NEAREST),(column*192,24))
        draw.text((column*192+5,5),f't={t}',fill='black')
    sheet.save(folder/'example_frames.png')


def main():
    c=freeze();root=HERE/'data/dynamics_world_v1';fresh_dir(root)
    for task in ['C01','C02']:update_status(task,'in_progress',['configs/dynamics_world_v1.json'],note='Building discrete trajectories; independent data replay and context checks pending.')
    manifests=[];orbit_split={};image_split={};started=time.perf_counter()
    for mode in c['modes']:
        for split,number in c['sizes'].items():
            if split=='force_ood' and mode not in c['force_ood_modes']:continue
            folder=fresh_dir(root/mode/split);n={'count3':3,'count4':4}.get(split,2)
            length=c['observation_frames']+(c['train_future_frames'] if split=='train' else c['eval_future_frames'])
            states=[];contexts=[];actions=[];events=[];rgb=[];segments=[];records=[];cf_states=[];cf_events=[]
            for i in range(number):
                s,context,act,target=d.sample(i,c['seed'],split,mode,n);act=act[:length-1]
                orbit=d.initial_orbit_hash(s,context,act[3]);group=f'{mode}/{split}/{i}'
                if orbit in orbit_split:raise AssertionError(('Duplicate physical orbit',group,orbit_split[orbit]))
                orbit_split[orbit]=group
                initial_render=d.render(s)['image'];image_orbit=r.orbit_hash(initial_render)
                if image_orbit in image_split:raise AssertionError(('Duplicate initial image orbit',group,image_split[image_orbit]))
                image_split[image_orbit]=group
                trajectory,event=d.rollout(s,context,act,physics=d.Physics(**c['physics']))
                assert max(d.violation(x) for x in trajectory)<=c['physics']['overlap_tolerance']*1.1
                observations=[d.render(x) for x in trajectory[:c['observation_frames']]]
                states.append(pad(trajectory,n));contexts.append(context);actions.append(pad(act,n));events.append(event)
                rgb.append(np.stack([v['image'] for v in observations]));segments.append(np.stack([v['segmentation'] for v in observations]))
                record={'id':i,'group':group,'n_objects':n,'target_id':target,'initial_orbit_sha256':orbit,
                    'initial_image_orbit_sha256':image_orbit,'seed_components':[c['seed'],split,mode,i]}
                if split in c['counterfactual_splits']:
                    cf_act=-act;cf,ce=d.rollout(s,context,cf_act,physics=d.Physics(**c['physics']))
                    cf_states.append(pad(cf,n));cf_events.append(ce)
                    record['counterfactual_same_group']=True
                    np.testing.assert_array_equal(cf[:4],trajectory[:4])
                records.append(record)
                if i==0:save_examples(folder,trajectory,context,act,cf if split in c['counterfactual_splits'] else None)
            data={'states':np.stack(states),'contexts':np.stack(contexts),'actions':np.stack(actions),
                'events':np.stack(events),'observations':np.stack(rgb),'observation_segmentation':np.stack(segments),
                'n_objects':np.full(number,n,np.int8),'target_ids':np.array([v['target_id'] for v in records],np.int8)}
            if cf_states:data.update(counterfactual_states=np.stack(cf_states),counterfactual_events=np.stack(cf_events))
            np.savez_compressed(folder/'trajectories.npz',**data);dump(folder/'records.json',records)
            item={'mode':mode,'split':split,'episodes':number,'frames':length,'n_objects':n,
                'counterfactual_episodes':len(cf_states),'pair_impulses':int(data['events'][:,:,0].sum()),
                'wall_impulses':int(data['events'][:,:,1].sum()),'episodes_with_pair_collision':int((data['events'][:,:,0].sum(1)>0).sum()),
                'path':str((folder/'trajectories.npz').relative_to(HERE)),
                'archive_sha256':r.sha(folder/'trajectories.npz'),'records_sha256':r.sha(folder/'records.json')}
            manifests.append(item);print(json.dumps(item),flush=True)
            dump(HERE/'reports/dynamics_world_v1/build_progress.json',{'completed_groups':len(manifests),'groups':manifests})
    dump(root/'manifest.json',{'groups':manifests,'episodes':sum(v['episodes'] for v in manifests),
        'counterfactual_episodes':sum(v['counterfactual_episodes'] for v in manifests),
        'unique_initial_orbits':len(orbit_split),'unique_initial_image_orbits':len(image_split),
        'config_sha256':r.sha(HERE/'configs/dynamics_world_v1.json'),'simulator_sha256':r.sha(HERE/'dynamics_world.py'),
        'builder_sha256':r.sha(__file__),'seconds':time.perf_counter()-started})


if __name__=='__main__':main()
