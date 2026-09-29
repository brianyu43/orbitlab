"""Fresh RGB-only inference followed by separately loaded evaluation labels."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'completion_v2'),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import torch
import perception_detail as p
import object_edit_confirmation as generator
import object_image_edit as edit
from object_oracle_study import image_from_slots
from p0 import dump,sha
import orbitlab as o

def prepare():
    cfg=p.freeze();base=p.BASE
    old_previous=generator.previous_orbits()
    for folder in (ROOT/'followup/data/object_image_edit_v1').iterdir():
        if not folder.is_dir():continue
        old_previous.update(v['orbit_sha256'] for v in json.loads((folder/'scenes.json').read_text()))
        old_previous.update(v['target_orbit_sha256'] for v in json.loads((folder/'tasks.json').read_text()))
    for prior in (ROOT/'completion_v2/perception/data').glob('d*/*/scenes.json'):
        old_previous.update(v['orbit_sha256'] for v in json.loads(prior.read_text()))
        old_previous.update(v['target_orbit_sha256'] for v in json.loads(prior.with_name('tasks.json').read_text()))
    for seed in cfg['confirmation_seeds']:
        name=f'd{seed}';root=base/'data'/name
        if not (root/'manifest.json').exists():
            c={'seed':seed,'evaluation_n':128,'splits':cfg['splits']}
            dest=base/f'configs/{name}.json'
            if not dest.exists():dump(dest,c)
            generator.HERE=base;generator.NAME=name;generator.previous_orbits=lambda:set(old_previous)
            generator.build(c)
        for split in cfg['splits']:
            old_previous.update(v['orbit_sha256'] for v in json.loads((root/split/'scenes.json').read_text()))
            old_previous.update(v['target_orbit_sha256'] for v in json.loads((root/split/'tasks.json').read_text()))

def summarize(rows):
    keys=['end_to_end_success','target_selection_correct','target_success_gt','non_target_correct_gt',
          'image_foreground_mae','changed_pixels_mae','input_passthrough_foreground_mae']
    result={}
    ids=sorted({v['source_index'] for v in rows})
    for key in keys:
        values=[np.mean([v[key] for v in rows if v['source_index']==i and v[key] is not None])
                for i in ids if any(v['source_index']==i and v[key] is not None for v in rows)]
        result[key]=float(np.mean(values)) if values else None
    return result

@torch.no_grad()
def evaluate(kind,init,seed):
    cfg=p.freeze();root=p.BASE/f'data/d{seed}';run=p.BASE/f'runs/{kind}_s{init}'
    ck=torch.load(run/'model.pt',weights_only=True);model=p.Reader();model.load_state_dict(ck['state_dict']);model.eval()
    for split in cfg['splits']:
        data=root/split;out=p.BASE/f'evaluation/d{seed}/{kind}_s{init}/{split}';out.mkdir(parents=True,exist_ok=True)
        if (out/'summary.json').exists():
            rr=json.loads((out/'summary.json').read_text());assert rr['model_sha256']==sha(run/'model.pt') and rr['source_sha256']==sha(Path(__file__));continue
        images=np.load(data/'rgb.npz')['images'].copy()
        # Input phase ends here; no labels are passed to the reader.
        states=p.infer(images,model,kind)
        q=np.load(data/'queries.npz');idx=q['source_indices'];commands=torch.from_numpy(q['commands'].copy())
        clicks=torch.from_numpy(q['click_xy'].copy());source=states[idx];selected=edit.select_target(source,clicks)
        torch.save({'states':states,'selected':selected},out/'inference.pt')
        truth=torch.from_numpy(np.load(data/'scene_labels.npz')['states'].copy())[idx]
        target=np.load(data/'targets.npz');targets=torch.from_numpy(target['states'].copy());target_images=target['images'].copy()
        records=json.loads((data/'tasks.json').read_text());results={}
        for name,controller in [('analytic','analytic'),('learned',edit.load_controller(init))]:
            pred=edit.apply_commands(source,selected,commands,controller)
            rendered=np.stack([image_from_slots(v) for v in pred])
            rows,_=edit.score(source,pred,selected,commands,truth,targets,records,rendered,target_images,images[idx],torch.zeros(len(idx),dtype=torch.long))
            dump(out/f'{name}_rows.json',rows);torch.save(pred,out/f'{name}_states.pt')
            np.savez_compressed(out/f'{name}_images.npz',images=rendered)
            results[name]=summarize(rows)
            if name=='analytic':
                examples=np.stack([im for i in range(8) for im in (images[idx[i]],target_images[i],rendered[i])])
                o.grid(torch.from_numpy(examples).permute(0,3,1,2).float()/255,out/'examples.png',6)
        dump(out/'summary.json',{'kind':kind,'initialization':init,'data_seed':seed,'split':split,'metrics':results,
                                'model_sha256':sha(run/'model.pt'),'source_sha256':sha(Path(__file__)),
                                'input_sha256':sha(data/'rgb.npz'),'files':{v.name:sha(v) for v in out.iterdir() if v.is_file()},
                                'scope':'Known-palette RGB masks and supervised readout; full four-way pose metric with full four-way training supervision. Shared training data across evaluation seeds.'})
        print('perception score',kind,init,seed,split,results,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','evaluate']);parser.add_argument('--kind',choices=['binary','alpha'],default='alpha')
    parser.add_argument('--init',type=int,default=0);parser.add_argument('--seed',type=int,default=885201);a=parser.parse_args();torch.set_num_threads(2)
    if a.stage=='prepare':prepare()
    else:evaluate(a.kind,a.init,a.seed)
