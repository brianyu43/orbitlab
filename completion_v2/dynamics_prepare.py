"""Fresh disjoint trajectory families for the stability confirmation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'completion_v2'),str(ROOT/'generation_recovery'),str(ROOT/'work'),str(ROOT/'followup')]
import numpy as np
import dynamics as study
import dynamics_world as world
from p0 import dump,sha
import research as r

def pad(a):
    out=np.zeros((*a.shape[:-2],4,a.shape[-1]),dtype=a.dtype);out[...,:a.shape[-2],:]=a;return out

def main():
    cfg=study.freeze();forbidden=set();image_orbits=set()
    for root in [ROOT/'followup/data/dynamics_world_v1',ROOT/'followup/data/dynamics_repeats_v1']:
        for path in root.rglob('records.json'):
            for row in json.loads(path.read_text()):
                forbidden.add(row['initial_orbit_sha256']);image_orbits.add(row['initial_image_orbit_sha256'])
    sizes={'train':256,'val':64,'test':128,'attribute_ood':128,'count3':64,'count4':64,'force_ood':128}
    files=[]
    for seed in cfg['confirmation_data_seeds']:
        for mode in study.MODES:
            for split,n in sizes.items():
                if split=='force_ood' and mode!='variable_force':continue
                folder=study.data_root(seed)/mode/split;archive=folder/'trajectories.npz'
                if (folder/'manifest.json').exists():
                    manifest=json.loads((folder/'manifest.json').read_text());assert manifest['archive_sha256']==sha(archive)
                    for row in json.loads((folder/'records.json').read_text()):
                        assert row['physical'] not in forbidden and row['image'] not in image_orbits
                        forbidden.add(row['physical']);image_orbits.add(row['image'])
                    files.append(manifest);continue
                folder.mkdir(parents=True,exist_ok=True);count={'count3':3,'count4':4}.get(split,2);length=20 if split=='train' else 65
                arrays={k:[] for k in ['states','actions','contexts','events','n_objects','target_ids','observations']};records=[]
                counter=[];i=0
                while len(records)<n:
                    state,context,actions,target=world.sample(i,seed,split,mode,count);i+=1
                    ph=world.initial_orbit_hash(state,context,actions[3]);im=r.orbit_hash(world.render(state)['image'])
                    if ph in forbidden or im in image_orbits:continue
                    forbidden.add(ph);image_orbits.add(im);actions=actions[:length-1]
                    states,events=world.rollout(state,context,actions)
                    arrays['states'].append(pad(states));arrays['actions'].append(pad(actions));arrays['contexts'].append(context)
                    arrays['events'].append(events);arrays['n_objects'].append(count);arrays['target_ids'].append(target)
                    arrays['observations'].append(np.stack([world.render(v)['image'] for v in states[:4]]))
                    if split in ['test','force_ood']:
                        cf,_=world.rollout(state,context,-actions);assert np.array_equal(cf[:4],states[:4]);counter.append(pad(cf))
                    records.append({'candidate':i-1,'physical':ph,'image':im})
                saved={k:np.stack(v) for k,v in arrays.items()}
                if counter:saved['counterfactual_states']=np.stack(counter)
                np.savez_compressed(archive,**saved);dump(folder/'records.json',records)
                manifest={'path':str(archive.relative_to(study.BASE)),'seed':seed,'mode':mode,'split':split,'n':n,
                          'archive_sha256':sha(archive),'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(study.BASE/'protocol.json')}
                dump(folder/'manifest.json',manifest);files.append(manifest)
                print('dynamics data',seed,mode,split,n,flush=True)
    path=study.BASE/'data/manifest.json'
    if not path.exists():dump(path,{'groups':files,'cross_prior_and_new_initial_orbit_overlap':0,'source_sha256':sha(Path(__file__))})

if __name__=='__main__':main()
