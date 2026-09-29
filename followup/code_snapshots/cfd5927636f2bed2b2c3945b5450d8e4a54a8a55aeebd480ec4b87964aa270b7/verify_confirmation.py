"""Verify factorial confirmation and report paired, conditional uncertainty."""
import csv
import json
import time
from common import HERE,dump,digest_tensor,np,torch,r


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def conditional_interval(delta,seed):
    """Resample noise within each of the fixed condition cells, not model seeds."""
    rng=np.random.default_rng(seed);draw=rng.integers(len(delta),size=(1000,len(delta),delta.shape[1]))
    boot=delta[draw,np.arange(delta.shape[1])[None,None,:]].mean((1,2))
    return [float(v) for v in np.quantile(boot,[.025,.975])]


def main():
    c=json.loads((HERE/'configs/confirmation_v1.json').read_text());summary=json.loads((HERE/'reports/confirmation_v1/summary.json').read_text())
    root=HERE/'data/confirmation_v1';manifest=json.loads((root/'manifest.json').read_text());hashes=[]
    for f in manifest['files']:assert r.sha(root/f['path'])==f['sha256']
    for ds in c['data_seeds']:
        for split in ['train','val','test','ood']:
            a=np.load(root/str(ds)/f'{split}.npz');meta=json.loads((root/str(ds)/f'{split}_metadata.json').read_text())
            assert len(a['images'])==len(meta)==(1024 if split=='train' else 256)
            assert all((int(k)==int(col))==(split=='ood') for k,col in a['labels'])
            for image,m in zip(a['images'],meta):
                h=r.orbit_hash(image);assert h==m['orbit_sha256'];hashes.append(h)
    assert len(hashes)==len(set(hashes))==5376
    result=[];noise_hashes=set();total=0
    for di,ds in enumerate(c['data_seeds']):
        for seed in c['init_seeds']:
            base=HERE/f'runs/confirmation_v1/d{ds}_s{seed}';ae_run=json.loads((base/'ae/run.json').read_text())
            assert r.sha(base/'ae/ae.pt')==ae_run['checkpoint_sha256']
            assert ae_run['steps']==6000 and ae_run['data_seed']==ds and ae_run['init_seed']==seed
            runs={};arrays={};noise=[]
            for mode in c['modes']:
                folder=base/mode;run=json.loads((folder/'run.json').read_text());m=json.loads((folder/'metrics.json').read_text())
                assert r.sha(folder/'flow.pt')==run['checkpoint_sha256']==m['flow_sha256']
                assert run['ae_sha256']==ae_run['checkpoint_sha256'] and run['ae_frozen_unchanged']
                if mode.endswith('fixed'):assert run['steps']==4000
                else:assert 0<=run['train_wall_seconds']-run['requested_seconds']<.1
                saved=torch.load(folder/'samples.pt',map_location='cpu',weights_only=True)
                assert digest_tensor(saved['noise'])==m['noise_sha256'];noise.append(m['noise_sha256'])
                rr=rows(folder/'samples.csv');assert len(rr)==1920
                assert [int(v['sample']) for v in rr]==list(range(1920))
                arrays[mode]={key:np.array([v[key]=='True' for v in rr],dtype=np.int8).reshape(80,24) for key in ['joint_correct','strict_accepted_and_joint']}
                for split in ['seen','ood']:
                    take=[v for v in rr if (v['ood']=='True')==(split=='ood')]
                    assert len(take)==(1600 if split=='seen' else 320)
                    for key in ['joint_correct','strict_accepted_and_joint']:
                        assert abs(np.mean([v[key]=='True' for v in take])-m[split][key])<1e-12
                runs[mode]=run;total+=1
            assert len(set(noise))==1;noise_hashes.add(noise[0])
            for key in ['initial_raw_weights_sha256','latent_sha256','parameters']:
                assert len({v[key] for v in runs.values()})==1
            for comparison in ['plain_fixed','plain_time']:
                for split in ['seen','ood']:
                    cols=[i for i in range(24) if ((i%4)==(i//4))==(split=='ood')]
                    for key in ['joint_correct','strict_accepted_and_joint']:
                        delta=(arrays['equivariant_fixed'][key]-arrays[comparison][key])[:,cols]
                        result.append({'data_seed':ds,'init_seed':seed,'reference':comparison,'split':split,'metric':key,
                            'paired_difference':float(delta.mean()),'conditional_noise_ci95':conditional_interval(delta,100+10*di+seed)})
    assert total==27 and len(noise_hashes)==9
    grouped={}
    for comparison in ['plain_fixed','plain_time']:
        for split in ['seen','ood']:
            for metric in ['joint_correct','strict_accepted_and_joint']:
                selected=[v for v in result if v['reference']==comparison and v['split']==split and v['metric']==metric]
                means=[float(np.mean([v['paired_difference'] for v in selected if v['data_seed']==ds])) for ds in c['data_seeds']]
                grouped[f'{comparison}/{split}/{metric}']={'mean_difference':float(np.mean(means)),
                    'per_data_seed_mean':means,'positive_cells':sum(v['paired_difference']>0 for v in selected),'cells':len(selected),
                    'between_data_seed_sd':float(np.std(means,ddof=1))}
    report={'all_passed':True,'ae_checkpoints':9,'flow_checkpoints':total,'unique_new_scene_orbits':len(hashes),
        'unique_generation_noise_seeds':len(noise_hashes),'paired_comparisons':result,'factorial_effects':grouped,
        'uncertainty_scope':'Intervals resample noise within fixed shape-color conditions for each trained model pair. Data-seed effects reported separately; only 3 independent data seeds, no population-significance claim.',
        'gate_failures':summary['gate_failures'],'completed_unix':time.time()}
    dump(HERE/'reports/confirmation_v1/verification.json',report)
    from common import update_status
    update_status('A09','complete',['reports/confirmation_v1/summary.json','reports/confirmation_v1/verification.json','data/confirmation_v1/manifest.json'])
    print(json.dumps({'all_passed':True,'effects':grouped,'gate_failures':summary['gate_failures']},indent=2))


if __name__=='__main__':torch.set_num_threads(2);main()
